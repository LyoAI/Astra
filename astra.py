import os
import math
import logging
import warnings
from collections.abc import Iterable
from typing import Any, Callable, Optional, Union, Dict
import torch
import torch.nn as nn
import torch.distributed as dist
from attr import dataclass
from tqdm import tqdm
import numpy as np

from peft.tuners.lora import LoraLayer
from peft.tuners.lora.config import LoraConfig
from peft.tuners.lora.model import LoraModel
from peft.utils.other import get_pattern_key, transpose
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils.integrations import gather_params_ctx
from astra_config import MyLoraConfig

logger = logging.getLogger(__name__)

@dataclass
class AstraEigens:
    S: torch.Tensor
    V: torch.Tensor

def setup_logger(log_file=None):
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    if log_file:
        handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def target_modules(model: nn.Module, config: LoraConfig) -> Iterable[nn.Module]:
    """
    Iterate over Astra target name and modules of a model. A module is a target if its name is in
    `config.target_modules` and is `nn.Linear`.
    """
    for name, module in model.named_modules():
        if LoraModel._check_target_module_exists(config, name) and isinstance(module, nn.Linear):
            yield name, module

def get_model_device(model: nn.Module) -> str:
    if hasattr(model, "module"):  # Handle DeepSpeed/DataParallel
        model = model.module
    return next(iter(model.parameters())).device.type

def effective_rank_fn(eigenvalues: torch.Tensor):
    eigenvalues = eigenvalues.to(dtype=torch.float32)
    phi_bar = eigenvalues / eigenvalues.sum()

    epsilon = 1e-8
    entropy = -torch.sum(phi_bar * torch.log(phi_bar + epsilon))

    R = torch.exp(entropy)
    return R.cpu().item()

def stable_effective_rank_fn(eigenvalues: torch.Tensor, epsilon: float = 1e-8):
    eigenvalues = eigenvalues.to(dtype=torch.float32)
    eigenvalues = torch.clamp(eigenvalues, min=epsilon)
    total_sum = eigenvalues.sum()
    
    if total_sum == 0:
        return 0.0
    
    phi_bar = eigenvalues / total_sum
    
    entropy = -torch.sum(phi_bar * torch.log(phi_bar + epsilon))
    
    R = torch.exp(entropy)
    return R.cpu().item()


def allocate_fn(
    effective_rank_scores: Dict[str, float],
    base_rank: int,
    min_rank: int = 1,
    max_rank: Optional[int] = None
):
    """
    LoRA rank allocation based on the given effective rank scores.
    
    Args:
        effective_rank_scores: Mapping of layer names to raw importance scores.
        base_rank: LoRA predefined per-layer rank; total budget = num_layers * base_rank.
        min_rank: Minimum rank per layer (default: 1).
        max_rank: Maximum rank per layer (default: None).
        
    Returns:
        Mapping of layer names to allocated integer ranks summing exactly to budget.
    """
    # Step 1: Set max_rank if not provided
    if max_rank is None:
        max_rank = max(2 * base_rank, 256)

    # Step 2: Log-normalize the scores (log(score + 1) to avoid negative or zero values)
    normalized_scores = {layer: np.log(score + 1) for layer, score in effective_rank_scores.items()}
    
    # Step 3: Calculate the total normalized score
    total_score = sum(normalized_scores.values())

    # Step 4: Calculate the total budget
    total_budget = len(effective_rank_scores) * base_rank

    # Step 5: Allocate ranks based on normalized scores
    allocated_ranks = {}
    for layer, score in normalized_scores.items():
        # Allocation proportional to normalized score, capped at max_rank
        rank = int(round(score / total_score * total_budget))

        # Ensure rank is within bounds: min_rank and max_rank
        rank = max(min_rank, min(rank, max_rank))

        # Store the allocated rank for the layer
        allocated_ranks[layer] = rank

    # Step 6: Adjust ranks to ensure total rank sum equals total_budget
    current_total = sum(allocated_ranks.values())
    rank_diff = total_budget - current_total

    # Step 7: Adjust ranks to match the total budget
    if rank_diff != 0:
        # Sort layers by score to determine which layers to increase/decrease
        sorted_layers = sorted(allocated_ranks.items(), key=lambda x: normalized_scores[x[0]], reverse=True)
        for i in range(abs(rank_diff)):
            layer, rank = sorted_layers[i % len(sorted_layers)]
            if rank_diff > 0:  # If we need to increase the total rank
                if allocated_ranks[layer] < max_rank:
                    allocated_ranks[layer] += 1
            else:  # If we need to decrease the total rank
                if allocated_ranks[layer] > min_rank:
                    allocated_ranks[layer] -= 1

    return allocated_ranks


@torch.no_grad()
def preprocess_astra(
    model: nn.Module,
    lora_config: MyLoraConfig,
    run_model: Optional[Callable[[], None]] = None,
    hooked_model: Optional[nn.Module] = None,
    log_file: Optional[str] = None
):
    """
    Build necessary Astra fields for a model.

    For each `M * N` linear layer, a `N * N` covariance matrix will be built temporarily during the preprocessing
    process, consuming roughly another `2 * MODEL_SIZE` memory for typical LLMs if model weight is FP16 and covariance
    is FP32. If that's too much, consider specifying `use_float16_for_covariance` in `lora_config.astra_config`.

    Args:
        model (`nn.Module`):
            Model to preprocess.
        lora_config (`LoraConfig`):
            Lora configuration of the model. `lora_config.astra_config` should be set.
        run_model (`Optional[Callable[[], None]]`):
            Callback to run the model when building covariance. Typically you should run model inference on your sample
            dataset in this callback. Experiments have shown that when token count per sample is 2048, hidden dimension
            is 4096, collecting 256 distinct samples is enough. If you collect too few or too repetitive samples, the
            covariance matrix may be low-ranked and unstabilize preprocessing. You can estimate sample count as
            `HIDDEN_DIM / TOKEN_PER_SAMPLE * 128`. `run_model` can be `None` only if covariance file in
            `lora_config.astra_config` is already created.
        hooked_model (`Optional[nn.Module]`):
            Model to hook when building covariance. If none, original model will be hooked. This is only useful when
            you want to hook a different model than the one you are training, typically you should leave this `None`.

    Upon completion, the following fields are set for each target module:
        eigens.S (`torch.Tensor`):
            eigenvalue of the collected output activations' covariance matrix.
        eigens.V (`torch.Tensor`):
            eigenvectors of the collected output activactions' covariance matrix
    """
    cache_file = lora_config.astra_config.cache_file
    covariance_file = lora_config.astra_config.covariance_file
    astra_method = lora_config.astra_config.astra_method
    verbose = lora_config.astra_config.verbose
    prune_temporary_fields = lora_config.astra_config.prune_temporary_fields

    # If cache exists, skip building
    if cache_file is not None and os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        cache = torch.load(cache_file, map_location=get_model_device(model))
        for name, module in target_modules(model, lora_config):
            module.eigens = AstraEigens(
                S=cache[f"{name}.eigens.S"],
                V=cache[f"{name}.eigens.V"],
            )
    else:
        if astra_method is None:
                raise ValueError("astra_method is required when cache_file is not provided.")
        for name, module in target_modules(model, lora_config):
            module.astra_method = astra_method

        # Calculate covariance matrix
        calib_cov_distribution(model, lora_config, run_model, hooked_model, covariance_file, log_file)

        # Calculate eigens
        collect_eigens(model, lora_config, verbose, log_file)

        # Crop Astra eigens so that there's less to save
        crop_astra_eigens(model, lora_config, log_file)

        # Remove redundant fields if exist
        if prune_temporary_fields:
            for name, module in target_modules(model, lora_config):
                if hasattr(module, "sample_count"):
                    del module.sample_count
                if hasattr(module, "covariance_matrix"):
                    del module.covariance_matrix
                if hasattr(module, "astra_method"):
                    del module.astra_method
                if hasattr(module, "rank"):
                    del module.rank

        # Save cache to disk
        if cache_file is not None:
            cache: dict[str, Any] = {}
            for name, module in target_modules(model, lora_config):
                cache[f"{name}.eigens.S"] = module.eigens.S
                cache[f"{name}.eigens.V"] = module.eigens.V

            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            torch.save(cache, cache_file)


@torch.no_grad()
def calib_cov_distribution(
    model: nn.Module,
    config: LoraConfig,
    run_model: Optional[Callable[[], None]],
    hooked_model: Optional[nn.Module],
    covariance_file: Optional[str],
    log_file: Optional[str]
):
    if covariance_file is not None and os.path.exists(covariance_file) and os.path.getsize(covariance_file) > 0:
        all_covariance_matrix = torch.load(covariance_file, map_location=get_model_device(model), weights_only=False)
        for name, module in target_modules(model, config):
            module.covariance_matrix = all_covariance_matrix[name]
        return

    if run_model is None:
        raise ValueError("run_model must be specified when covariance file and cache file aren't built.")
    if hooked_model is None:
        hooked_model = model
    hooked_model.eval()

    def hook(module, input, output):
        output = output[0].detach().squeeze(0).data  ## (context_length = 2048, dim)
        if not config.astra_config.use_float16_for_covariance:
            output = output.float()
        output = output / torch.max(output).abs()

        # check if output is valid
        if torch.isnan(output).any() or torch.isinf(output).any():
            raise ValueError("Invalid value found in output, please check your output data.")

        # calculate covariance and check if it's valid
        covariance = output.t().matmul(output)
        if torch.isnan(covariance).any() or torch.isinf(covariance).any():
            raise ValueError()

        # add to module
        module.sample_count += 1
        module.covariance_matrix += covariance

        # free memory
        del covariance, output

    handles = []
    for name, module in target_modules(hooked_model, config):
        module.sample_count = 0
        module.covariance_matrix = 0
        handles.append(module.register_forward_hook(hook))

    run_model()

    # Clear the hooks
    for handle in handles:
        handle.remove()

    # In some edge cases you might need to hook a model different from the model to add adapters,
    # this case you would specify `hooked_model` and set it to a different model from `model`.
    if hooked_model is not model:
        targets = {}
        for name, module in target_modules(model, config):
            targets[name] = module
        for name, module in target_modules(hooked_model, config):
            # There can be modules used only in inference, but not training
            # Exclude modules not in target model to prevent KeyError in this case
            if name in targets:
                targets[name].sample_count = module.sample_count
                targets[name].covariance_matrix = module.covariance_matrix

    # Divide by sample count
    for name, module in target_modules(model, config):
        module.covariance_matrix /= module.sample_count

    # Save covariance to disk
    if covariance_file is not None:
        all_covariance_matrix = {}
        for name, module in target_modules(model, config):
            all_covariance_matrix[name] = module.covariance_matrix
        os.makedirs(os.path.dirname(covariance_file), exist_ok=True)
        torch.save(all_covariance_matrix, covariance_file)


@torch.no_grad()
def collect_eigens(
    model: nn.Module,
    config: LoraConfig,
    verbose: bool,
    log_file: Optional[str]
):
    """Call collect_eigens_for_layer and store result in key `eigens` of each layer."""
    linear_modules = []
    for name, module in target_modules(model, config):
        linear_modules.append((name, module))
    if verbose:
        linear_modules = tqdm(linear_modules, desc="Collecting eigens")
    for name, module in linear_modules:
        module.eigens, module.dim = collect_eigens_for_layer(module, config, log_file)


@torch.no_grad()
def collect_eigens_for_layer(
    linear: nn.Linear,
    config: LoraConfig,
    log_file: Optional[str]
) -> AstraEigens:
    if not hasattr(linear, "covariance_matrix"):
        raise ValueError(
            "Covariance matrix not found in linear module. Please do not call this function directly, "
            "instead call `preprocess_astra`. If your usage is correct but this error still encounters, "
        )
    covariance_matrix = linear.covariance_matrix.float()
    dim = covariance_matrix.size(0)
    S, V = torch.linalg.eigh(covariance_matrix)

    # Sanity check, temporarily U and V are large, they will be crop after rank search
    if S.size(0) != dim:
        raise ValueError(
            f"Matrix S size mismatch: {S.size()} vs. ({dim},), "
        )
    if V.size(0) != dim or V.size(1) != dim:
        raise ValueError(
            f"Matrix V size mismatch: {V.size()} vs. ({dim}, {dim}), "
        )

    # Offload U and V to CPU, they consume too much memory
    S = S.cpu()
    V = V.cpu()
    return AstraEigens(S=S, V=V), dim


@torch.no_grad()
def crop_astra_eigens(model: nn.Module, config: MyLoraConfig, log_file: Optional[str]):
    rank_allocation = config.astra_config.rank_allocation
    rank_pattern = config.astra_config.rank_pattern

    setup_logger(log_file)

    if rank_allocation and rank_pattern is not None and os.path.exists(rank_pattern) and os.path.getsize(rank_pattern) > 0:
        logger.info(f"Load rank pattern from cache {rank_pattern}...")
        rank_pattern = torch.load(rank_pattern, weights_only=False)
    elif rank_allocation and (rank_pattern is None or not os.path.exists(rank_pattern)):
        rank_pattern = {}
        effective_rank_scores = {}
        for name, module in target_modules(model, config):
            S = module.eigens.S
            effective_rank_scores[name] = stable_effective_rank_fn(eigenvalues=S)
        logger.info(effective_rank_scores)
        rank_pattern = allocate_fn(effective_rank_scores, config.r)
        torch.save(rank_pattern, "cache/rank_pattern.pt")
    else:
        rank_pattern = {}
    config.rank_pattern = rank_pattern

    for name, module in target_modules(model, config):
        r_key = get_pattern_key(config.rank_pattern.keys(), name)
        module.rank = config.rank_pattern.get(r_key, config.r)

    for name, module in target_modules(model, config):
        module.eigens.S = module.eigens.S.clone()
        if module.astra_method.lower().strip() == "ipm":
            module.eigens.V = module.eigens.V[:, -module.rank:].clone().to(get_model_device(model))
        elif module.astra_method.lower().strip() == "kpm":
            module.eigens.V = module.eigens.V[:, :module.rank].clone().to(get_model_device(model))
        else:
            module.eigens.V = module.eigens.V[:, module.dim // 2: module.dim // 2 + module.rank].clone().to(get_model_device(model))
        # Sanity check
        if module.eigens.V.size(0) != module.dim:
            raise ValueError(
                f"In {name} module, Matrix V size mismatch: {module.eigens.V.size(0)} vs. ({module.dim},)"
            )
        if module.eigens.V.size(1) != module.rank:
            raise ValueError(
                f"In {name} module, Matrix V size mismatch: {module.eigens.V.size(1)} vs. ({module.rank},)"
            )


class AstraLayer(nn.Module, LoraLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = "astra",
        use_rslora: bool = False,
        **kwargs
    ):
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
    
    
    def update_layer(
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, lora_bias: bool = False
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=lora_bias)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r
        
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("astra"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.astra_init(adapter_name, init_lora_weights)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name=adapter_name, init_lora_weights=init_lora_weights)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        
        self.set_adapter(self.active_adapters)
    
    def astra_init(self, adapter_name, init_lora_weights):
        linear = self.get_base_layer()
        weight = linear.weight
        dtype = weight.dtype
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError(
                "Please initialize astra under float32, float16, or bfloat16. "
                "Subsequently, re-quantize the residual model to help minimize quantization errors."
            )
        weight = weight.to(torch.float32)
        out_dim = weight.data.size(0)
        in_dim = weight.data.size(1)

        if not hasattr(linear, "eigens"):
            raise ValueError(
                "`eigens` attribute not found for layer, please run `preprocess_astra` first. "
            )
        eigens = linear.eigens
        V = eigens.V
        r = self.r[adapter_name]

        if torch.isnan(V).any() or torch.isinf(V).any():
            raise ValueError()

        if V.size(0) != out_dim or V.size(1) != r:
            raise ValueError(
            f"Matrix V size mismatch: {V.size()} vs. ({out_dim}, {r}). Please make sure the `lora_config` and "
            "`model` argument of `preprocess_astra` is consistent with `get_peft_model`. If you're using cache "
            "in `preprocess_astra`, please make sure the cache is built with the same model and LoRA rank."
        )
        # Init lora_A and lora_B weights
        lora_A = (V.t() @ weight).contiguous().to(dtype)
        lora_B = V.contiguous().to(dtype)
        self.lora_A[adapter_name].weight.data = lora_A
        self.lora_B[adapter_name].weight.data = lora_B
        weight = weight.data - self.scaling[adapter_name] * lora_B @ lora_A
        weight = weight.to(dtype)
        self.get_base_layer().weight.data = weight

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        orig_weights += delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(orig_weights, transpose(delta_weight, self.fan_in_fan_out), scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        orig_weights = dora_factor * (orig_weights + delta_weight)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data += delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(
                                base_layer.weight, transpose(delta_weight, self.fan_in_fan_out), scaling=1
                            )
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        new_weight = dora_factor * (base_layer.weight.data + delta_weight)
                        base_layer.weight.data = new_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    weight_orig = weight.data / dora_factor.view(-1, 1) - delta_weight
                    weight.data = weight_orig

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    x = dropout(x)
                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                    )

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "astra." + rep

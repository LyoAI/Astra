from peft.tuners.lora.config import LoraConfig
from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "AstraConfig",
    "MyLoraConfig"
]


@dataclass
class AstraConfig:
    """
    This is the sub-configuration class to store the configuration of a [`LoraModel`] for Astra.

    Args:
        cache_file (`Optional[str]`):
            File to store the SVD cache. The SVD cache is much smaller than the residual model (for example, residual
            model of Llama-3-8b is 15GB, while SVD cache is 1.4GB), but with SVD cache and original model weights,
            residual model weights can be built quickly. If you need to reuse residual model weights with limited
            storage, you can store the SVD cache instead.
        covariance_file (`Optional[str]`):
            File to store the covariance matrix. If you wish to train multiple models with different ranks, but they
            sample from the same dataset, you can store the covariance matrix and reuse it for different ranks. Note
            that covariance file is usually large (comparable to model size), so you will need sufficient storage.
        verbose (`bool`):
            If true, prints the progress of Astra initialization. Defaults to `False`.
        use_float16_for_covariance (`bool`):
            If true, uses float16 for the covariance matrix. This can reduce the memory usage of the covariance matrix
            by half, but may lead to numerical instability. Defaults to `False`.
        prune_temporary_fields (`bool`):
            If true, temporary fields generated in Astra preprocessing will be pruned. Defaults to `True`.
        rank_allocation (`bool`):
            Whether to perform dynamic rank allocation. If True, dynamic rank allocation will be performed during preprocessing.
            Defaults to False.
        rank_pattern (`Optional[str]`):
            Path to cache file for dynamic rank allocation results. If specified, dynamic rank allocation results will be loaded/saved here.
            Defaults to None.
        astra_method (`str`):
            The method used for Astra. 'IPM' stands for Instuction-Previewed Mode, which focusing on adapting to downstream tasks.
            'KPM' stands for Knowledge-Previewed Mode, which focusing on adapting to downstream tasks.
            Defaults to 'IPM'.
    """
    cache_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "File to store the SVD cache. The SVD cache is much smaller than the residual model (for example, "
                "residual model of Llama-3-8b is 15GB, while SVD cache is 1.4GB), but with SVD cache and original model "
                "weights, residual model weights can be built quickly. If you need to reuse residual model weights with "
                "limited storage, you can store the SVD cache instead."
            )
        },
    )
    covariance_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "File to store the covariance matrix. If you wish to train multiple models with different ranks, but "
                "they sample from the same dataset, you can store the covariance matrix and reuse it for different ranks. "
                "Note that covariance file is usually large (comparable to model size), so you will need sufficient storage."
            )
        },
    )
    astra_method: str = field(
        default="IPM",
        metadata={
            "help": (
                "The method used for Astra. 'IPM' stands for Instuction-Previewed Mode, which focusing on adapting to downstream tasks. "
                "'KPM' stands for Knowledge-Previewed Mode, which focusing on preserving the original knowledge of the model when adapting to downstream tasks."
            )
        },
    )
    verbose: bool = field(default=False, metadata={"help": "If true, prints the progress of Astra initialization."})
    use_float16_for_covariance: bool = field(
        default=False,
        metadata={
            "help": (
                "If true, uses float16 for the covariance matrix. This can reduce the memory usage of the covariance matrix "
                "by half, but may lead to numerical instability."
            )
        },
    )
    prune_temporary_fields: bool = field(
        default=True, metadata={"help": "If true, temporary fields generated in Astra preprocessing will be pruned."}
    )
    rank_allocation: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to perform dynamic rank allocation. If True, dynamic rank allocation will be performed during preprocessing. "
                "Defaults to False."
            )
        },
    )
    rank_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to cache file for dynamic rank allocation results. If specified, dynamic rank allocation results will be loaded/saved here. "
                "Defaults to None."
            )
        },
    )

@dataclass
class MyLoraConfig(LoraConfig):
    astra_config: Optional[AstraConfig] = field(
        default=None,
        metadata={
            "help": (
                "The configuration of Astra. If this is passed, then Astra will be used to build the adapter layers. "
                "Also set `init_lora_weights='Astra'` in this case."
            )
        },
    )
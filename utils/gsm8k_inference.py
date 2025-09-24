import argparse
import logging
import json
import re
import jsonlines
from fraction import Fraction
from vllm import LLM, SamplingParams
import sys
from grader import math_equal
from setproctitle import setproctitle

setproctitle("GSM8K-Evaluate")

logger = logging.getLogger(__name__)

def setup_logger(log_file=None):
    logger.setLevel(logging.INFO)
    # Remove existing handlers to avoid duplicate logs
    logger.handlers = []
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

MAX_INT = sys.maxsize

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def extract_answer_number(completion):
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data


def gsm8k_test(model, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1, log_file=None):
    setup_logger(log_file)
    INVALID_ANS = "[invalid]"
    gsm8k_ins = []
    gsm8k_answers = []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    logger.info('promt ===== %s', problem_prompt)
    with open(data_path,"r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["question"])
            gsm8k_ins.append(temp_instr)
            temp_ans = item['answer'].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            gsm8k_answers.append(temp_ans)

    gsm8k_ins = gsm8k_ins[start:end]
    gsm8k_answers = gsm8k_answers[start:end]
    logger.info('lenght ==== %d', len(gsm8k_ins))
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=batch_size)

    stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=1024, stop=stop_tokens)
    logger.info('sampleing ===== %s', sampling_params)
    llm = LLM(model=model,tensor_parallel_size=tensor_parallel_size, dtype="float16")
    result = []
    res_completions = []
    for idx, (prompt, prompt_answer) in enumerate(zip(batch_gsm8k_ins, gsm8k_answers)):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]

        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    invalid_outputs = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(gsm8k_ins, res_completions, gsm8k_answers)):
        doc = {'question': prompt}
        y_pred = extract_answer_number(completion)
        if y_pred != None:
            result.append(float(y_pred) == float(prompt_answer) or math_equal(y_pred, prompt_answer))
        else:
            result.append(False)
            temp = {'question': prompt, 'output': completion, 'answer': prompt_answer}
            invalid_outputs.append(temp)
    acc = sum(result) / len(result)
    logger.info('len invalid outputs ==== %d, valid_outputs=== %s', len(invalid_outputs), invalid_outputs)
    logger.info('start=== %d, end==== %d', start, end)
    logger.info('gsm8k length==== %d, gsm8k acc==== %f', len(result), acc)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)  # model path
    parser.add_argument("--data_file", type=str, default='dataset/gsm8k_test.jsonl')  # data path
    parser.add_argument("--start", type=int, default=0) #start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=60)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=2)  # tensor_parallel_size
    parser.add_argument("--log_file", type=str, default=None, help="Log file path")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    gsm8k_test(model=args.model, data_path=args.data_file, start=args.start, end=args.end, batch_size=args.batch_size, tensor_parallel_size=args.tensor_parallel_size, log_file=args.log_file)
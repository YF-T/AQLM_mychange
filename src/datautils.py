import os
import random
from itertools import chain
from typing import Optional, Sequence
import json

import numpy as np
import torch
import torch.distributed
from datasets import load_dataset
from torch import nn
from tqdm import trange
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F


def set_seed(seed: Optional[int]):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_red_pajama(nsamples, seqlen, tokenizer, eval_mode=False):
    print("Loading red_pajama from togethercomputer/RedPajama-Data-1T-Sample")
    assert not eval_mode, "Only train set is supported in RedPajama"
    traindata = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train")
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    trainloader = []
    for _ in trange(nsamples, desc="Making red_pajama calibration set", leave=False):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        assert inp.shape[1] == seqlen
        trainloader.append(inp)
    return trainloader


def get_wikitext2(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            trainloader.append(inp)
        return trainloader
    else:
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc


def get_ptb(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
        trainenc = tokenizer("\n\n".join(traindata["sentence"]), return_tensors="pt")
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            trainloader.append(inp)
        return trainloader
    else:
        valdata = load_dataset("ptb_text_only", "penn_treebank", split="validation")
        testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")
    return testenc


def get_c4(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        traindata = load_dataset(
            "allenai/c4",
            "default",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        )
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            trainloader.append(inp)
        return trainloader

    else:
        valdata = load_dataset(
            "allenai/c4",
            "default",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        )
        random.seed(0)
        valenc = []
        for _ in range(256):
            while True:
                i = random.randint(0, len(valdata) - 1)
                tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
                if tmp.input_ids.shape[1] >= seqlen:
                    break
            if tmp.input_ids.shape[1] == seqlen:
                # rare case, discovered with Yi tokenizer
                valenc.append(tmp.input_ids)
            else:
                i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
                j = i + seqlen
                valenc.append(tmp.input_ids[:, i:j])
        valenc = torch.hstack(valenc)
        return valenc


def get_ptb_new(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
        trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            trainloader.append(inp)
        return trainloader
    else:
        testdata = load_dataset("ptb_text_only", "penn_treebank", split="test")
        testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")
        return testenc


def get_c4_new(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        traindata = load_dataset(
            "allenai/c4",
            "default",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        )
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            trainloader.append(inp)
        return trainloader
    else:
        valdata = load_dataset(
            "allenai/c4",
            "default",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        )
        valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
        valenc = valenc.input_ids[:, : (256 * seqlen)]
        return valenc


def get_openmathreasoning(nsamples, seqlen, tokenizer, model_path, trust_remote_code):
    """
    Loads and processes the nvidia/OpenMathReasoning dataset for calibration,
    and computes entropy on the fly.
    """
    print("Loading and processing dataset from nvidia/OpenMathReasoning...")
    dataset = load_dataset("nvidia/OpenMathReasoning", split="train")
    
    # Load a separate model instance for entropy calculation
    print(f"Loading model for entropy calculation: {model_path}")
    entropy_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=trust_remote_code).to("cuda" if torch.cuda.is_available() else "cpu")
    entropy_model.eval()

    datalist = []
    for _ in trange(nsamples, desc="Building OpenMathReasoning calibration set with entropy", leave=False):
        while True:
            i = random.randint(0, len(dataset) - 1)
            sample = dataset[i]
            question = sample.get('question')
            solution_dict = sample.get('solution')

            if not question or not solution_dict or not solution_dict.get('generated_solution'):
                continue
            
            answer = solution_dict['generated_solution']
            messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
            tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt")

            if tokenized_chat.shape[1] >= seqlen:
                inp = tokenized_chat[:, :seqlen]
                
                with torch.no_grad():
                    outputs = entropy_model(inp.to(entropy_model.device))
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=-1)
                    entropies = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
                
                datalist.append({'input_ids': inp.cpu(), 'entropies': entropies.cpu()})
                break
    del entropy_model
    torch.cuda.empty_cache()
    return datalist

def get_openmathreasoning_prepared(nsamples, seqlen, filepath, **kwargs):
    """从预处理的jsonl文件中加载数据"""
    data = []
    print(f"Loading prepared dataset from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if len(data) >= nsamples:
                break
            record = json.loads(line)
            input_ids = torch.tensor(record['input_ids'], dtype=torch.long).unsqueeze(0)
            entropies = torch.tensor(record['entropies'], dtype=torch.float32).unsqueeze(0)

            if input_ids.shape[1] >= seqlen:
                input_ids = input_ids[:, :seqlen]
                entropies = entropies[:, :seqlen]
                data.append({'input_ids': input_ids, 'entropies': entropies})
    return data


def get_loaders(
    name,
    nsamples=128,
    seed=0,
    seqlen=2048,
    eval_mode=False,
    model_path=None,
    use_fast_tokenizer=False,
    trust_remote_code=None,
    # --- 新增参数 ---
    use_entropy=False,
    entropy_thresholds: Optional[dict] = None, 
    weight_from_entropy: bool = False,
    weight_hyperparam_C: float = 2.0,
):
    set_seed(seed)

    if use_entropy:
        if eval_mode:
            raise ValueError("Entropy-based quantization is only supported for calibration (non-eval mode).")
        is_prepared = "openmathreasoning_prepared" in name and os.path.isfile(name)
        is_live = name.lower() == "openmathreasoning"
        if not is_prepared and not is_live:
            raise ValueError("Entropy-based quantization currently only supports 'openmathreasoning' or a prepared file path containing 'openmathreasoning_prepared'.")

    # --- Data Loading ---
    if name.lower() == "none":
        print("Not loading any dataset.")
        return None
    
    data_with_entropy = []
    if use_entropy:
        if "openmathreasoning_prepared" in name and os.path.isfile(name):
            data_with_entropy = get_openmathreasoning_prepared(nsamples, seqlen, name)
        elif name.lower() == "openmathreasoning":
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast_tokenizer, trust_remote_code=trust_remote_code)
            data_with_entropy = get_openmathreasoning(nsamples, seqlen, tokenizer, model_path, trust_remote_code)
    
    elif os.path.isfile(name):
        try:
            data = torch.load(name)[:nsamples]
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Failed to load custom data from {name}.")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast_tokenizer, trust_remote_code=trust_remote_code)
        if name.lower() == "wikitext2":
            data = get_wikitext2(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "pajama":
            data = get_red_pajama(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "ptb":
            data = get_ptb(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "ptb_new":
            data = get_ptb_new(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "c4":
            data = get_c4(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "c4_new":
            data = get_c4_new(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        else:
            raise ValueError(f"Unknown dataset {name}")
        return data

    # --- Post-processing for entropy-based methods ---
    if use_entropy:
        processed_data = []
        for sample in data_with_entropy:
            input_ids = sample['input_ids']
            entropies = sample['entropies']
            
            # 1. Generate mask
            final_mask = torch.ones_like(entropies, dtype=torch.bool)
            if entropy_thresholds:
                if 'fixed' in entropy_thresholds:
                    final_mask &= (entropies > entropy_thresholds['fixed'])
                if 'percentile' in entropy_thresholds:
                    quantile_value = torch.quantile(entropies.to(torch.float32), entropy_thresholds['percentile'])
                    final_mask &= (entropies >= quantile_value)
            
            # 2. Generate weight
            weight = torch.ones_like(entropies, dtype=torch.float32)
            if weight_from_entropy:
                min_e = entropies.min()
                max_e = entropies.max()
                denom = max_e - min_e + 1e-9
                normalized_entropies = (entropies - min_e) / denom
                weight = 1.0 + (weight_hyperparam_C - 1.0) * normalized_entropies

            processed_data.append((input_ids, final_mask, weight))
        
        print(f"Loaded and processed {len(processed_data)} samples with entropy data.")
        return processed_data

    raise RuntimeError("Should not reach here in get_loaders")


def split_long_texts(inputs: Sequence[str], split_max_length: int):
    """Split examples that exceed split_max_length into multiple sub-examples"""
    outputs = []
    for index, input_str in enumerate(inputs):
        while True:
            truncation_index = input_str.find("\n", split_max_length)
            if truncation_index == -1:
                outputs.append(input_str)
                break
            outputs.append(input_str[:truncation_index])
            input_str = input_str[truncation_index + 1 :]  # continue after \n
    return outputs


def group_texts(examples: Sequence[Sequence[int]], block_size: int, add_labels: bool = True):
    """Group tokenized examples together and split them into blocks of up to block_size tokens"""
    # based on https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated_examples.items()
    }
    if add_labels:
        result["labels"] = result["input_ids"].copy()
    return result


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module, data: torch.Tensor, seqlen: int, device: torch.device, amp_dtype: Optional[torch.dtype] = None
) -> float:
    """Perplexity evaluation as per https://github.com/IST-DASLab/gptq (standard among quantization research)"""
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    inps = [
        data[:, start : start + seqlen] for start in range(0, data.shape[1], seqlen) if start + seqlen < data.shape[1]
    ]  # ignore last incomplete sequence as in the GPTQ paper
    num_sequences_without_padding = len(inps)

    # pad sequences to be divisible by world_size for DDP/FSDP compatibility
    num_padding_sequences = -len(inps) % world_size
    inps.extend([inps[-1]] * num_padding_sequences)

    total_nll_and_tokens = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
    total_nll, total_tokens = total_nll_and_tokens[0], total_nll_and_tokens[1]

    for sequence_index, input_ids in enumerate(tqdm(inps, desc="Evaluating perplexity") if rank == 0 else inps):
        if sequence_index % world_size != rank:
            continue
        input_ids = input_ids.to(device)
        with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype or torch.float32):
            lm_logits = model(input_ids).logits

        if sequence_index < num_sequences_without_padding:
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_nll += loss.float() * shift_labels.numel()
            total_tokens += shift_labels.numel()

    if world_size > 1:
        torch.distributed.all_reduce(total_nll_and_tokens, op=torch.distributed.ReduceOp.SUM)
    ppl = torch.exp(total_nll / total_tokens)
    return ppl.item()
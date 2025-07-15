# . 项目代码分析与开发规划

## 1. 项目概述与开发目标

这是AQLM量化方法的官方代码仓库。我希望在其基础上，探索一种更适用于长文本、长代码等长推理场景的量化新方法。

该方法的核心思路是进行选择性量化，即识别并**仅仅针对包含信息量更丰富的高熵（high-entropy）token**进行复杂的量化计算，从而在保证模型性能的同时，优化计算效率。为了实现这一目标，我们需要对校准数据的处理方式进行针对性调整。

**现有的方法处理数据集**时，主要采用两种策略：一种是将多个文本拼接成一个长序列后，再进行随机切片 [cite: 45-53, 60-68]；另一种则是从众多文档中逐个随机采样，直到找到足够长的文档再进行切片 [cite: 20-36, 90-106]。

在我的新探索中，**我不希望拼接数据集**，因为这可能无法精确模拟特定任务（如代码生成）的输入分布。我希望直接使用更具针对性的数据，例如来自 **nvidia/OpenMathReasoning** 数据集。具体来说，我计划**直接使用模型生成的解答（`generated_solution`）的一部分**作为校准数据。这种方式不仅能保证数据内容与目标任务的高度相关性，也更符合我对长代码或长逻辑链进行推理量化的最终目标。

## 2. 原始项目目录结构

```
./
├── LICENSE
├── README.md
├── __init__.py
├── aq_engine.py
├── convert_legacy_model_format.py
├── convert_to_hf.py
├── finetune.py
├── lmeval.py
├── main.py
├── pyproject.toml
└── requirements.txt
├── benchmark/
│   │   ├── benchmark_generate_cpu.py
│   │   ├── generate_benchmark.py
│   │   ├── matmul_benchmark.py
│   │   └── matmul_benchmark_cpu.py
├── inference_lib/
│   │   ├── MANIFEST.in
│   │   ├── pyproject.toml
│   │   └── setup.cfg
│   ├── src/
│   │   ├── aqlm/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── inference.py
│   │   │   │   └── utils.py
│   │   │   ├── inference_kernels/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── cuda_kernel.cpp
│   │   │   │   │   ├── cuda_kernel.cu
│   │   │   │   │   ├── cuda_kernel.py
│   │   │   │   │   ├── dequantization.py
│   │   │   │   │   ├── kernel_selector.py
│   │   │   │   │   ├── numba_kernel.py
│   │   │   │   │   └── triton_kernel.py
├── notebooks/
│   │   ├── aq_simple.ipynb
│   │   ├── aqlm_2bit_training.ipynb
│   │   ├── aqlm_cuda_graph.ipynb
│   │   ├── aqlm_vllm.ipynb
│   │   ├── colab_example.ipynb
│   │   └── streaming_example.ipynb
├── src/
│   │   ├── __init__.py
│   │   ├── aq.py
│   │   ├── beam_search_l2.py
│   │   ├── beam_search_xtx.py
│   │   ├── configurable_adam.py
│   │   ├── datautils.py
│   │   ├── finetune.py
│   │   ├── kmeans.py
│   │   ├── memory_efficient_loss.py
│   │   ├── modelutils.py
│   │   ├── pv_optimizer.py
│   │   ├── pv_utils.py
│   │   └── utils.py
```

## 3. 核心源代码分析

### `README.md`

```markdown
# AQLM

Official PyTorch implementation for [Extreme Compression of Large Language Models via Additive Quantization](https://arxiv.org/pdf/2401.06118.pdf)

**[2025.04]** Released aqlm v1.1.7. Added support for arbitrary 8-dimensional codebooks on GPU, improved accuracy for 1-bit models, e.g. [ISTA-DASLab/Llama-2-7b-AQLM-1Bit-1x8-hf](https://huggingface.co/ISTA-DASLab/Llama-2-7b-AQLM-1Bit-1x8-hf) at ~1 bit achieves WikiText 2 PPL 7.85. To quantize your own models this way, use `num_codebooks=1, nbits_per_codebook=256` as per the tutorial below.

**[2024.11]** [PV-tuning](https://proceedings.neurips.cc/paper_files/paper/2024/hash/091166620a04a289c555f411d8899049-Abstract-Conference.html) was accepted to [NeurIPS'2024](https://neurips.cc/Conferences/2024) for oral presentation!

**[2024.05]** AQLM was accepted to [ICML'2024](https://icml.cc/Conferences/2024)! If you're attending, meet us around [this poster](https://icml.cc/virtual/2024/poster/34964).

**[2024.06]** We released a new paper that extends AQLM with new finetuning algorithm called [PV-tuning](https://arxiv.org/abs/2405.14852).
We're also releasing PV-tuned AQLM models [**in this collection**](https://huggingface.co/collections/ISTA-DASLab/aqlmpv-66564dff5d84f00a893ba93f)

**[2024.08]** We have [merged](https://github.com/Vahe1994/AQLM/commit/a441a3f0ece4cbaa2a91a3421c95a8b7432e4d99) the PV-Tuning branch into the main branch.
To reproduce results with old finetuning (before Aug 21), use commit [559a366](https://github.com/Vahe1994/AQLM/commit/559a36681398d7189297fccf3b1e59e8e030e942).

## Inference

### Demo

Learn how to run the prequantized models using this Google Colab examples:

| Basic AQLM <br> generation | Streaming with <br> GPU/CPU | Inference with CUDA <br> graphs (3x speedup) | Fine-tuning <br> with PEFT | Serving with <br> `vLLM` |
|:-----------:|:-------:|:---------------:|:----------:|:--------:|
| <a target="_blank" href="https://colab.research.google.com/github/Vahe1994/AQLM/blob/main/notebooks/colab_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="AQLM In Colab"/></a>         | <a target="_blank" href="https://colab.research.google.com/github/Vahe1994/AQLM/blob/main/notebooks/streaming_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="AQLM In Colab"/></a> | <a target="_blank" href="https://colab.research.google.com/github/Vahe1994/AQLM/blob/main/notebooks/aqlm_cuda_graph.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | <a target="_blank" href="https://colab.research.google.com/github/Vahe1994/AQLM/blob/main/notebooks/aqlm_2bit_training.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>  | <a target="_blank" href="https://colab.research.google.com/github/Vahe1994/AQLM/blob/main/notebooks/aqlm_vllm.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |


### Models

This repository is currently designed to work with models of `LLaMA`, `Mistral` and `Mixtral` families.
The models reported below use **full model fine-tuning** as described in appendix A, with cross-entropy objective with teacher logits.

We provide a number of prequantized AQLM models without PV-Tuning (scroll down for PV-Tuned models):

| Model      | AQLM scheme | WikiText-2 PPL | MMLU (5-shot) FP16→AQLM | Model size, Gb | Hub link                                                                 |
|------------|-------------|----------------|---------------|----------------|--------------------------------------------------------------------------|
| Llama-3-8b | 1x16        | -          | 0.65→0.56 | 4.1            | [Link](https://huggingface.co/ISTA-DASLab/Meta-Llama-3-8B-AQLM-2Bit-1x16) |
| Llama-3-8b-Instruct | 1x16        | -          | 0.66→0.59 | 4.1            | [Link](https://huggingface.co/ISTA-DASLab/Meta-Llama-3-8B-Instruct-AQLM-2Bit-1x16) |
| Llama-3-70b | 1x16        | -          | 0.79→0.75 | 21.9            | [Link](https://huggingface.co/ISTA-DASLab/Meta-Llama-3-70B-AQLM-2Bit-1x16) |
| Llama-3-70b-Instruct | 1x16        | -          | 0.80→0.76 | 21.9            | [Link](https://huggingface.co/ISTA-DASLab/Meta-Llama-3-70B-Instruct-AQLM-2Bit-1x16) |
| Command-R | 1x16      | -           | 0.68→0.57 | 12.7            | [Link](https://huggingface.co/ISTA-DASLab/c4ai-command-r-v01-AQLM-2Bit-1x16)|
| Command-R+ | 1x16      | -           | 0.74→0.68 | 31.9            | [Link](https://huggingface.co/ISTA-DASLab/c4ai-command-r-plus-AQLM-2Bit-1x16)|
| Mistral-7b| 1x16       | 5.40           | - | 2.5            | [Link](https://huggingface.co/ISTA-DASLab/Mistral-7B-v0.1-AQLM-2Bit-1x16-hf)|
| Mistral-7B-Instruct-v0.2 | 2x8       | -           | 0.59→0.44 | 2.5            | [Link](https://huggingface.co/ISTA-DASLab/Mistral-7B-Instruct-v0.2-AQLM-2Bit-2x8)|
| Mixtral-8x7b| 1x16       | 3.35           | -| 12.6            | [Link](https://huggingface.co/ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf)|
| Mixtral-8x7b-Instruct| 1x16       | -           | -| 12.6            | [Link](https://huggingface.co/ISTA-DASLab/Mixtral-8x7B-Instruct-v0_1-AQLM-2Bit-1x16-hf)|
| Llama-2-7b | 1x16        | 5.92          | 0.46→0.39 | 2.4            | [Link](https://huggingface.co/ISTA-DASLab/Llama-2-7b-AQLM-2Bit-1x16-hf) |
| Llama-2-7b | 2x8         | 6.69          | - | 2.2            | [Link](https://huggingface.co/ISTA-DASLab/Llama-2-7b-AQLM-2Bit-2x8-hf)  |
| Llama-2-7b | 8x8         | 6.61          | - | 2.2            | [Link](https://huggingface.co/ISTA-DASLab/Llama-2-7b-AQLM-2Bit-8x8-hf)  |
| Llama-2-13b| 1x16        | 5.22           | 0.55→0.49 | 4.1            | [Link](https://huggingface.co/ISTA-DASLab/Llama-2-13b-AQLM-2Bit-1x16-hf)|
| Llama-2-13b| 2x8        |  5.63          | - | 3.8            | [Link](https://huggingface.co/ISTA-DASLab/Llama-2-13b-AQLM-2Bit-2x8-hf)|
| Llama-2-70b| 1x16        | 3.83           | 0.69→0.65 | 18.8           | [Link](https://huggingface.co/ISTA-DASLab/Llama-2-70b-AQLM-2Bit-1x16-hf)|
| Llama-2-70b| 2x8         | 4.21           | - | 18.2           | [Link](https://huggingface.co/ISTA-DASLab/Llama-2-70b-AQLM-2Bit-2x8-hf) |
| gemma-2b | 1x16      | -           | - | 1.7            | [Link](https://huggingface.co/ISTA-DASLab/gemma-2b-AQLM-2Bit-1x16-hf)|
| gemma-2b | 2x8      | -           | - | 1.6            | [Link](https://huggingface.co/ISTA-DASLab/gemma-2b-AQLM-2Bit-2x8-hf)|

You can also download AQLM models tuned via PV-tuning:

| Model      | AQLM scheme | WikiText-2 PPL | Model size, Gb | Hub link                                                                 |
|------------|-------------|----------------|----------------|--------------------------------------------------------------------------|
| Llama-2-7b | 1x16g8        | 5.68          | 2.4            | [Link](https://huggingface.co/ISTA-DASLab/Llama-2-7b-AQLM-PV-2Bit-1x16-hf) |
| Llama-2-7b | 2x8g8         | 5.90          | 2.2            | [Link](https://huggingface.co/ISTA-DASLab/Llama-2-7b-AQLM-PV-2Bit-2x8-hf)  |
| Llama-2-7b | 1x16g16     | 9.21          | 1.7            | [Link](https://huggingface.co/justheuristic/Llama-2-7b-AQLM-PV-1Bit-1x16-hf)  |
| Llama-2-7b | 1x8g8 (**New!**)     | 7.85          | 1.34            | [Link](https://huggingface.co/ISTA-DASLab/Llama-2-7b-AQLM-1Bit-1x8-hf)  |
| Llama-2-13b| 1x16g8        | 5.05           | 4.1            | [Link](https://huggingface.co/ISTA-DASLab/Llama-2-13b-AQLM-PV-2Bit-1x16-hf)|
| Llama-2-70b| 1x16g8        | 3.78           | 18.8           | [Link](https://huggingface.co/ISTA-DASLab/Llama-2-70b-AQLM-PV-2Bit-1x16-hf)|
| Meta-Llama-3-8B | 1x16g8        | 6.99          | 4.1            | [Link](https://huggingface.co/ISTA-DASLab/Meta-Llama-3-8B-AQLM-PV-2Bit-1x16) |
| Meta-Llama-3-8B  | 1x16g16        | 9.43          | 3.9            | [Link](https://huggingface.co/ISTA-DASLab/Meta-Llama-3-8B-AQLM-PV-1Bit-1x16) |
| Meta-Llama-3-70B | 1x16g8        | 4.57           | 21.9           | [Link](https://huggingface.co/ISTA-DASLab/Meta-Llama-3-70B-AQLM-PV-2Bit-1x16)|
| Meta-Llama-3-70B | 1x16g16        | 8.67           | 13           | [Link](https://huggingface.co/ISTA-DASLab/Meta-Llama-3-70B-AQLM-PV-1Bit-1x16)|
| Mistral-7B-v0.1 | 1x16g8  | 5.22 | 2.51 | [Link](https://huggingface.co/ISTA-DASLab/Mistral-7B-v0.1-AQLM-PV-2Bit-1x16-hf) |
| Phi-3-mini-4k-instruct | 1x16g8 | 6.63 | 1.4 | [Link](https://huggingface.co/ISTA-DASLab/Phi-3-mini-4k-instruct-AQLM-PV-2Bit-1x16-hf) |



Note that models with "g16" in their scheme require aqlm inference library v1.1.6 or newer: 
```bash
pip install aqlm[gpu,cpu]>=1.1.6
```

Above perplexity is evaluated on **4k** context length for Llama 2 models and **8k** for Mistral/Mixtral and Llama 3. 
Please also note that token-level perplexity can only be compared within the same model family, but should not be compared between models that use different vocabularies.
While Mistral has a lower perplexity than Llama 3 8B but this does not mean that Mistral is better: Llama's perplexity is computed on a much larger dictionary and has higher per-token perplexity because of that.

For more evaluation results and detailed explanations, please see our papers: [Egiazarian et al. (2024)](https://arxiv.org/abs/2401.06118) for pure AQLM and [Malinovskii et al. (2024)](https://arxiv.org/abs/2405.14852) for PV-Tuned models.

### Inference kernels

AQLM quantization setpus vary mainly on the number of codebooks used as well as the codebook sizes in bits. The most popular setups, as well as inference kernels they support are:
 
| Kernel | Number of codebooks | Codebook size, bits | Scheme Notation | Accuracy | Speedup     | Fast GPU inference | Fast CPU inference |
|---|---------------------|---------------------|----------|-------------|-------------|--------------------|--------------------|
| Triton | K                   | N                  | KxN     | -        | Up to ~0.7x | ✅                  | ❌                  |
| CUDA | 1                   | 16                  | 1x16     | Best        | Up to ~1.3x | ✅                  | ❌                  |
| CUDA | 2                   | 8                   | 2x8      | OK          | Up to ~3.0x | ✅                  | ❌                  |
| Numba | K                   | 8                   | Kx8      | Good        | Up to ~4.0x | ❌                  | ✅                  |

### Installation



To run the models, one would have to install an inference library:
```bash
pip install aqlm[gpu,cpu]
```
, specifying either `gpu`, `cpu` or both based on one's inference setting.


Then, one can use the familiar `.from_pretrained` method provided by the [transformers](https://github.com/huggingface/transformers) library:
```python
from transformers import AutoModelForCausalLM

quantized_model = AutoModelForCausalLM.from_pretrained(
    "ISTA-DASLab/Llama-2-7b-AQLM-2Bit-1x16-hf",
    trust_remote_code=True, torch_dtype="auto"
).cuda()
```
Notice that `torch_dtype` should be set to either `torch.float16` or `"auto"` on GPU and `torch.float32` on CPU. After that, the model can be used exactly the same as one would use and unquantized model. 



## Quantization

### Dependencies

Install packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Loading / caching datasets and tokenizer

The script will require downloading and caching locally the relevant tokenizer and the datasets. 
They will be saved in default Huggingface Datasets directory unless alternative location is provided by env variables.
See [relevant Datasets documentation section](https://huggingface.co/docs/datasets/main/en/cache#cache-directory)

### Data

When quantizing models with AQLM, we recommend that you use a subset of the original data the model was trained on.

For Llama-2 models, the closest available dataset is [RedPajama](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample) . To load subset of RedPajama provide "pajama" in --dataset argument.
This will process nsamples data and tokenize it using provided model tokenizer.

Additionally we provide tokenized Redpajama for LLama and Solar/Mistral models for 4096 context lengths stored in [Hunggingface](https://huggingface.co/datasets/Vahe1994/AQLM) .
To load it, use:

```python
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="Vahe1994/AQLM", filename="data/name.pth", repo_type="dataset")
```

To use downloaded data from HF, place it in data folder(optional) and set correct path to it in "--dataset" argument in main.py.

**Warning:** These subsets are already processed with the corresponding model tokenizer. If you want to quantize another model (e.g. mistral/mixtral), please re-tokenize the data with provided script in src/datautils.

### WandB logging

One can optionally log the data to `Weights and Biases` service (wandb).
Run `pip install wandb` for W&B logging.
Specify `$WANDB_ENTITY`, `$WANDB_PROJECT`, `$WANDB_NAME` environment variables prior to running experiments. use `--wandb` argument to enable logging

### GPU and RAM requirements
This code was developed and tested using a several A100 GPU with 80GB GPU RAM. 
You can use the `--offload activations` option to reduce VRAM usage.
For `Language Model Evaluation Harness` evaluation one needs to have enough memory to load whole model  + activation tensors 
on one or several devices.

### Quantization time

AQLM quantization takes considerably longer to calibrate than simpler quantization methods such as GPTQ. This only impacts quantization time, not inference time.

For instance, quantizing a 7B model with default configuration takes about 1 day on a single A100 gpu. Similarly, quantizing a 70B model on a single GPU would take 10-14 days. If you have multiple GPUs with fast interconnect, you can run AQLM multi-gpu to speed up comparison - simply set CUDA_VISIBLE_DEVICES for multiple GPUs. Quantizing 7B model on two gpus reduces quantization time to ~14.5 hours. Similarly, quantizing a 70B model on 8 x A100 GPUs takes 3 days 18 hours.

If you need to speed up quantization without adding more GPUs, you may also increase `--relative_mse_tolerance` or set `--init_max_points_per_centroid` or limit `--finetune_max_epochs`. 
However, that usually comes at a cost of reduced model accuracy.

### Model downloading
The code requires the LLaMA model to be downloaded in Huggingface format and saved locally. The scripts below assume that `$TRANSFORMERS_CACHE` variable points to the Huggingface Transformers cache folder.
To download and cache the models, run this in the same environment:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "meta-llama/Llama-2-7b-hf"  # or whatever else you wish to download
tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype="auto")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
```


### How to quantize a model with AQLM
This script compresses the model and then tests its performance in terms of perplexity using WikiText2, C4, and Penn Treebank datasets. 

The command to launch the script should look like this: 

```bash
export CUDA_VISIBLE_DEVICES=0   # or e.g. 0,1,2,3
export MODEL_PATH=<PATH_TO_MODEL_ON_HUB>
export DATASET_PATH=<INSERT DATASET NAME OR PATH TO CUSTOM DATA>
export SAVE_PATH=/path/to/save/quantized/model/
export WANDB_PROJECT=MY_AQ_EXPS
export WANDB_NAME=COOL_EXP_NAME

python main.py $MODEL_PATH $DATASET_PATH \
 --nsamples=1024 \
 --val_size=128 \
 --num_codebooks=1 \
 --nbits_per_codebook=16 \
 --in_group_size=8 \
 --relative_mse_tolerance=0.01 \
 --finetune_batch_size=32 \
 --finetune_max_epochs=10 \
 --finetune_early_stop=3 \
 --finetune_keep_best \
 --local_batch_size=1 \
 --offload_activations \
 --wandb \
 --resume \
 --save $SAVE_PATH
```

Main CLI arguments:
- `CUDA_VISIBLE_DEVICES` - by default, the code will use all available GPUs. If you want to use specific GPUs (or one GPU), use this variable.
- `MODEL_PATH` - a path to either Hugging Face hub (e.g. meta-llama/Llama-2-7b-hf) or a local folder with transformers model and a tokenizer.
- `DATASET_PATH` - either a path to calibration data (see above) or a standard dataset `[c4, ptb, wikitext2]`
   - for llama-2 models, you can use `DATASET_PATH=./data/red_pajama_n=1024_4096_context_length.pth` for a slice of RedPajama (up to 1024 samples)
- `--nsamples` - the number of calibration data _sequences_ (train + validation). If this parameter is not set, take all calibration data avaialble.
- `--val_size` - the number of validation sequences for early stopping on block finetuning. By default equal to 0. Must be smaller than `--nsamples`.
- `--num_codebooks` - number of codebooks per layer
- `--nbits_per_codebook` - each codebook will contain 2 ** nbits_per_codebook vectors
- `--in_group_size` - how many weights are quantized together (aka "g" in the arXiv paper)
- `--finetune_batch_size` - (for fine-tuning only) the total number of sequences used for each optimization step
- `--local_batch_size` - when accumulating finetune_batch_size, process this many samples per GPU per forward pass (affects GPU RAM usage)
- `--relative_mse_tolerance`- (for initial calibration) - stop training when (current_epoch_mse / previous_epoch_mse) > (1 - relative_mse_tolerance)
- `--finetune_max_epochs` - maximal number of passes through calibration data on block tuning.
- `--finetune_early_stop` -  maximal number of passes through calibration data without improvement on validation.
- `--offload_activations` -- during calibration, move activations from GPU memory to RAM. This reduces VRAM usage while slowing calibration by ~10% (depending on your hardware). 
- `--save` -- path to save/load quantized model. (see also: `--load`)
- `--wandb` - if this parameter is set, the code will log results to wandb
- `--attn_implementation` - specify attention (for transformers >= `4.38`). Sdpa attention sometimes causes issues and it is recommended to use `eager` implementation.

There are additional hyperparameters aviailable. Run `python main.py --help` for more details on command line arguments, including compression parameters.


### Preparing fine-tuning dataset

This is a script is used to pre-tokenize a subset of RedPajama data for future fine-tuning.

```sh
TARGET_MODEL=meta-llama/Llama-2-7b-hf  # used for tokenization
SEQLEN=4096
DATASET=togethercomputer/RedPajama-Data-1T-Sample
OUTPUT_PATH=./redpajama_tokenized_llama2

CUDA_VISIBLE_DEVICES=0 HF_HOME=/mnt/LLM OMP_NUM_THREADS=16 torchrun --master-port 3456 --nproc-per-node=1 finetune.py --base_model $TARGET_MODEL --quantized_model ./doesnt_matter --dtype bfloat16 --block_type LlamaDecoderLayer --dataset_name=$DATASET --split train --dataset_config_name plain_text --cache_dir=./cache_dir --trust_remote_code --model_seqlen=$SEQLEN --preprocessing_num_workers=64 --preprocessing_chunk_length 100000 --save_dataset_and_exit $OUTPUT_PATH

tar -cvf tokenized_data_llama2.tar $OUTPUT_PATH   # optionally pack for distribution
```

The tokenized dataset is specific the model family (or more specifically, its tokenizer). For instance, Llama-3 8B is compatible with Llama-3 70B, but not with Llama-2 because it uses a different tokenizer.
To tokenize the data for another model, you need to set 1) --base_model 2) model_seqlen and 3) the path to --save_dataset_and_exit .

You can also set --preprocessing_num_workers to something hardware-appropriate. Note that setting --download_num_workers > 1 may cause download errors, possibly due to rate limit. These and other parameters are explained in the script's --help.
The job requires 150-200 GiB of disk space to store the dataset sample and preprocessing cache. Both are stored in ./cache_dir and can be deleted afterwards.

### Finetuning

**Note** to reproduce results with old finetuning (before Aug 21), use commit [559a366](https://github.com/Vahe1994/AQLM/commit/559a36681398d7189297fccf3b1e59e8e030e942).
Old version of finetuning produced worse results than new one even without PV-tuning, but was faster.

The accuracy of the quantized model can be further improved via finetuning.

To use our new PV-Tuning algorithm, the command to launch the script should look like this: 

```bash
torchrun --nproc-per-node=$NUM_GPUS finetune.py \
    --base_model $MODEL_PATH \
    --quantized_model $QUANTIZED_WEIGHTS_PATH \
    --model_seqlen=$SEQLEN \
    --block_type LlamaDecoderLayer \
    --load_dtype bfloat16 \
    --amp_dtype bfloat16 \
    --code_dtype uint16 \
    --dataset_name=$TOKENIZED_DATASET_PATH \
    --split none \
    --seed 42 \
    --preprocessing_chunk_length 100000 \
    --cache_dir=$CACHE_DIR \
    --trust_remote_code \
    --update_codes \
    --update_codebooks_and_scales \
    --update_non_quantized_parameters \
    --lamb \
    --debias \
    --lr 3e-4 \
    --adam_beta1 0.90 \
    --adam_beta2 0.95 \
    --max_code_change_per_step 1e-2 \
    --code_lr 1e-2 \
    --code_beta1 0.0 \
    --code_beta2 0.95 \
    --beam_size 5 \
    --delta_decay 0 \
    --batch_size=128 \
    --microbatch_size=1 \
    --max_epochs 1 \
    --gradient_checkpointing \
    --print_every_steps=1 \
    --verbose_optimizer \
    --wandb \
    --eval_every_steps=10 \
    --keep_best_model \
    --save $SAVE_PATH \
    --save_every_steps 100 \
    --attn_implementation flash_attention_2
```

### Zero-shot benchmarks via LM Evaluation Harness

To perform zero-shot evaluation, we adopt [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) framework. Our code works with models in standard `transformers`` format and may (optionally) load
the weights of a quantized model via `--aqlm_checkpoint_path` argument.

The evalution results in PV-Tuning were produced with `lm-eval=0.4.0`. 

To run evaluation make sure that proper version is installed or install it via:
`pip install lm-eval==0.4.0`. 

The main script for launching the evaluation procedure is `lmeval.py`.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # optional: select GPUs
export QUANTIZED_MODEL=<PATH_TO_SAVED_QUANTIZED_MODEL_FROM_MAIN.py>
export MODEL_PATH=<INSERT_PATH_TO_ORIINAL_MODEL_ON_HUB>
export DATASET=<INSERT DATASET NAME OR PATH TO CUSTOM DATA>
export WANDB_PROJECT=MY_AQLM_EVAL
export WANDB_NAME=COOL_EVAL_NAME

# for 0-shot evals
python lmeval.py \
    --model hf \
    --model_args pretrained=$MODEL_PATH,dtype=float16,parallelize=True \
    --tasks winogrande,piqa,hellaswag,arc_easy,arc_challenge \
    --batch_size <EVAL_BATCH_SIZE> \
    --aqlm_checkpoint_path QUANTIZED_MODEL # if evaluating quantized model

# for 5-shot MMLU
python lmeval.py \
    --model hf \
    --model_args pretrained=$MODEL_PATH,dtype=float16,parallelize=True \
    --tasks mmlu \
    --batch_size <EVAL_BATCH_SIZE> \
    --num_fewshot 5 \
    --aqlm_checkpoint_path QUANTIZED_MODEL # if evaluating quantized model
```

### Preparing models for inference

To convert a model into a _Hugging Face_ compatible format, use `convert_to_hf.py model in_path out_path` with corresponding arguments:
 - `model` - the original pretrained model (corresponds to `MODEL_PATH` of `main.py`, e.g. `meta-llama/Llama-2-7b-hf`).
 - `in_path` - the folder containing an initially quantized model (corresponds to `--save` of `main.py`).
 - `out_path` - the folder to save `transformers` model to.

You may also specify flags such as `--save_safetensors` to control the saved model format (see `--help` for details).

Example command: `python convert_to_hf.py meta-llama/Llama-2-7b-hf ./path/to/saved/quantization ./converted-llama2-7b-hf  --save_safetensors`

# Instructions for QuIP# finetuning
Instructions for QuIP# finetuning can be found [here](https://github.com/Vahe1994/AQLM/blob/quip-sharp-patch/QUIP_SHARP_INSTRUCTIONS.md).

## Contributing

If you want to contribute something substantial (more than a typo), please open an issue first.
We use black and isort for all pull requests. Before committing your code run `black . && isort .`

## Cite

If you found this work useful, please consider citing:

```
@misc{egiazarian2024extreme,
      title={Extreme Compression of Large Language Models via Additive Quantization}, 
      author={Vage Egiazarian and Andrei Panferov and Denis Kuznedelev and Elias Frantar and Artem Babenko and Dan Alistarh},
      year={2024},
      eprint={2401.06118},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
@misc{malinovskii2024pvtuning,
      title={PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression}, 
      author={Vladimir Malinovskii and Denis Mazur and Ivan Ilin and Denis Kuznedelev and Konstantin Burlachenko and Kai Yi and Dan Alistarh and Peter Richtarik},
      year={2024},
      eprint={2405.14852},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
```

### `__init__.py`

```python

```

### `aq_engine.py`

```python
from __future__ import annotations

import math
import random
from argparse import Namespace
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.nn.parallel.scatter_gather import Gather

from src.aq import QuantizedWeight
from src.utils import ellipsis


class AQEngine(nn.Module):
    """A wrapper class that runs AQ training for a single linear layer. All the important math is in aq.py"""

    def __init__(self, layer: nn.Linear, accumulator_dtype: torch.dtype = torch.float64):
        super().__init__()
        self.layer = layer
        self.device = layer.weight.device
        self.columns = self.layer.weight.data.shape[1]
        self.register_buffer(
            "XTX", torch.zeros((self.columns, self.columns), dtype=accumulator_dtype, device=self.device)
        )
        self.quantized_weight: Optional[QuantizedWeight] = None
        self.nsamples = 0

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor):
        """Accumulate a minibatch of layer inputs and update the X.T @ X (aka half hessian)"""
        assert self.XTX is not None, "Already ran quantization; cannot add more data batches"
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        tmp = inp.shape[0]
        inp = inp.t()

        self.XTX *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(1 / self.nsamples) * inp.to(self.XTX.dtype)
        self.XTX += inp.matmul(inp.t())

    @torch.enable_grad()
    def quantize(self, *, args: Namespace, verbose: bool = True) -> QuantizedWeight:
        """create a QuantizedLinear with specified args based on the collected hessian (XTX) data"""
        assert isinstance(args.devices, (list, tuple)) and len(args.devices) >= 1, f"Found devices = {args.devices}"
        assert args.devices[0] == self.device, (args.devices[0], self.XTX.device)
        self.quantized_weight = QuantizedWeight(
            reference_weight=self.layer.weight.detach().to(device=self.device, dtype=torch.float32),
            out_group_size=args.out_group_size,
            in_group_size=args.in_group_size,
            num_codebooks=args.num_codebooks,
            nbits_per_codebook=args.nbits_per_codebook,
            codebook_value_nbits=args.codebook_value_nbits,
            codebook_value_num_groups=args.codebook_value_num_groups,
            scale_nbits=args.scale_nbits,
            max_iter=args.init_max_iter,
            max_points_per_centroid=args.init_max_points_per_centroid,
            devices=args.devices,
            verbose=True,
        )

        differentiable_parameters = nn.ParameterDict(
            {name: param for name, param in self.quantized_weight.named_parameters() if param.requires_grad}
        )
        opt = torch.optim.Adam(differentiable_parameters.values(), lr=args.lr, betas=(0.0, 0.95), amsgrad=True)

        replicas = None
        if len(args.devices) > 1:
            replicas = torch.nn.parallel.replicate(self, args.devices)
            replicas[0] = self

        previous_best_loss = float("inf")  # for early stopping
        for epoch in range(args.max_epochs):
            # train codebooks and scales
            for step in range(args.steps_per_epoch):
                if len(args.devices) == 1:
                    loss = self._compute_mse()
                else:
                    loss = self._compute_mse_parallel(args.devices, replicas, differentiable_parameters)

                if not torch.isfinite(loss).item():
                    raise ValueError(f"Quantization loss is {loss}")
                if step == 0 and args.relative_mse_tolerance is not None:
                    if loss.item() / previous_best_loss > (1.0 - args.relative_mse_tolerance):
                        return self.quantized_weight  # early stopping; no updates after last epoch's beam search
                    previous_best_loss = min(previous_best_loss, loss.item())

                opt.zero_grad()
                loss.backward()
                opt.step()
                if verbose and (epoch * args.steps_per_epoch + step) % args.print_frequency == 0:
                    print(f"epoch={epoch}\tstep={step}\tloss={loss.item():.10f}\t")

            # search for better codes (cluster indices)
            seed = random.getrandbits(256)
            self.beam_search_update_codes_(
                args.devices,
                replicas,
                differentiable_parameters,
                seed=seed,
                beam_size=args.beam_size,
                verbose=True,
            )
        return self.quantized_weight

    def _compute_mse(self, selection: Union[slice, ellipsis] = ...) -> torch.Tensor:
        """
        Compute the activation MSE error = ||X @ quantized_weight - X @ reference_weight||^2
        Use the square-of-difference formula to avoid materializing per-batch predictions
        :param selection:  By default, compute MSE normally. If selection is specified, this method will instead
            compute MSE over a portion of output channels that align with the selected out_groups (for parallelism)
            The indices / slices must correspond to output channels (if out_group_size==1) or groups (if > 1).
            Formally, the indices must be in range [ 0 , self.out_features // self.out_group_size )
        """
        assert self.quantized_weight is not None, "must be called inside / after AQUtil.quantize"
        quantized_weight = self.quantized_weight(selection)

        if isinstance(selection, ellipsis):
            reference_weight = self.layer.weight.detach().to(quantized_weight.dtype)
        else:
            assert isinstance(selection, slice)
            out_channel_selection = slice(
                selection.start * self.quantized_weight.out_group_size,
                selection.stop * self.quantized_weight.out_group_size,
            )

            reference_weight = self.layer.weight.detach()[out_channel_selection].to(quantized_weight.dtype)
        delta_weight = (quantized_weight - reference_weight).to(self.XTX.dtype)
        return (delta_weight @ self.XTX).flatten() @ delta_weight.flatten() / self.quantized_weight.out_features

    def _replace_and_compute_mse(self, params_to_replace: nn.ParameterDict, selection: slice) -> torch.Tensor:
        """Utility for parallelism: replace the specified parameters of self.quantized_weight, then compute MSE"""
        for param_name, param_value in params_to_replace.items():
            replace_parameter_(self.quantized_weight, param_name, param_value)
        return self._compute_mse(selection)

    def _compute_mse_parallel(
        self, devices: Sequence[torch.device], replicas: Sequence[AQEngine], parameters_to_replicate: nn.ParameterDict
    ) -> torch.Tensor:
        """Compute MSE in parallel over output channels"""
        replicated_parameters = torch.nn.parallel.replicate(parameters_to_replicate, devices, detach=False)
        num_output_groups = self.quantized_weight.out_features // self.quantized_weight.out_group_size
        shard_size = (num_output_groups - 1) // len(devices) + 1
        active_slices_by_replica = [
            slice(i * shard_size, min((i + 1) * shard_size, num_output_groups)) for i in range(len(devices))
        ]
        funcs_by_replica = [replica._replace_and_compute_mse for replica in replicas]
        inputs_by_replica = [(dict(), active_slices_by_replica[0])]  # no replacements needed for 0-th replica (master)
        for i in range(1, len(devices)):
            inputs_by_replica.append((replicated_parameters[i], active_slices_by_replica[i]))
        mse_components = torch.nn.parallel.parallel_apply(funcs_by_replica, inputs_by_replica, devices=devices)
        return Gather.apply(devices[0], 0, *(mse.view(1) for mse in mse_components)).sum()

    def _replace_and_beam_search(self, params_to_replace: nn.ParameterDict, selection: slice, **kwargs) -> torch.Tensor:
        """Utility for parallelism: replace the specified parameters of self.quantized_weight, then run beam search"""
        dtype = self.quantized_weight.codebooks.dtype
        for param_name, param_value in params_to_replace.items():
            replace_parameter_(self.quantized_weight, param_name, param_value)
        out_channel_selection = slice(
            selection.start * self.quantized_weight.out_group_size,
            selection.stop * self.quantized_weight.out_group_size,
        )
        reference_weight = self.layer.weight.detach()[out_channel_selection].to(dtype)
        return self.quantized_weight.beam_search_update_codes_(
            XTX=self.XTX.to(dtype), reference_weight=reference_weight, selection=selection, **kwargs
        ).clone()

    @torch.no_grad()
    def beam_search_update_codes_(
        self,
        devices: Sequence[torch.device],
        replicas: Sequence[AQEngine],
        parameters_to_replicate: nn.ParameterDict,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """Update quantized_weight codes in-place via beam search"""
        if len(devices) == 1:  # single device
            assert replicas is None
            dtype = self.quantized_weight.codebooks.dtype
            self.quantized_weight.beam_search_update_codes_(
                XTX=self.XTX.to(dtype),
                reference_weight=self.layer.weight.detach().to(dtype),
                dim_rng=random.Random(seed),
                **kwargs,
            )
        else:
            assert replicas[0] is self
            replicated_parameters = torch.nn.parallel.replicate(parameters_to_replicate, devices)
            num_output_groups = self.quantized_weight.out_features // self.quantized_weight.out_group_size
            shard_size = (num_output_groups - 1) // len(devices) + 1
            active_slices_by_replica = [
                slice(i * shard_size, min((i + 1) * shard_size, num_output_groups)) for i in range(len(devices))
            ]

            funcs_by_replica = [replica._replace_and_beam_search for replica in replicas]
            inputs_by_replica = [(dict(), active_slices_by_replica[0])]
            for i in range(1, len(devices)):
                inputs_by_replica.append((replicated_parameters[i], active_slices_by_replica[i]))
            kwargs_by_replica = [dict(kwargs, dim_rng=random.Random(seed)) for _ in range(len(devices))]
            new_code_parts_by_replica = torch.nn.parallel.parallel_apply(
                funcs_by_replica, inputs_by_replica, kwargs_by_replica, devices=devices
            )
            # gather all code parts and assign them to each replica
            for device, replica in zip(devices, replicas):
                replica.quantized_weight.set_codes(Gather.apply(device, 0, *new_code_parts_by_replica))


def replace_parameter_(module: nn.Module, name: str, new_value: torch.Tensor):
    """A hacky way to substitute an already registered parameter with a non-parameter tensor. Breaks future use."""
    if name in module._parameters:
        module._parameters[name] = new_value
    else:
        setattr(module, name, new_value)
```

### `convert_legacy_model_format.py`

```python
"""
This abomination converts between one of several quantized model formats to the same format as returned by main.py .
This code exists because we failed to produce a single data format for quantized model.
We should eventually switch to saving all models in the same data format. Once we do, this file should be deleted.
"""
import argparse
import os
import warnings
from copy import deepcopy

import torch
import transformers.models
from torch import nn

from src.aq import QuantizedLinear, QuantizedWeight
from src.modelutils import get_model, save_quantized_model
from src.utils import is_signed


def load_quantized_model_with_old_pickle(base_model_name: str, quantized_model_name: str, **kwargs):
    """Hacky way to allow compatibility between old *pickled* layers and new transformers"""
    # because patching it for the fourth time is better than writing a proper saver once >.<
    import transformers.activations

    if not hasattr(transformers.activations, "SiLUActivation"):
        transformers.activations.SiLUActivation = deepcopy(torch.nn.SiLU)
        transformers.activations.SiLUActivation.inplace = False
        # https://github.com/huggingface/transformers/issues/28496
    if not hasattr(transformers.models.llama.modeling_llama.LlamaAttention, "attention_dropout"):
        transformers.models.llama.modeling_llama.LlamaAttention.attention_dropout = 0
    quantized_model = get_model(base_model_name, None, **kwargs)
    quantized_model_src = get_model(base_model_name, quantized_model_name, **kwargs)
    for module in quantized_model_src.modules():
        if isinstance(module, QuantizedWeight) and not hasattr(module, "codes_storage"):
            module.codes_storage = None  # backwards compatibility with older pickled snapshots

    lut = {}
    for name, module in quantized_model_src.named_modules():
        for child_name, child_module in module.named_children():
            if isinstance(child_module, QuantizedWeight):
                lut[name + "." + child_name] = child_module
    print(f"found {len(lut)} quantized weight matrices")
    for name, module in quantized_model.named_modules():
        for child_name, child_module in module.named_children():
            if name + "." + child_name + ".quantized_weight" in lut:
                quantized_weight = lut.pop(name + "." + child_name + ".quantized_weight")
                assert isinstance(child_module, nn.Linear)
                setattr(module, child_name, QuantizedLinear(quantized_weight, bias=child_module.bias))
    assert not lut, list(lut.keys())
    quantized_model.load_state_dict(quantized_model_src.state_dict())
    warnings.warn("You should be ashamed of yourself.")
    return quantized_model


import functools


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def load_quantized_model_from_fdsp_checkpoint(base_model_name: str, fsdp_checkpoint_path: str, **kwargs):
    original_model = get_model(base_model_name, None, **kwargs)

    state_filenames = os.listdir(fsdp_checkpoint_path)

    non_quant_fname = "non_quantized_state_dict.pth"
    non_quant_path = os.path.join(fsdp_checkpoint_path, non_quant_fname)
    non_quant_states = torch.load(non_quant_path)

    incomp_keys = original_model.load_state_dict(non_quant_states, strict=False)
    assert not incomp_keys.unexpected_keys

    missing_keys = list()
    for module_name, module in original_model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        assert not module.bias
        state_fname = f"{module_name}.weight.pth"

        if state_fname not in state_filenames:
            missing_keys.append(module_name)
            continue

        state_path = os.path.join(fsdp_checkpoint_path, state_fname)
        quantized_weight = torch.load(state_path, map_location="cpu")
        quantized_linear = QuantizedLinear(quantized_weight, bias=None)
        rsetattr(original_model, module_name, quantized_linear)

    return original_model


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="path or name of the teacher model",
    )
    parser.add_argument(
        "--quantized_model",
        type=str,
        required=True,
        help="path to quantized model",
    )
    parser.add_argument(
        "--load_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model in",
    )
    parser.add_argument(
        "--code_dtype",
        type=str,
        default=None,
        help="if specified, cast quantized layers' codes to this dtype; default = keep loaded dtype",
    )
    parser.add_argument(
        "--p_finetuned_state_dict",
        type=str,
        default=None,
        help="path to quantized model state dict saved by the old FSDP finetuning code",
    )
    parser.add_argument(
        "--pv_fsdp_dir",
        type=str,
        default=None,
        help="path to quantized model state dict saved by the old FSDP finetuning code",
    )
    parser.add_argument(
        "--monkeypatch_old_pickle",
        action="store_true",
        help="If set, load quantized_model in a hacky way that allows pickled models with older transformers/torch.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        help="Attention implementation for both teacher and student models: eager, sdpa, or flash_attention_2",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code when loading base model.",
    )
    parser.add_argument("--save", type=str, required=True, help="Save the converted quantized model here")

    args = parser.parse_args()
    assert args.p_finetuned_state_dict or args.pv_fsdp_dir, "either one of those must be specified"
    print(f"{args.p_finetuned_state_dict=}, {args.pv_fsdp_dir=}")
    assert (args.p_finetuned_state_dict is not None) != (args.pv_fsdp_dir is not None)

    args.load_dtype = getattr(torch, args.load_dtype) if args.load_dtype != "auto" else "auto"
    args.code_dtype = getattr(torch, args.code_dtype) if args.code_dtype is not None else None

    if not args.monkeypatch_old_pickle:
        quantized_model = get_model(
            args.base_model,
            args.quantized_model,
            dtype=args.load_dtype,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
        )
    elif args.p_finetuned_state_dict:
        quantized_model = load_quantized_model_with_old_pickle(
            args.base_model,
            args.quantized_model,
            dtype=args.load_dtype,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
        )
    elif args.pv_fsdp_dir:
        quantized_model = load_quantized_model_from_fdsp_checkpoint(
            args.base_model,
            args.pv_fsdp_dir,
            dtype=args.load_dtype,
            trust_remote_code=args.trust_remote_code,
        )

    for module in quantized_model.modules():
        if isinstance(module, QuantizedWeight):
            if not hasattr(module, "codes_storage"):
                module.codes_storage = None
            if module.codes is None:
                module.unwrap_codes_()
            assert module.codes is not None
            if args.code_dtype is not None:
                assert module.nbits_per_codebook <= torch.iinfo(args.code_dtype).bits - is_signed(args.code_dtype)
                module.codes = nn.Parameter(module.codes.to(args.code_dtype), requires_grad=module.codes.requires_grad)

    if args.p_finetuned_state_dict is not None:
        state_dict = torch.load(args.p_finetuned_state_dict, map_location="cpu")
        state_dict = {k: v for k, v in state_dict.items() if not k.endswith(".codes_storage.data")}
        status = quantized_model.load_state_dict(state_dict, strict=False)
        assert all(key.endswith("codes") for key in status.missing_keys)
        assert not status.unexpected_keys
        del state_dict, status  # note: in this case, it is okay not to load codes since P step does not change them

    save_quantized_model(quantized_model, args.save)


if __name__ == "__main__":
    main()
```

### `convert_to_hf.py`

```python
import json
import os
import re
import shutil

import torch
from tqdm.auto import trange
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    import safetensors
except ModuleNotFoundError:
    safetensors = None


def get_int_dtype(nbits: int) -> torch.dtype:
    if nbits <= 8:
        return torch.int8
    if nbits <= 16:
        return torch.int16
    if nbits <= 32:
        return torch.int32
    if nbits <= 64:
        return torch.int64
    raise ValueError(f"No dtype available for {nbits}-bit codebooks")


@torch.inference_mode()
def pack_int_data(data: torch.IntTensor, nbits: int) -> torch.IntTensor:
    data[data >= 2 ** (nbits - 1)] -= 2**nbits
    return data.to(get_int_dtype(nbits))


def get_num_layers(config) -> int:
    match config.model_type:
        case "llama" | "mistral" | "mixtral" | "gemma" | "phi3" | "qwen2":
            return config.num_hidden_layers
        case unknown_type:
            raise NotImplementedError(f"Can't get number of layers for {unknown_type}")


def get_layers_prefix(config) -> str:
    match config.model_type:
        case "llama" | "mistral" | "mixtral" | "gemma" | "phi3" | "qwen2":
            return "model.layers"
        case unknown_type:
            raise NotImplementedError(f"Can't get layers prefix for {unknown_type}")


def get_converted_state_dict(config, nbits: int, in_path: os.PathLike) -> [dict, list[str]]:
    state_dict = {}
    linear_weights_not_to_quantize = []

    num_layers = get_num_layers(config)
    layers_prefix = get_layers_prefix(config)

    for i in trange(num_layers):
        layer = torch.load(os.path.join(in_path, f"{i}.pth"), weights_only=False)
        for name, p in layer.named_parameters():
            if torch.is_floating_point(p.data):
                p.data = p.data.half()
            else:
                p.data = pack_int_data(p.data, nbits)
            if "quantized_weight." not in name:
                linear_weights_not_to_quantize.append(f"{layers_prefix}.{i}.{name}")
            else:
                name = re.sub("quantized_weight.", "", name)
            state_dict[f"{layers_prefix}.{i}.{name}"] = p.data

    for key, value in torch.load(os.path.join(in_path, "not_quantized_weights.pt")).items():
        state_dict[key] = value.half()
        linear_weights_not_to_quantize.append(key)

    if "lm_head.weight" not in linear_weights_not_to_quantize:
        linear_weights_not_to_quantize.append("lm_head.weight")

    return state_dict, linear_weights_not_to_quantize


def get_metadata(in_path: os.PathLike) -> dict:
    quant_args = torch.load(os.path.join(in_path, "args.pt"))
    return {
        "nbits_per_codebook": quant_args["nbits_per_codebook"],
        "num_codebooks": quant_args["num_codebooks"],
        "out_group_size": quant_args["out_group_size"],
        "in_group_size": quant_args["in_group_size"],
    }


def update_config(config_dict: dict, aqlm_metadata: dict[str, int], linear_weights_not_to_quantize: list[str]):
    config_dict["quantization_config"] = {
        "quant_method": "aqlm",
        "nbits_per_codebook": aqlm_metadata["nbits_per_codebook"],
        "num_codebooks": aqlm_metadata["num_codebooks"],
        "out_group_size": aqlm_metadata["out_group_size"],
        "in_group_size": aqlm_metadata["in_group_size"],
        "linear_weights_not_to_quantize": linear_weights_not_to_quantize,
    }
    config_dict["torch_dtype"] = "float16"
    return config_dict


def add_inference_code(model_type: str, save_path: os.PathLike):
    if os.path.isdir(f"./transformers/{model_type}"):
        shutil.copytree(f"./transformers/{model_type}", save_path, dirs_exist_ok=True)
    else:
        print(f"No predefined PreTrainedModel exists for {model_type}. You'll have to copy-paste some code yourself.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "model",
        type=str,
        help="Path to the model to base config on, as in AutoConfig.from_pretrained()",
    )
    parser.add_argument(
        "in_path",
        type=str,
        help="Path of the checkpoint to convert",
    )
    parser.add_argument(
        "out_path",
        type=str,
        help="Path to save HF compatible checkpoint to",
    )
    parser.add_argument(
        "--save_safetensors",
        action="store_true",
        help="Whether to save in safetensors format",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code",
    )
    parser.add_argument(
        "--load_model",
        action="store_true",
        help="Whether to load model",
    )
    parser.add_argument(
        "--save_tokenizer",
        action="store_true",
        help="Whether to save tokenizer",
    )
    args = parser.parse_args()

    old_config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    metadata = get_metadata(args.in_path)

    # load dummy model
    if args.load_model:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=args.trust_remote_code, low_cpu_mem_usage=True, torch_dtype=torch.float16
        )

    state_dict, linear_weights_not_to_quantize = get_converted_state_dict(
        old_config, metadata["nbits_per_codebook"], args.in_path
    )
    torch.save(state_dict, os.path.join(args.out_path, "pytorch_model.bin"))

    new_config_dict = update_config(old_config.to_diff_dict(), metadata, linear_weights_not_to_quantize)
    with open(os.path.join(args.out_path, "config.json"), "w") as config_file:
        json.dump(new_config_dict, config_file, indent=4)

    # convert to safetensors
    if args.save_safetensors:
        assert safetensors
        model = AutoModelForCausalLM.from_pretrained(args.out_path, trust_remote_code=True, torch_dtype=torch.float16)
        shutil.rmtree(args.out_path)
        model.save_pretrained(args.out_path)

    if args.save_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.save_pretrained(args.out_path)
```

### `finetune.py`

```python
"""
Fine-tune an LLM that was previously quantized with AQLM;
based on https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
"""
import argparse
import os
from contextlib import nullcontext
from functools import partial
from typing import Dict, Optional, Tuple

import datasets
import torch
import torch.distributed
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import transformers
from torch import nn as nn
from torch.distributed.fsdp import (
    CPUOffload,
    FullStateDictConfig,
    FullyShardedDataParallel,
    MixedPrecision,
    StateDictType,
)
from tqdm.auto import tqdm

from convert_legacy_model_format import load_quantized_model_with_old_pickle
from src.aq import QuantizedWeight
from src.configurable_adam import ConfigurableAdamW
from src.datautils import evaluate_perplexity, get_loaders, group_texts, split_long_texts
from src.memory_efficient_loss import compute_kl_divergence_loss_values
from src.modelutils import get_model, is_model_for_causal_lm
from src.pv_optimizer import StraightThroughAdamW
from src.pv_utils import (
    YourQuantizedWeightIsInAnotherRank,
    create_dequantized_model,
    get_original_named_parameters_from_fsdp_module,
    infer_module_classes,
    split_quantized_weights_between_ranks,
)
from src.utils import IntCodes, is_signed, master_rank_first, one_rank_at_a_time

try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False


def add_model_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="path or name of the teacher model",
    )
    parser.add_argument(
        "--quantized_model",
        type=str,
        required=True,
        help="path to quantized model",
    )
    parser.add_argument(
        "--monkeypatch_old_pickle",
        action="store_true",
        help="If set, load quantized_model in a hacky way that allows pickled models with older transformers/torch.",
    )
    parser.add_argument(
        "--model_seqlen",
        type=int,
        default=4096,
        help="Model seqlen and calibration data context length.",
    )
    parser.add_argument(
        "--master_dtype",
        type=str,
        default="float32",
        help="data type for storing master parameters and computing optimizer updates",
    )
    parser.add_argument(
        "--embed_dtype",
        type=str,
        default=None,
        help="data type for storing master input and output embeddings; defaults to master_dtype",
    )
    parser.add_argument(
        "--load_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model in",
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default=None,
        help="if specified, runs automated mixed precision with this dtype",
    )
    parser.add_argument(
        "--straight_through_buffer_dtype",
        type=str,
        default=None,
        help="data type for storing optimized straight through buffers, defaults to master_dtype",
    )
    parser.add_argument(
        "--code_dtype",
        type=str,
        default=None,
        help="if specified, cast quantized layers' codes to this dtype; default = keep loaded dtype",
    )
    parser.add_argument(
        "--block_type",
        type=str,
        required=True,
        help="string name of a transformer layer to wrap, e.g. LlamaDecoderLayer",
    )
    parser.add_argument(
        "--wrap_separately",
        type=str,
        nargs="*",
        default=[],
        help="module classes (by name, similar to block_type) that will be wrapped in a separate fsdp instance and do "
        "not participate in FSDP AMP (if used). Applies to the student (de)quantized model, not the teacher model.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        help="Attention implementation for both teacher and student models: eager, sdpa, or flash_attention_2",
    )
    parser.add_argument(
        "--limit_parallel_inits",
        type=int,
        default=1,
        help="this many ranks (per host) initialize their model in parallel. This parameter is meant to save host RAM.",
    )


def add_finetuning_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--update_codes",
        action="store_true",
        help="If set, train discrete codes; if not, freeze them",
    )
    parser.add_argument(
        "--update_codebooks_and_scales",
        action="store_true",
        help="If set, train continuous parameters of quantized representations; if not, freeze them",
    )
    parser.add_argument(
        "--update_non_quantized_parameters",
        action="store_true",
        help="If set, train the non-quantized model parameters (layernorm scales, biases, logits); if not, freeze them",
    )
    parser.add_argument(
        "--force_dequantize",
        action="store_true",
        help="If set, the algorithm will create a de-quantized model instead of dequantizing weights just in time even"
        "when doing p-only tuning. This version only has effect if --update_codes is not set. Setting this will"
        " make the training run faster, but it will also use substantially more memory.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="finetuning learning rate for continuous params",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.90,
        help="Adam beta1 for continuous params",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.95,
        help="Adam beta2 for continuous params",
    )

    parser.add_argument(
        "--code_lr",
        type=float,
        default=1e-2,
        help="finetuning learning rate for discrete codes",
    )
    parser.add_argument(
        "--code_beta1",
        type=float,
        default=0.0,
        help="Adam beta1 for discrete params",
    )
    parser.add_argument(
        "--code_beta2",
        type=float,
        default=0.95,
        help="Adam beta2 for discrete params",
    )
    parser.add_argument(
        "--delta_decay",
        type=float,
        default=0.0,
        help="Determines whether to use direct training, straight-through estimation or a mixture thereof. "
        "If delta_decay is 0, use straight-through estimation. If delta_decay is 1, do not use it at all. "
        "If between 0 and 1, every straight-through buffer will decay to the quantized weight with moving average."
        " straight_through_buffer = (1 - delta_decay) * straight_through_buffer + delta_decay * quantized_weight."
        " Please refer to the docstring of StraightThroughAdam for details.",
    )
    parser.add_argument(
        "--max_code_change_per_step",
        type=float,
        default=1e-2,
        help="Maximum number of code groups that can be changed during one update to codes. "
        "This constraint is enforced on a per-tensor level. If the weight is represented with multiple codes, "
        "changing any of the codes will count towards the limit. If more than this many code groups have changed, "
        "the algorithm will rollback the changes with least update norm until the constraint is satisfied.",
    )
    parser.add_argument(
        "--code_trust_ratio",
        type=float,
        default=None,
        help="By default, the optimizer can make arbitrary changes to quantized weights. If this parameter is set,"
        "the optimizer ensures that the change to quantized weights is not too large by undoing some of the change"
        "until ||new_quantized_weights - prev_quantized_weights|| / ||prev_quantized_weight|| <= code_trust_ratio."
        " See StraightThroughAdam docstring for details.",
    )
    parser.add_argument(
        "--force_code_update",
        action="store_true",
        help="If set, force discrete codes to change in the direction of optimizer update, even if previous codes"
        "were optimal in terms of MSE. See StraightThroughAdam docstring for details. Use when delta_decay==1.",
    )
    parser.add_argument(
        "--code_selection_temperature",
        type=float,
        default=0,
        help="If max_code_change_per_step or code_trust_ratio is set and code_selection_temperature=0, beam search will"
        " prioritize updating codes that have the largest continuosu update norm. If code_selection_temperature is"
        " not 0, sample a subset of codes for update stochastically. See StraightThroughAdam for details.",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=1,
        help="Beam size when updating codes; higher is slower but more accurate. For single codebook, use beam_size=1",
    )
    parser.add_argument(
        "--code_adam_16bit",
        action="store_true",
        help="If set, adam statistics for codes will be stored as float16 (exp_avg and v_hat) or bfloat16(exp_avg_sq)",
    )
    parser.add_argument(
        "--offload_optimizer",
        action="store_true",
        help="If set, adam statistics will be offloaded to RAM",
    )
    parser.add_argument(
        "--offload_teacher_params",
        action="store_true",
        help="If set, the teacher model will be offloaded to RAM and paged using FSDP's CPUOffload",
    )
    parser.add_argument(
        "--offload_student_params",
        action="store_true",
        help="If set, the student model will be offloaded to RAM and paged using FSDP's CPUOffload",
    )
    parser.add_argument(
        "--limit_all_gathers",
        action="store_true",
        help="sets limit_all_gathers in both FSDP instances",
    )
    parser.add_argument(
        "--forward_prefetch",
        action="store_true",
        help="sets forward_prefetech in both FSDP instances",
    )
    parser.add_argument(
        "--lamb",
        action="store_true",
        help="If set, use Lamb (aka Adam with trust ratio)",
    )
    parser.add_argument(
        "--amsgrad",
        action="store_true",
        help="if True, use the AMSGrad variant of adam/lamb",
    )
    parser.add_argument(
        "--debias",
        action="store_true",
        default=None,
        help="Whether or not to debias optimizer statistics; defaults to True for adam and False for Lamb",
    )
    parser.add_argument(
        "--no_debias",
        action="store_false",
        dest="debias",
        help="Disable optimizer debiasing (see above)",
    )
    parser.add_argument(
        "--verbose_optimizer",
        action="store_true",
        help="If set, the optimizer will print beam search results, tensors norms, etc",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="training batch size - how many samples are processed per optimizer step, between all GPUs in total",
    )
    parser.add_argument(
        "--microbatch_size",
        type=int,
        default=None,
        help="training microbatch size - how many samples are processed per GPU per forward pass",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether to apply gradient checkpointing for transformer blocks",
    )
    parser.add_argument(
        "--loss_tokens_per_chunk",
        type=int,
        default=None,
        help="If specified, compute LM logits and loss using gradient checkpointing in chunks of this size."
        "This option slows down loss computation, but reduces memory usage. Recommended for large vocabularies",
    )
    parser.add_argument(
        "--use_fsdp_amp",
        action="store_true",
        help="Whether to use FSDP native mixed precision (excluding registered layernorms and --wrap_separately).",
    )
    parser.add_argument(
        "--minimize_sync",
        action="store_true",
        help="if True, accumulate microbatch gradients locally and synchronize once per optimizer step. If False, "
        "synchronize after every step. This reduces communication overhead but increases memory usage. See "
        "https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.no_sync",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for calibration data and initialization. "
        "Note that the main training is not strictly deterministic.",
    )
    parser.add_argument("--wandb", action="store_true", help="Whether to use wandb or store locally.")
    parser.add_argument("--save", type=str, default=None, help="Path to save training snapshot.")
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1000,
        help="Total number of training epochs (passes over calibration data) after which the training will conclude",
    )
    parser.add_argument(
        "--print_every_steps",
        type=int,
        default=None,
        help="print training metrics once in this many optimizer steps (this many updates to model parameters)",
    )
    parser.add_argument(
        "--eval_every_steps",
        type=int,
        default=None,
        help="evaluate once in this many optimizer steps (this many updates to model parameters)",
    )
    parser.add_argument(
        "--save_every_steps",
        type=int,
        default=None,
        help="save state once in this many optimizer steps (this many updates to model parameters)",
    )
    parser.add_argument("--keep_best_model", action="store_true", help="Save best model state separately")
    parser.add_argument(
        "--on_save",
        type=str,
        default=None,
        help="Optional callback (python code string) to call after each saved layer. Example: when "
        "training on preemptible compute, upload partially finetuned model and resume later",
    )


def add_data_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Training dataset name (from HF datasets) or path to data where to extract calibration data from",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Training dataset split name, e.g. 'train'",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache dir for huggingface datasets",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="If set, re-run data preprocessing even if it is cached",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of CPU workers for preprocessing and data loading",
    )
    parser.add_argument(
        "--download_num_workers",
        type=int,
        default=None,
        help="Number of CPU workers for downloading the training dataset; overrides num_workers",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="Number of CPU workers for preprocessing; overrides num_workers",
    )
    parser.add_argument(
        "--preprocessing_chunk_length",
        type=int,
        default=100_000,
        help="Texts exceeding this length will be split approximately in the middle",
    )
    parser.add_argument(
        "--preprocessing_keep_in_memory",
        action="store_true",
        help="If set, do not save intermediate preprocessing steps in memory",
    )
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        type=str,
        default=["wikitext2", "c4"],
        help="Datasets to run evaluation on",
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        action="store_true",
        help="Whether to use fast tokenizer.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code when loading base model.",
    )
    parser.add_argument(
        "--save_dataset_and_exit",
        type=str,
        default=None,
        help="If not None, save tokenized dataset to this path and exit training immediately",
    )


def prepare_training_dataset(args: argparse.Namespace, tokenizer: transformers.PreTrainedTokenizer) -> datasets.Dataset:
    if os.path.exists(args.dataset_name):
        dataset = datasets.load_from_disk(args.dataset_name)
    else:
        dataset = datasets.load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=args.split,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
            num_proc=args.download_num_workers if args.download_num_workers is not None else args.num_workers,
            streaming=False,
        )

    def is_tokenized(dataset):
        return "input_ids" in dataset.column_names

    if is_tokenized(dataset):
        if torch.distributed.get_rank() == 0:
            print("Dataset already tokenized")
            assert len(dataset[0]["input_ids"]) == args.model_seqlen
        return dataset

    text_column_name = "text" if "text" in dataset.column_names else next(iter(dataset.column_names))

    if args.preprocessing_chunk_length is not None:
        dataset = dataset.map(
            lambda examples: {
                text_column_name: split_long_texts(examples[text_column_name], args.preprocessing_chunk_length)
            },
            batched=True,
            num_proc=args.preprocessing_num_workers if args.preprocessing_num_workers is not None else args.num_workers,
            remove_columns=list(dataset.column_names),
            keep_in_memory=args.preprocessing_keep_in_memory,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Splitting dataset over newline into chunks of ~{args.preprocessing_chunk_length} characters",
        )

    tokenized_dataset = dataset.map(
        lambda example: tokenizer(example[text_column_name]),
        num_proc=args.preprocessing_num_workers if args.preprocessing_num_workers is not None else args.num_workers,
        remove_columns=list(dataset.column_names),
        keep_in_memory=args.preprocessing_keep_in_memory,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )
    lm_dataset = tokenized_dataset.map(
        partial(group_texts, block_size=args.model_seqlen, add_labels=False),
        batched=True,
        num_proc=args.preprocessing_num_workers if args.preprocessing_num_workers is not None else args.num_workers,
        keep_in_memory=args.preprocessing_keep_in_memory,
        load_from_cache_file=not args.overwrite_cache,
        desc=f"Grouping texts in chunks of {args.model_seqlen}",
    )
    assert is_tokenized(lm_dataset)
    return lm_dataset


def load_teacher_model(args: argparse.Namespace, device: torch.device) -> FullyShardedDataParallel:
    """Load unquantized model with frozen parameters"""
    model = get_model(
        args.base_model,
        load_quantized=None,
        dtype=args.load_dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    ).to(dtype=args.load_dtype if args.load_dtype != "auto" else None)
    model.train(False)
    for param in model.parameters():
        param.requires_grad = False

    model.config.use_cache = False
    transformer_block_types = infer_module_classes(model, args.block_type)

    return wrap_model_with_fsdp_(
        model,
        auto_wrap_policy=lambda module, recurse, **_etc: recurse or isinstance(module, transformer_block_types),
        cpu_offload=CPUOffload(offload_params=args.offload_teacher_params) if args.offload_teacher_params else None,
        limit_all_gathers=args.limit_all_gathers,
        forward_prefetch=args.forward_prefetch,
        device_id=device,
    )


def load_student_model(
    args: argparse.Namespace, device: torch.device, dequantize: bool
) -> Tuple[FullyShardedDataParallel, Optional[Dict[str, QuantizedWeight]]]:
    """
    load student model for fine-tuning. If dequantize is set, dequantize all quantized weights to accumulate full grads
    """
    if not args.monkeypatch_old_pickle:
        student_model = get_model(
            args.base_model,
            args.quantized_model,
            dtype=args.load_dtype,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
        ).to(
            args.master_dtype
        )  # master parameters
    else:
        student_model = load_quantized_model_with_old_pickle(
            args.base_model,
            args.quantized_model,
            dtype=args.load_dtype,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
        ).to(args.master_dtype)

    if args.embed_dtype != args.master_dtype:
        student_model.set_output_embeddings(student_model.get_output_embeddings().to(args.embed_dtype))
        student_model.set_input_embeddings(student_model.get_input_embeddings().to(args.embed_dtype))

    student_model.config.use_cache = False
    student_model.train(True)  # note: HF gradient checkpoints do not work for some models without train(True); see
    # https://github.com/huggingface/transformers/blob/2d92db8/src/transformers/models/llama/modeling_llama.py#L1006
    if args.gradient_checkpointing:
        student_model.gradient_checkpointing_enable()
        student_model.enable_input_require_grads()

    # convert QuantizedModel state dict to make it compatible with FSDP
    for name, module in student_model.named_modules():
        if isinstance(module, QuantizedWeight):
            assert module.codes is not None
            if args.code_dtype is not None:
                assert module.nbits_per_codebook <= torch.iinfo(args.code_dtype).bits - is_signed(args.code_dtype)
                module.codes = nn.Parameter(module.codes.to(args.code_dtype), requires_grad=module.codes.requires_grad)
            module.wrap_codes_for_fsdp_()
            assert module.codes is None and isinstance(module.codes_storage, IntCodes)
    assert any(isinstance(module, IntCodes) for module in student_model.modules())

    if dequantize:
        student_model, named_quantized_params = create_dequantized_model(
            student_model, dequantized_dtype=args.amp_dtype, reuse_non_quantized=True
        )
    else:
        named_quantized_params = None

    transformer_block_types = list(infer_module_classes(student_model, args.block_type))
    layernorm_types = list(transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
    extra_block_types = list()
    for extra_module_name in args.wrap_separately:
        extra_block_types.extend(infer_module_classes(student_model, extra_module_name))
    block_types_to_wrap = tuple(
        set(
            transformer_block_types
            + layernorm_types
            + extra_block_types
            + [
                IntCodes,
            ]
        )
    )
    if torch.distributed.get_rank() == 0:
        print(f"Blocks to be wrapped separately: {block_types_to_wrap}\n")

    mixed_precision = None
    if args.use_fsdp_amp:
        assert args.amp_dtype is not None, "requested to use_fsdp_amp, but amp_dtype is not None"
        block_types_for_amp_to_ignore = tuple(set(layernorm_types + extra_block_types))
        if torch.distributed.get_rank() == 0:
            print(f"Blocks excluded from AMP: {block_types_for_amp_to_ignore}\n")
        mixed_precision = MixedPrecision(
            param_dtype=args.amp_dtype,
            reduce_dtype=args.amp_dtype,
            _module_classes_to_ignore=block_types_for_amp_to_ignore,
        )
    else:
        if torch.distributed.get_rank() == 0:
            print(f"Not using FSDP native MixedPrecision; Local amp_dtype={args.amp_dtype}.")

    student_model = wrap_model_with_fsdp_(
        student_model,
        use_orig_params=True,
        auto_wrap_policy=lambda module, recurse, **_etc: recurse or isinstance(module, block_types_to_wrap),
        cpu_offload=CPUOffload(offload_params=args.offload_student_params) if args.offload_student_params else None,
        limit_all_gathers=args.limit_all_gathers,
        forward_prefetch=args.forward_prefetch,
        mixed_precision=mixed_precision,
        device_id=device,
    )

    if named_quantized_params is not None:
        if torch.distributed.get_world_size() > 1:
            # distributed pv: each rank holds a subset of all quantized weights; the rest are replaced with pointers
            named_quantized_params = split_quantized_weights_between_ranks(
                named_quantized_params, verify_checksums=False
            )
        for quantized_weight in named_quantized_params.values():
            if isinstance(quantized_weight, QuantizedWeight):
                quantized_weight.to(device)
            else:
                assert isinstance(quantized_weight, YourQuantizedWeightIsInAnotherRank)

    return student_model, named_quantized_params


def wrap_model_with_fsdp_(
    model: transformers.PreTrainedModel, auto_wrap_policy: callable, **kwargs
) -> FullyShardedDataParallel:
    """Wrap a model *ForCausalLM components: transformer and lm_head are wrapped as FSDP instances"""
    assert isinstance(model, transformers.PreTrainedModel) and is_model_for_causal_lm(model)
    base_model, lm_head = model.base_model, model.get_output_embeddings()

    def _modified_auto_wrap_policy(module, recurse, **kwargs):
        return auto_wrap_policy(module, recurse, **kwargs) or (module in (base_model, lm_head))

    model = FullyShardedDataParallel(model, auto_wrap_policy=_modified_auto_wrap_policy, **kwargs)

    assert isinstance(model.module, transformers.PreTrainedModel)
    assert isinstance(model.base_model, FullyShardedDataParallel)
    assert isinstance(model.get_output_embeddings(), FullyShardedDataParallel)
    return model


def trigger_fsdp_lazy_init_(
    tokenizer: transformers.PreTrainedTokenizer,
    teacher_model: FullyShardedDataParallel,
    student_model: FullyShardedDataParallel,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
):
    """Trigger FullyShardedDataParallel lazy init in the correct order to allow both training and eval"""
    print("Initializing FSDP root")
    dummy_batch = tokenizer("I am the monument to all your sins", return_tensors="pt")
    dummy_batch = {k: v.to(device) for k, v in dummy_batch.items()}
    with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype):
        with torch.no_grad():
            teacher_model(**dummy_batch)
        (student_model(**dummy_batch).logits * 0).sum().backward()


def create_pv_optimizer(
    args: argparse.Namespace,
    student_model: FullyShardedDataParallel,
    named_quantized_params: Dict[str, QuantizedWeight],
) -> torch.optim.Optimizer:
    """Create optimizer for PV-Tuning using a de-quantized student model and a dictionary of quantized weights"""
    named_dequantized_params = get_original_named_parameters_from_fsdp_module(student_model)
    opt_device = torch.device("cpu") if args.offload_optimizer else next(student_model.parameters()).device
    assert all(name in named_dequantized_params for name in named_quantized_params)
    return StraightThroughAdamW(
        named_dequantized_params=named_dequantized_params,
        named_quantized_params=named_quantized_params,
        update_codes=dict(
            lr=args.code_lr,
            betas=(args.code_beta1, args.code_beta2),
            lamb=args.lamb,
            debias=args.debias,
            amsgrad=args.amsgrad,
            compute_dtype=args.master_dtype,
            exp_avg_dtype=torch.float16 if args.code_adam_16bit else args.master_dtype,
            exp_avg_sq_dtype=torch.bfloat16 if args.code_adam_16bit else args.master_dtype,
            v_hat_max_dtype=torch.float16 if args.code_adam_16bit else args.master_dtype,
            exp_avg_device=opt_device,
            exp_avg_sq_device=opt_device,
            v_hat_max_device=opt_device,
        )
        if args.update_codes
        else None,
        update_codebooks_and_scales=dict(
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            lamb=args.lamb,
            debias=args.debias,
            amsgrad=args.amsgrad,
            compute_dtype=args.master_dtype,
            exp_avg_dtype=args.master_dtype,
            exp_avg_sq_dtype=args.master_dtype,
            v_hat_max_dtype=args.master_dtype,
            exp_avg_device=opt_device,
            exp_avg_sq_device=opt_device,
            v_hat_max_device=opt_device,
        )
        if args.update_codebooks_and_scales
        else None,
        update_non_quantized_parameters=dict(
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            lamb=args.lamb,
            debias=args.debias,
            amsgrad=args.amsgrad,
            compute_dtype=args.master_dtype,
            exp_avg_dtype=args.master_dtype,
            exp_avg_sq_dtype=args.master_dtype,
            v_hat_max_dtype=args.master_dtype,
            exp_avg_device=opt_device,
            exp_avg_sq_device=opt_device,
            v_hat_max_device=opt_device,
        )
        if args.update_non_quantized_parameters
        else None,
        delta_decay=args.delta_decay,
        max_code_change_per_step=args.max_code_change_per_step,
        force_code_update=args.force_code_update,
        code_trust_ratio=args.code_trust_ratio,
        beam_size=args.beam_size,
        straight_through_buffer_dtype=args.straight_through_buffer_dtype,
        verbose=args.verbose_optimizer,
    )


def create_p_optimizer(args: argparse.Namespace, student_model: FullyShardedDataParallel) -> torch.optim.Optimizer:
    """Create optimizer for training only continuous parameters of a quantized model"""
    quantized_weight_continuous_parameters = set()
    for module in student_model.modules():
        if isinstance(module, QuantizedWeight):
            for param in module.parameters():
                if torch.is_floating_point(param) and param.requires_grad:
                    quantized_weight_continuous_parameters.add(param)
    all_trainable_params = []
    if args.update_codebooks_and_scales:
        all_trainable_params.extend(
            param for param in student_model.parameters() if param in quantized_weight_continuous_parameters
        )  # use iteration instead of simply adding list(set) to ensure deterministic order of parameters
    if args.update_non_quantized_parameters:
        all_trainable_params.extend(
            param
            for param in student_model.parameters()
            if torch.is_floating_point(param)
            and param.requires_grad
            and param not in quantized_weight_continuous_parameters
        )
    if args.update_codes:
        raise RuntimeError("When asked to update_codes, one should create_pv_optimizer, but this is create_p_optimizer")
    assert len(all_trainable_params) > 0, (
        "found no trainable parameters. Did you specify update_codes, "
        "update_codebooks_and_scales or update_non_quantized_parameters?"
    )
    opt_device = torch.device("cpu") if args.offload_optimizer else next(student_model.parameters()).device
    return ConfigurableAdamW(
        params=list(all_trainable_params),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        lamb=args.lamb,
        debias=args.debias,
        amsgrad=args.amsgrad,
        compute_dtype=args.master_dtype,
        exp_avg_dtype=args.master_dtype,
        exp_avg_sq_dtype=args.master_dtype,
        v_hat_max_dtype=args.master_dtype,
        exp_avg_device=opt_device,
        exp_avg_sq_device=opt_device,
        v_hat_max_device=opt_device,
    )


def save_training_state(
    args: argparse.Namespace, metadata: dict, quantized_model: nn.Module, optimizer: torch.optim.Optimizer
):
    """Save model, optimizer state dict and training metadata to be loaded via load_training_state"""
    if args.save is None:
        return
    rank = torch.distributed.get_rank()
    os.makedirs(args.save, exist_ok=True)
    if rank == 0:
        print(f"Saving snapshot to {args.save}")
        torch.save(metadata, os.path.join(args.save, "metadata.pt"))
    with FullyShardedDataParallel.state_dict_type(quantized_model, StateDictType.LOCAL_STATE_DICT):
        torch.save(quantized_model.state_dict(), os.path.join(args.save, f"quantized_model_state_dict_rank{rank}.pt"))
        # model saves non-quantized weights and dequantized versions of QuantizedWeight; the latter is not necessary
    torch.save(optimizer.state_dict(), os.path.join(args.save, f"optimizer_state_dict_rank{rank}.pt"))
    # optimizer state dict saves statistics QuantizedWeight instances and straight-through buffers
    if args.on_save:
        exec(args.on_save)


def load_training_state(
    args: argparse.Namespace, metadata: dict, quantized_model: nn.Module, optimizer: torch.optim.Optimizer
):
    """Load model, optimizer state dict and metadata saved via save_training_state; update parameters in-place"""
    rank = torch.distributed.get_rank()
    if args.save is None or not os.path.exists(args.save):
        if args.save is not None and rank == 0:
            print(f"No checkpoint found at {args.save}")
    else:
        with FullyShardedDataParallel.state_dict_type(quantized_model, StateDictType.LOCAL_STATE_DICT):
            # this loads non-quantized weights and de-quantized versions of QuantizedWeight instances
            state_dict_ptr = quantized_model.state_dict()
            loaded_state_dict = torch.load(os.path.join(args.save, f"quantized_model_state_dict_rank{rank}.pt"))
            with torch.no_grad():
                for key in state_dict_ptr:
                    state_dict_ptr[key].copy_(loaded_state_dict.pop(key))
                assert len(loaded_state_dict) == 0, f"Unused keys:, {tuple(loaded_state_dict.keys())}"
            del state_dict_ptr, loaded_state_dict

        # v-- loading optimizer state dict also loads all QuantizedWeights and straight-through buffers
        optimizer.load_state_dict(
            torch.load(os.path.join(args.save, f"optimizer_state_dict_rank{rank}.pt"), map_location="cpu")
        )
        metadata.update(torch.load(os.path.join(args.save, "metadata.pt")))
        if args.eval_datasets is not None and metadata["early_stop_on"] not in args.eval_datasets:
            if rank == 0:
                print(f"Stopping criterion {metadata['early_stop_on']} is not in eval_datasets; resetting best loss.")
            metadata["early_stop_on"] = next(iter(args.eval_datasets))
            metadata["best_eval_perplexity"] = float("inf")
            metadata["best_step"] = 0
        if rank == 0:
            print(f"Loaded training state from {args.save}: {metadata}")


def save_model(args: argparse.Namespace, student_model: FullyShardedDataParallel, optimizer: torch.optim.Optimizer):
    """Save model for either P- or PV-Tuning using the appropriate saver"""
    if isinstance(optimizer, StraightThroughAdamW):
        save_pv_model(args, student_model, optimizer)
    else:
        assert any(isinstance(module, QuantizedWeight) for module in student_model.modules())
        save_p_model(args, student_model)


def save_pv_model(
    args: argparse.Namespace, dequantized_model: FullyShardedDataParallel, optimizer: StraightThroughAdamW
):
    """Save consolidated model from PV tuning, can be exported later via convert_legacy_model_format.py"""
    output_path = os.path.join(args.save, "best_model")
    os.makedirs(output_path, exist_ok=True)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    local_quantized_weight_names = set()
    for name, quantized_weight in optimizer.iterate_local_quantized_weights():
        torch.save(quantized_weight, os.path.join(output_path, f"{name}.pth"))
        local_quantized_weight_names.add(name)

    quantized_weight_names_by_rank = [None for _ in range(world_size)] if rank == 0 else None
    torch.distributed.gather_object(local_quantized_weight_names, quantized_weight_names_by_rank, dst=0)

    with FullyShardedDataParallel.state_dict_type(
        dequantized_model,
        StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        model_state_dict = dequantized_model.state_dict()
        if rank == 0:
            all_quantized_weight_names = set()
            for local_quantized_weight_names in quantized_weight_names_by_rank:
                all_quantized_weight_names |= set(local_quantized_weight_names)

            non_quantized_state_dict = dict()
            for name, tensor in model_state_dict.items():
                if name in all_quantized_weight_names:
                    all_quantized_weight_names.remove(name)  # do not save de-quantized versions of quantized weights
                else:
                    non_quantized_state_dict[name] = tensor
            assert len(all_quantized_weight_names) == 0, f"mismatched names: {all_quantized_weight_names}"
            torch.save(non_quantized_state_dict, os.path.join(output_path, "non_quantized_state_dict.pth"))
    torch.distributed.barrier()
    if rank == 0:
        print(f"Saved best model shards to {output_path}")


def save_p_model(args: argparse.Namespace, quantized_model: FullyShardedDataParallel):
    """Save consolidated model state dict from P-only tuning, can be exported via convert_legacy_model_format.py"""
    os.makedirs(args.save, exist_ok=True)
    rank = torch.distributed.get_rank()
    with FullyShardedDataParallel.state_dict_type(
        quantized_model,
        StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        model_state_dict = quantized_model.state_dict()
        if rank == 0:
            torch.save(model_state_dict, os.path.join(args.save, f"best_model_state_dict.pt"))
    torch.distributed.barrier()
    if rank == 0:
        print(f"Saved best model state dict to {os.path.join(args.save, f'best_model_state_dict.pt')}")


def compute_loss_on_batch(
    batch: dict,
    teacher_model: FullyShardedDataParallel,
    student_model: FullyShardedDataParallel,
    *,
    amp_dtype: Optional[torch.dtype],
    max_tokens_per_chunk: Optional[int],
) -> torch.Tensor:
    if max_tokens_per_chunk is not None:  # chunked inference, transformer and lm head must be separate FSDP instances
        with torch.no_grad():
            teacher_hidden_states = teacher_model.base_model(**batch).last_hidden_state
        with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype):
            student_hidden_states = student_model.base_model(**batch).last_hidden_state
            return compute_kl_divergence_loss_values(
                student_hidden_states=student_hidden_states,
                student_lm_head=student_model.get_output_embeddings(),
                teacher_hidden_states=teacher_hidden_states,
                teacher_lm_head=teacher_model.get_output_embeddings(),
                max_tokens_per_chunk=max_tokens_per_chunk,
                checkpoint_last_chunk=False,
                use_reentrant=False,
                determinism_check="none",
            ).mean()

    else:  # combined inference without gradient checkpointing
        with torch.no_grad():
            teacher_logprobs = F.log_softmax(teacher_model(**batch).logits, dim=-1)
        with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype):
            student_logprobs = F.log_softmax(student_model(**batch).logits, dim=-1)
            loss = F.kl_div(
                input=student_logprobs.flatten(0, -2),
                target=teacher_logprobs.flatten(0, -2),
                log_target=True,
                reduction="batchmean",
            ).mean()
        return loss


def compute_validation_perplexities(args: argparse.Namespace, model: nn.Module, eval_datasets: dict):
    rank = torch.distributed.get_rank()
    perplexities = {}
    for dataset_name, eval_dataset in eval_datasets.items():
        if rank == 0:
            print(f"Evaluating perplexity on {dataset_name} ...")
        device = next(model.parameters()).device
        original_dtype = args.load_dtype if args.load_dtype != "auto" else None
        amp_dtype = args.amp_dtype if args.amp_dtype is not None else original_dtype
        ppl = evaluate_perplexity(model, eval_dataset, args.model_seqlen, device=device, amp_dtype=amp_dtype)
        if rank == 0:
            print(f"{dataset_name} perplexity: {ppl:.9f}")
        perplexities[dataset_name] = ppl
    return perplexities


def main():
    assert torch.cuda.is_available() and torch.distributed.is_available()
    torch.distributed.init_process_group()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    parser = argparse.ArgumentParser(add_help=True)
    add_model_args(parser)
    add_data_args(parser)
    add_finetuning_args(parser)
    args = parser.parse_args()

    assert torch.distributed.is_initialized()
    assert args.batch_size is not None, "please specify batch size"
    assert args.batch_size % world_size == 0
    if args.microbatch_size is None:
        args.microbatch_size = args.batch_size // world_size
    assert args.batch_size % (world_size * args.microbatch_size) == 0
    grad_accumulation_steps = args.batch_size // (world_size * args.microbatch_size)

    args.master_dtype = getattr(torch, args.master_dtype)
    args.embed_dtype = getattr(torch, args.embed_dtype) if args.embed_dtype is not None else args.master_dtype
    args.load_dtype = getattr(torch, args.load_dtype) if args.load_dtype != "auto" else "auto"
    args.code_dtype = getattr(torch, args.code_dtype) if args.code_dtype is not None else None
    args.amp_dtype = getattr(torch, args.amp_dtype) if args.amp_dtype is not None else None

    if args.straight_through_buffer_dtype is not None:
        args.straight_through_buffer_dtype = getattr(torch, args.straight_through_buffer_dtype)
    else:
        args.straight_through_buffer_dtype = args.master_dtype

    if args.save_every_steps is not None:
        assert args.save is not None, f"save_every_steps={args.save_every_steps}, but --save path not specified"
    if args.keep_best_model:
        assert args.save is not None, f"--keep_best_model requires --save path"
        assert args.eval_every_steps is not None, f"--keep_best_model requires --eval_every_steps"
        assert args.eval_datasets is not None, f"--keep_best_model requires --eval_datasets"

    if args.wandb and rank == 0:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        wandb.init(config={a: getattr(args, a) for a in dir(args) if not a.startswith("_")})

    if rank == 0:
        print(args)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model)
    assert tokenizer.eos_token_id is not None
    tokenizer.pad_token = tokenizer.eos_token

    with master_rank_first(local=True):
        dataset = prepare_training_dataset(args, tokenizer)
        if args.save_dataset_and_exit is not None:
            if rank == 0:
                dataset.save_to_disk(args.save_dataset_and_exit)

    if args.save_dataset_and_exit is not None:
        torch.distributed.barrier()
        return

    sampler = torch.utils.data.DistributedSampler(
        dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.seed
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.microbatch_size,
        num_workers=args.num_workers,
        sampler=sampler,
        collate_fn=transformers.default_data_collator,
    )
    eval_datasets = {
        dataset_name: get_loaders(
            dataset_name,
            seed=args.seed,
            model_path=args.base_model,
            seqlen=args.model_seqlen,
            eval_mode=True,
        )
        for dataset_name in args.eval_datasets
    }

    use_pv_tuning = args.update_codes and not args.force_dequantize
    if rank == 0:
        print(f"Training {['without', 'with'][use_pv_tuning]} PV-Tuning")

    with one_rank_at_a_time(local=True, group_size=args.limit_parallel_inits):
        teacher_model = load_teacher_model(args, device)
        student_model, named_quantized_params = load_student_model(args, device, dequantize=use_pv_tuning)
        if rank == 0:
            print("Wrapped model:")
            print(student_model)
            for name, param in student_model.named_parameters():
                print(name, param.shape, param.dtype)

    if use_pv_tuning:
        optimizer = create_pv_optimizer(args, student_model, named_quantized_params)
    else:
        optimizer = create_p_optimizer(args, student_model)
    del named_quantized_params

    metadata = dict(
        current_epoch=0,
        microbatches_since_epoch_start=0,
        total_microbatches=0,
        total_optimizer_steps=0,
        loss_numerator=0,
        loss_denominator=0,
        aggregated_loss=float("nan"),
        grad_steps_accumulated=0,
        early_stop_on=next(iter(args.eval_datasets)) if args.eval_datasets else None,
        best_eval_perplexity=float("inf"),
        best_step=0,
    )

    load_training_state(args, metadata, student_model, optimizer)
    torch.distributed.barrier()
    trigger_fsdp_lazy_init_(tokenizer, teacher_model, student_model, device, amp_dtype=args.amp_dtype)

    for current_epoch in range(args.max_epochs):
        if current_epoch < metadata["current_epoch"]:
            continue  # skip finished epochs
        sampler.set_epoch(current_epoch)

        batch_iter = tqdm(train_dataloader, desc=f"Training epoch #{current_epoch}") if rank == 0 else train_dataloader
        for batch_index, batch in enumerate(batch_iter):
            if batch_index <= metadata["microbatches_since_epoch_start"]:
                continue  # skip batches processed before checkpoint
            metadata["microbatches_since_epoch_start"] += 1
            metadata["total_microbatches"] += 1

            batch = {k: v.to(device) for k, v in batch.items()}
            loss = compute_loss_on_batch(
                batch,
                teacher_model,
                student_model,
                amp_dtype=args.amp_dtype,
                max_tokens_per_chunk=args.loss_tokens_per_chunk,
            )

            metadata["loss_numerator"] += loss.item()
            metadata["loss_denominator"] += 1
            metadata["grad_steps_accumulated"] += 1
            if metadata["grad_steps_accumulated"] < grad_accumulation_steps:
                with student_model.no_sync() if args.minimize_sync else nullcontext():
                    (loss / grad_accumulation_steps).backward()
            else:
                (loss / grad_accumulation_steps).backward()
                optimizer.step()
                optimizer.zero_grad()
                metadata["grad_steps_accumulated"] = 0
                metadata["total_optimizer_steps"] += 1

                if args.print_every_steps and metadata["total_optimizer_steps"] % args.print_every_steps == 0:
                    loss_numerator_and_denominator = torch.tensor(
                        [metadata["loss_numerator"], metadata["loss_denominator"]], dtype=torch.float64, device=device
                    )

                    torch.distributed.all_reduce(loss_numerator_and_denominator, op=torch.distributed.ReduceOp.SUM)
                    loss_numerator, loss_denominator = loss_numerator_and_denominator.tolist()
                    metadata["aggregated_loss"] = loss_numerator / loss_denominator
                    metadata["loss_numerator"] = metadata["loss_denominator"] = 0
                    if rank == 0:
                        print(
                            f"epoch {metadata['current_epoch']}\tbatch {batch_index}",
                            f"\t| total updates = {metadata['total_optimizer_steps']}",
                            f"\tloss = {metadata['aggregated_loss']:.9f}",
                        )

                if args.eval_every_steps and metadata["total_optimizer_steps"] % args.eval_every_steps == 0:
                    perplexity_scores = compute_validation_perplexities(args, student_model, eval_datasets)
                    for dataset_name, perplexity in perplexity_scores.items():
                        metadata[f"perplexity_{dataset_name}"] = perplexity
                    metric_name = metadata["early_stop_on"]
                    if perplexity_scores[metric_name] < metadata["best_eval_perplexity"]:
                        if rank == 0:
                            print(f"New best perplexity ({metric_name}) = {perplexity_scores[metric_name]:.9f}")
                        metadata["best_eval_perplexity"] = perplexity_scores[args.eval_datasets[0]]
                        metadata["best_step"] = metadata["total_optimizer_steps"]
                        if args.keep_best_model:
                            save_model(args, student_model, optimizer)
                if args.wandb and rank == 0:
                    wandb.log(metadata, step=metadata["total_microbatches"])
                if args.save_every_steps and metadata["total_optimizer_steps"] % args.save_every_steps == 0:
                    save_training_state(args, metadata, student_model, optimizer)

        metadata["microbatches_since_epoch_start"] = 0
        metadata["current_epoch"] += 1

    save_training_state(args, metadata, student_model, optimizer)


if __name__ == "__main__":
    main()
```

### `lmeval.py`

```python
# Copied from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/__main__.py

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Union

import numpy as np
from lm_eval import evaluator, utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.tasks import include_path, initialize_tasks
from lm_eval.utils import make_table
from transformers import AutoModelForCausalLM

from src.modelutils import load_dequantized_model


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", "-m", default="hf", help="Name of model e.g. `hf`")
    parser.add_argument(
        "--tasks",
        "-t",
        default=None,
        metavar="task1,task2",
        help="To get full list of tasks, use the command lm-eval --tasks list",
    )
    parser.add_argument(
        "--model_args",
        "-a",
        default="",
        help="Comma separated string arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    parser.add_argument(
        "--num_fewshot",
        "-f",
        type=int,
        default=None,
        metavar="N",
        help="Number of examples in few-shot context",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=str,
        default=1,
        metavar="auto|auto:N|N",
        help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        metavar="N",
        help="Maximal batch size to try with --batch_size auto.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        default=None,
        type=str,
        metavar="DIR|DIR/file.json",
        help="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
    parser.add_argument(
        "--limit",
        "-L",
        type=float,
        default=None,
        metavar="N|0<N<1",
        help="Limit the number of examples per task. " "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument(
        "--use_cache",
        "-c",
        type=str,
        default=None,
        metavar="DIR",
        help="A path to a sqlite db file for caching model responses. `None` if not caching.",
    )
    parser.add_argument("--decontamination_ngrams_path", default=None)  # TODO: not used
    parser.add_argument(
        "--check_integrity",
        action="store_true",
        help="Whether to run the relevant part of the test suite for the tasks.",
    )
    parser.add_argument(
        "--write_out",
        "-w",
        action="store_true",
        default=False,
        help="Prints the prompt for the first few documents.",
    )
    parser.add_argument(
        "--log_samples",
        "-s",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis. Use with --output_path.",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        metavar="DIR",
        help="Additional path to include if there are external tasks to include.",
    )
    parser.add_argument(
        "--gen_kwargs",
        default=None,
        help=("String arguments for model generation on greedy_until tasks," " e.g. `temperature=0,top_k=0,top_p=0`."),
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        type=str.upper,
        default="INFO",
        metavar="CRITICAL|ERROR|WARNING|INFO|DEBUG",
        help="Controls the reported logging error level. Set to DEBUG when testing + adding new task configurations for comprehensive log output.",
    )
    parser.add_argument("--aqlm_checkpoint_path", type=str, default=None, help="Path to drop AQLM checkpoint.")
    return parser.parse_args()


def cli_evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    if not args:
        # we allow for args to be passed externally, else we parse them ourselves
        args = parse_eval_args()
    if args.aqlm_checkpoint_path:
        aqlm_checkpoint_path = args.aqlm_checkpoint_path
        # Backup old init
        from_pretrained_old = AutoModelForCausalLM.from_pretrained
        # Define new init
        def from_pretrained_aqlm(*args, **kwargs):
            model = from_pretrained_old(*args, **kwargs)
            model = load_dequantized_model(model, aqlm_checkpoint_path)
            return model

        # Override init
        AutoModelForCausalLM.from_pretrained = staticmethod(from_pretrained_aqlm)

    eval_logger = utils.eval_logger
    eval_logger.setLevel(getattr(logging, f"{args.verbosity}"))
    eval_logger.info(f"Verbosity set to {args.verbosity}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    initialize_tasks(args.verbosity)

    if args.limit:
        eval_logger.warning(
            " --limit SHOULD ONLY BE USED FOR TESTING." "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )
    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")
        include_path(args.include_path)

    if args.tasks is None:
        task_names = ALL_TASKS
    elif args.tasks == "list":
        eval_logger.info("Available Tasks:\n - {}".format("\n - ".join(sorted(ALL_TASKS))))
        sys.exit()
    else:
        if os.path.isdir(args.tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(args.tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            tasks_list = args.tasks.split(",")
            task_names = utils.pattern_match(tasks_list, ALL_TASKS)
            for task in [task for task in tasks_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
            task_missing = [
                task for task in tasks_list if task not in task_names and "*" not in task
            ]  # we don't want errors if a wildcard ("*") task name was used

            if task_missing:
                missing = ", ".join(task_missing)
                eval_logger.error(
                    f"Tasks were not found: {missing}\n"
                    f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
                )
                raise ValueError(
                    f"Tasks not found: {missing}. Try `lm-eval --tasks list` for list of available tasks, or '--verbosity DEBUG' to troubleshoot task registration issues."
                )

    if args.output_path:
        path = Path(args.output_path)
        # check if file or 'dir/results.json' exists
        if path.is_file() or Path(args.output_path).joinpath("results.json").is_file():
            eval_logger.warning(f"File already exists at {path}. Results will be overwritten.")
            output_path_file = path.joinpath("results.json")
            assert not path.is_file(), "File already exists"
        # if path json then get parent dir
        elif path.suffix in (".json", ".jsonl"):
            output_path_file = path
            path.parent.mkdir(parents=True, exist_ok=True)
            path = path.parent
        else:
            path.mkdir(parents=True, exist_ok=True)
            output_path_file = path.joinpath("results.json")
    elif args.log_samples and not args.output_path:
        assert args.output_path, "Specify --output_path"

    eval_logger.info(f"Selected Tasks: {task_names}")

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        use_cache=args.use_cache,
        limit=args.limit,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        log_samples=args.log_samples,
        gen_kwargs=args.gen_kwargs,
    )

    if results is not None:
        if args.log_samples:
            samples = results.pop("samples")
        dumped = json.dumps(results, indent=2, default=_handle_non_serializable, ensure_ascii=False)
        if args.show_config:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        if args.output_path:
            output_path_file.open("w").write(dumped)

            if args.log_samples:
                for task_name, config in results["configs"].items():
                    output_name = "{}_{}".format(re.sub("/|=", "__", args.model_args), task_name)
                    filename = path.joinpath(f"{output_name}.jsonl")
                    samples_dumped = json.dumps(
                        samples[task_name],
                        indent=2,
                        default=_handle_non_serializable,
                        ensure_ascii=False,
                    )
                    filename.open("w").write(samples_dumped)

        print(
            f"{args.model} ({args.model_args}), gen_kwargs: ({args.gen_kwargs}), limit: {args.limit}, num_fewshot: {args.num_fewshot}, "
            f"batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
        )
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))


cli_evaluate()
```

### `main.py`

```python
import os
import time
from argparse import Namespace
from itertools import chain
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from tqdm import trange
from tqdm.auto import trange
from transformers import PreTrainedModel

from aq_engine import AQEngine
from src.aq import QuantizedLinear
from src.datautils import get_loaders
from src.finetune import finetune_groupwise
from src.modelutils import (
    FALCON_TYPES,
    find_sublayers,
    get_layers,
    get_lm_logits,
    get_model,
    get_model_head_with_norm,
    get_sequential_groups,
    save_not_quantized_weights,
)
from src.utils import using_tf32

try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False


def quantize_model(model: PreTrainedModel, args: Namespace):
    """main entry point to functions for model quantization"""
    tick = time.time()
    print("Loading data ...")
    data = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model_path=args.model_path,
        seqlen=args.model_seqlen,
        use_fast_tokenizer=args.use_fast_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    if args.val_size > 0:
        all_ids = torch.randperm(len(data))
        train_ids, val_ids = all_ids[args.val_size :], all_ids[: args.val_size]
        train_data = [data[i] for i in train_ids]
        val_data = [data[i] for i in val_ids]
    else:
        train_data = data
        val_data = None

    results = quantize_aq(model, train_data, val_data, args)
    print(f"quantization time: {time.time() - tick:.1f}")
    return results


@torch.no_grad()
def get_inps(
    model: PreTrainedModel,
    data: Sequence,
    model_seqlen: int,
    devices: Sequence[torch.device],
    offload_activations: bool,
) -> Tuple[Sequence[torch.Tensor], Dict]:
    """
    mocks model launch to collect inputs to the first model layer
    :returns: a list of torch tensors with activations for each device in args.devices.
    Each tensor has shape [nsample_per_device, seq_len, hid_size]
    """
    print("catching layer inputs from data", flush=True)
    layers = get_layers(model)
    device = devices[0] if not offload_activations else torch.device("cpu")

    if isinstance(data, torch.Tensor) and data.shape[0] == 1:  # given a single long tensor, split it into sequences
        assert data.ndim == 2, "data must be either a single tensor with a long sequence or a list of pre-cut sequences"
        num_sequences, num_tokens_dropped = data.numel() // model_seqlen, data.numel() % model_seqlen
        data = [data[:, i * model_seqlen : (i + 1) * model_seqlen].to(device) for i in range(num_sequences)]
        print(f"Got {len(data)} sequences of {model_seqlen} tokens, dropped last {num_tokens_dropped} tokens")
        del num_sequences, num_tokens_dropped

    assert all(sequence.shape[1] == model_seqlen for sequence in data)

    emb = model.get_input_embeddings()
    emb_device = emb.weight.device
    if emb_device.type != "cuda":
        emb = emb.to(device)
        # opt has other embeddings
        if model.config.model_type == "opt":
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
            if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
                model.model.decoder.project_in = model.model.decoder.project_in.to(device)
    device = emb.weight.device  # now default device is the one where the embeddings are.
    layer_device = next(layers[0].parameters()).device
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    nsamples_per_device = (len(data) - 1) // len(devices) + 1
    inps = [
        torch.zeros(
            (min(nsamples_per_device, len(data) - i * nsamples_per_device), model_seqlen, model.config.hidden_size),
            dtype=dtype,
            device=devices[i] if not offload_activations else "cpu",
            pin_memory=offload_activations,
        )
        for i in range(len(devices))
    ]
    forward_arg_names = ["attention_mask", "position_ids"]
    if model.config.model_type.lower() in FALCON_TYPES:
        forward_arg_names.append("alibi")

    cache = {"i": 0, "alibi": None}

    class CatcherExit(Exception):
        pass

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"] // nsamples_per_device][cache["i"] % nsamples_per_device] = inp
            cache["i"] += 1
            for forward_arg_name in forward_arg_names:
                cache[forward_arg_name] = kwargs.get(forward_arg_name)
            raise CatcherExit()

    layers[0] = Catcher(layers[0])
    saved_num_threads = torch.get_num_threads()
    torch.set_num_threads(min(16, saved_num_threads))
    for batch_inps in data:
        try:
            if isinstance(batch_inps, (list, tuple)):
                batch_inps, *_ = batch_inps
            batch_inps = batch_inps.to(device)
            # call model.forward to trigger the Catcher
            model(batch_inps, attention_mask=torch.ones_like(batch_inps))
        except CatcherExit:
            pass  # exit after catcher finished without running the rest of the model layers

    torch.set_num_threads(saved_num_threads)
    layers[0] = layers[0].module

    layers[0] = layers[0].to(layer_device)
    model.get_input_embeddings().to(emb_device)
    if model.config.model_type == "opt":
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(emb_device)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(emb_device)
    torch.cuda.empty_cache()

    forward_args = {k: cache[k] for k in forward_arg_names}
    assert cache["i"] == sum(len(inp_tensor) for inp_tensor in inps), "internal error: found empty rows in inps"
    return inps, forward_args


@torch.no_grad()
def quantize_aq(model: PreTrainedModel, data: Sequence, val_data: Optional[Sequence], args: Namespace):
    assert not torch.backends.cuda.matmul.allow_tf32
    print("\nStarting AQ quantization ...")
    inps, forward_args = get_inps(model, data, args.model_seqlen, args.devices, args.offload_activations)
    outs = [torch.zeros_like(inp_tensor, pin_memory=inp_tensor.is_pinned()) for inp_tensor in inps]

    if val_data:
        run_validation = True
        val_inps, _ = get_inps(model, val_data, args.model_seqlen, args.devices, args.offload_activations)
        val_outs = [torch.zeros_like(inp_tensor, pin_memory=inp_tensor.is_pinned()) for inp_tensor in val_inps]
    else:
        run_validation = False
        val_inps, val_outs = None, None

    use_cache = model.config.use_cache
    num_codebooks = args.num_codebooks
    model.config.use_cache = False

    quantizers = {}
    overall_bits = 0
    number_of_quantized_params = 0
    layers = get_layers(model)

    for layer_index in range(len(layers)):
        print(f"\n---------------- Layer {layer_index} of {len(layers)} ----------------")
        stats_payload = {}
        start_time = time.time()

        # quantized layer will return there
        layer_device_original = next(layers[layer_index].parameters()).device
        # backup layer dtype
        layer_dtype_original = next(layers[layer_index].parameters()).dtype
        print(f"{layer_device_original=}")
        layer = layers[layer_index].to(args.devices[0])
        for k, v in forward_args.items():
            forward_args[k] = v.to(args.devices[0]) if isinstance(v, torch.Tensor) else v

        if args.true_sequential:
            sequential = get_sequential_groups(model)
        else:
            sequential = [list(find_sublayers(layer).keys())]

        loaded_layer = False
        if args.resume:
            assert args.save is not None, "using --resume requires a --save path to resume from"
            layer_save_path = os.path.join(args.save, f"{layer_index}.pth")
            if os.path.exists(layer_save_path):
                print(f"Loading layer {layer_index} from {layer_save_path}")
                layer = torch.load(layer_save_path, map_location=args.devices[0])
                loaded_layer = True

        # prepare validation  outputs
        if run_validation and not loaded_layer:  # note: if we skip validation, val_outs will still be updated later
            if len(args.devices) == 1:
                assert len(val_inps) == len(val_outs) == 1
                update_outs(layer, val_inps[0], val_outs[0], compute_mse=not args.skip_out_loss, **forward_args)
            else:
                update_outs_parallel(
                    args.devices, layer, val_inps, val_outs, compute_mse=not args.skip_out_loss, **forward_args
                )

        for names in sequential:
            if loaded_layer:
                print("Skipping quantization: loaded a previously quantized layer")
                break
            if len(args.devices) == 1:
                assert len(inps) == len(outs) == 1  # number of per-device inputs/outputs
                aq_handlers = init_aq_engines(
                    layer,
                    [
                        name
                        for name in names
                        if ((".gate" not in name.lower()) or ("mixtral" not in model.config.model_type.lower()))
                    ],
                    inps[0],
                    outs[0],
                    **forward_args,
                )
            else:
                aq_handlers = init_aq_engines_parallel(
                    args.devices,
                    layer,
                    [
                        name
                        for name in names
                        if ((".gate" not in name.lower()) or ("mixtral" not in model.config.model_type.lower()))
                    ],
                    inps,
                    outs,
                    **forward_args,
                )
            for sublayer_name in aq_handlers.keys():
                print(f"Quantizing module {sublayer_name} of layer {layer_index}")
                if "mixtral" in model.config.model_type.lower() and args.mix_compression:
                    assert "mixtral" in model.config.model_type.lower()
                    if "self_attn" in sublayer_name.lower():
                        args.num_codebooks = 2 * num_codebooks
                    else:
                        args.num_codebooks = num_codebooks
                    print(sublayer_name.lower(), " mixtral num codebooks", args.num_codebooks)
                quantized_weight = aq_handlers[sublayer_name].quantize(args=args, verbose=True)

                with torch.no_grad():
                    assert aq_handlers[sublayer_name].layer.weight in set(
                        layer.parameters()
                    )  # test that this is not a replica

                    new_linear = QuantizedLinear(quantized_weight, aq_handlers[sublayer_name].layer.bias)
                    if args.use_checkpointing:
                        new_linear.use_checkpoint = True
                        print("ENABLED CHECKPOINTING FOR", sublayer_name)
                    found_original = False
                    for submodule in layer.modules():
                        for child_name, child_module in submodule.named_children():
                            if child_module is aq_handlers[sublayer_name].layer:
                                setattr(submodule, child_name, new_linear)
                                found_original = True  # note: do not break to handle tied layers

                    assert found_original, f"could not find {sublayer_name}"

                weight_avg_bits = quantized_weight.estimate_nbits_per_parameter()
                overall_bits += int(weight_avg_bits * torch.numel(aq_handlers[sublayer_name].layer.weight.data))
                number_of_quantized_params += torch.numel(aq_handlers[sublayer_name].layer.weight.data)
                print("curent_avg_bits", overall_bits / number_of_quantized_params)
                quantizers["model.layers.%d.%s" % (layer_index, sublayer_name)] = ()  # to be updated

            del aq_handlers
            assert not loaded_layer

            print("PREPARING TO FINETUNE")
            print(layer)
            layer = layer.to(dtype=torch.float32)
            with using_tf32(enabled=True):
                layer = finetune_groupwise(
                    layer=layer,
                    train_inps=inps,
                    train_outs=outs,
                    args=args,
                    valid_inps=val_inps,
                    valid_outs=val_outs,
                    **forward_args,
                )
            layer = layer.to(dtype=layer_dtype_original)
            print("FINISHED FINETUNING")

        if args.save and not loaded_layer:
            os.makedirs(args.save, exist_ok=True)
            layer_save_path = os.path.join(args.save, f"{layer_index}.pth")
            print(f"Saving layer {layer_index}... to {layer_save_path}")
            torch.save(layer, layer_save_path)
            if args.on_save:
                exec(args.on_save)  # a callback e.g. to save progress in slurm or similar distributed infrastructure

        should_compute_mse = not (args.skip_out_loss or loaded_layer)
        if len(args.devices) == 1:
            assert len(inps) == len(outs) == 1
            out_losses = update_outs(layer, inps[0], outs[0], compute_mse=should_compute_mse, **forward_args)
        else:
            out_losses = update_outs_parallel(
                args.devices, layer, inps, outs, compute_mse=should_compute_mse, **forward_args
            )
        stats_payload["out_loss"] = torch.mean(torch.Tensor(out_losses)).item()

        if run_validation:
            if len(args.devices) == 1:
                assert len(val_inps) == len(val_outs) == 1
                out_val_losses = update_outs(
                    layer, val_inps[0], val_outs[0], compute_mse=should_compute_mse, **forward_args
                )
            else:
                out_val_losses = update_outs_parallel(
                    args.devices, layer, val_inps, val_outs, compute_mse=should_compute_mse, **forward_args
                )
            stats_payload["out_val_loss"] = torch.mean(torch.Tensor(out_val_losses)).item()

        layers[layer_index] = layer.to(layer_device_original)
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        if run_validation:
            val_inps, val_outs = val_outs, val_inps

        # Logging
        stats_payload["layer_time"] = time.time() - start_time
        stats_payload["Step"] = layer_index
        if args.wandb:
            wandb.log(stats_payload, step=layer_index)
        if not loaded_layer:
            print(stats_payload)

    print("=====================\nFinal stats:")
    if args.save:
        torch.save(vars(args), os.path.join(args.save, "args.pt"))
        save_not_quantized_weights(model, args.save)
        if args.on_save:
            exec(args.on_save)  # a callback e.g. to save progress in slurm or similar distributed infrastructure

    if args.wandb:
        wandb.log({"max_cuda_mem_quantize": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
        if number_of_quantized_params > 0:  # do not report avg bits if we load all pre-quantized layers via --resume
            wandb.log({"Avg_bits": overall_bits / number_of_quantized_params})
    model.config.use_cache = use_cache
    print(f"quantize: {torch.cuda.max_memory_allocated()=:,}")
    return quantizers


@torch.no_grad()
def perplexity_eval(model: PreTrainedModel, testenc: torch.LongTensor, args: Namespace) -> float:
    print(f"\nEvaluating perplexity for {args.dataset_name} dataset ...")

    nsamples = testenc.numel() // args.model_seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False

    inps, forward_args = get_inps(model, testenc, args.model_seqlen, args.devices, args.offload_activations)
    outs = [torch.zeros_like(inp_tensor, pin_memory=inp_tensor.is_pinned()) for inp_tensor in inps]
    device = args.devices[0]
    for k, v in forward_args.items():
        forward_args[k] = v.to(device) if isinstance(v, torch.Tensor) else v

    layers = get_layers(model)
    for i in trange(len(layers), desc="processing eval data by layer"):
        layer = layers[i].to(device)
        if len(args.devices) == 1:
            assert len(inps) == len(outs) == 1
            update_outs(layer, inps[0], outs[0], compute_mse=False, **forward_args)
        else:
            update_outs_parallel(args.devices, layer, inps, outs, compute_mse=False, **forward_args)
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    get_model_head_with_norm(model).to(device)
    testenc = testenc.to(device)
    nsamples_per_device = len(inps[0])
    assert len(set(map(len, inps[:-1]))) <= 1 and len(inps[-1]) <= len(inps[0])

    nlls = []
    for i in range(nsamples):
        inp = inps[i // nsamples_per_device][i % nsamples_per_device].to(args.devices[0], non_blocking=True)
        lm_logits = get_lm_logits(inp.to(device), model)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * args.model_seqlen) : ((i + 1) * args.model_seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * args.model_seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * args.model_seqlen)).item()
    print(f"\n{args.dataset_name} perplexity = {ppl:.4f}\n")

    get_model_head_with_norm(model).to(torch.device("cpu"))

    if args.wandb:
        wandb.log({args.dataset_name: ppl})

    model.config.use_cache = use_cache
    return ppl


@torch.no_grad()
def init_aq_engines(
    layer: nn.Module,
    names: Sequence[str],
    inps_tensor: torch.Tensor,
    outs_tensor: torch.Tensor,
    **forward_args: Dict[str, Any],
) -> Dict[str, AQEngine]:
    """
    Create a dictionary of AQUtil instances for each quantized layer;
    Run forward pass on each sample in inps_tensor; write output activations to outs_tensor (in-plance)
    Accumulate XTX to each one of aq_handlers
    :param layer: transformer layer with one or more linear layer to be quantized
    :param names: a list/tuple of string names for linear sub-layers inside :layer: that shall be quantized
    :param inps_tensor: a tensor of input activations, [nsamples_per_device, seq_len, hidden_size]
    :param outs_tensor: a tensor to write output activations into, [nsamples_per_device, seq_len, hidden_size]
    :param forward_args: additional keyword arguments, e.g. attention mask
    :returns: a dictionary where keys are full layer names and values are AQUtil instances ready to run .quantize
    """
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    all_sublayers = find_sublayers(layer)
    subset = {name: all_sublayers[name] for name in names}
    assert len(subset) > 0
    aq_handlers = {}
    for sublayer_name in subset:
        aq_handlers[sublayer_name] = AQEngine(subset[sublayer_name])

    # wrap all quantized sub-layers with a wrapper that accumulates inputs on forward
    # note: the code below uses wrappers instead of hooks because hooks cause bugs in multi-gpu code
    wrapped_layer_to_hander = {aq_handler.layer: aq_handler for aq_handler in aq_handlers.values()}
    for module in list(layer.modules()):
        for child_name, child in list(module.named_children()):
            if child in wrapped_layer_to_hander:
                setattr(module, child_name, _LayerWrapperThatAccumulatesXTX(child, wrapped_layer_to_hander[child]))

    # compute output activations and accumulate XTX
    for j in trange(len(inps_tensor), desc="calc outs before quantization", leave=False):
        outs_tensor[j].copy_(
            layer(inps_tensor[j].to(device).unsqueeze(0), **forward_args)[0].view_as(outs_tensor[j]), non_blocking=True
        )

    # remove wrappers
    for module in list(layer.modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, _LayerWrapperThatAccumulatesXTX):
                setattr(module, child_name, child.wrapped_layer)
    return aq_handlers


class _LayerWrapperThatAccumulatesXTX(nn.Module):
    def __init__(self, layer: nn.Module, aq_handler: AQEngine):
        super().__init__()
        self.wrapped_layer, self.aq_handler = layer, aq_handler

    def forward(self, input, *args, **kwargs):
        self.aq_handler.add_batch(input)
        return self.wrapped_layer(input, *args, **kwargs)


@torch.no_grad()
def init_aq_engines_parallel(
    devices: Sequence[torch.device],
    layer: nn.Module,
    names: Sequence[str],
    inps: Sequence[torch.Tensor],
    outs: Sequence[torch.Tensor],
    **forward_args,
):
    """Parallel version of init_aq_engines; works on lists of input/output tensors"""
    layer_replicas = torch.nn.parallel.replicate(layer, devices=devices, detach=True)
    layer_replicas[0] = layer  # this ensures that aq_handlers returned by 0-th replica operate on the main layer
    funcs_by_device = [init_aq_engines for _ in devices]
    inputs_by_device = []
    kwargs_by_device = []
    for i in range(len(devices)):
        inputs_by_device.append((layer_replicas[i], names, inps[i], outs[i]))
        kwargs_by_device.append(
            {
                k: (v.to(devices[i], non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in forward_args.items()
            }
        )
    aq_handles_by_device: Sequence[Dict[str, AQEngine]] = torch.nn.parallel.parallel_apply(
        funcs_by_device, inputs_by_device, kwargs_by_device, devices=devices
    )
    aq_handlers = aq_handles_by_device[0]
    for key, aq_handler in aq_handlers.items():
        replica_handlers = [device_aq_handlers[key] for device_aq_handlers in aq_handles_by_device]
        replica_nsamples = [replica_handler.nsamples for replica_handler in replica_handlers]
        total_nsamples = sum(replica_nsamples)
        aq_handler.XTX = sum(
            (replica_handlers[i].XTX * (replica_nsamples[i] / total_nsamples)).to(devices[0], non_blocking=True)
            for i in range(len(devices))
        )
        aq_handler.nsamples = total_nsamples
    return aq_handlers


@torch.no_grad()
def update_outs(
    layer: nn.Module, inps_tensor: torch.Tensor, outs_tensor: torch.Tensor, compute_mse: bool, **forward_args
) -> Sequence[float]:
    """
    Update outs_tensor with new activations and optionally compute sample-wise mse loss with previous activations
    :param layer: transformer layer with one or more linear layer to be quantized
    :param inps_tensor: a tensor of input activations, [nsamples_per_device, seq_len, hidden_size]
    :param outs_tensor: a tensor to write output activations into, [nsamples_per_device, seq_len, hidden_size]
    :note: outs_tensor must contain previous activations with which to compute MSE loss
    :param compute_mse: if True, return a list of sample-wise mse losses; if False, return an empty sequence
    :param forward_args: additional keyword arguments, e.g. attention mask
    :returns: a list of mean squared errors for each sequence
    """
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    out_losses = []
    for j in trange(len(inps_tensor), desc="calc outs after quantization", leave=False):
        outs_batch = layer(inps_tensor[j].to(device).unsqueeze(0), **forward_args)[0]
        if compute_mse:
            batch_size = outs_batch.shape[0]
            outs_batch_loss = (
                (outs_batch - outs_tensor[j].to(device)).float().square().view(batch_size, -1).mean(dim=-1)
            )
            outs_batch_loss /= outs_batch.float().square().view(batch_size, -1).mean(dim=-1).clamp(min=1e-6)
            outs_batch_loss = outs_batch_loss.mean()
            out_losses.append(outs_batch_loss.item())
        outs_tensor[j].copy_(outs_batch.reshape_as(outs_tensor[j]), non_blocking=True)
    return out_losses


@torch.no_grad()
def update_outs_parallel(
    devices: Sequence[torch.device],
    layer: nn.Module,
    inps: Sequence[torch.Tensor],
    outs: Sequence[torch.Tensor],
    compute_mse: bool,
    **forward_args,
) -> Sequence[float]:
    """Parallel version of update_outs_and_compute_losses; works on lists of input/output tensors"""
    layer_replicas = torch.nn.parallel.replicate(layer, devices=devices, detach=True)
    funcs_by_device = [update_outs for _ in devices]
    inputs_by_device = []
    kwargs_by_device = []
    for i in range(len(devices)):
        inputs_by_device.append((layer_replicas[i], inps[i], outs[i], compute_mse))
        kwargs_by_device.append(
            {
                k: (v.to(devices[i], non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in forward_args.items()
            }
        )
    out_losses_by_device: Sequence[Sequence[float]] = torch.nn.parallel.parallel_apply(
        funcs_by_device, inputs_by_device, kwargs_by_device, devices=devices
    )
    return list(chain(*out_losses_by_device))


def main():
    import argparse

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "model_path",
        type=str,
        help="path to llama model to load, as in LlamaForCausalLM.from_pretrained()",
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset name [c4, pajama] or path to data where to extract calibration data from.",
    )
    parser.add_argument(
        "--new_eval",
        action="store_true",
        help="if this is set, evaluate on new (and slightly more realistic!) val dataset versions",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=None,
        help="Number of calibration data samples.If None take all calibration data.",
    )
    parser.add_argument(
        "--model_seqlen",
        type=int,
        default=4096,
        help="Model seqlen and calibration data context length.",
    )
    parser.add_argument(
        "--use_checkpointing",
        action="store_true",
        help="Whether to use checkpoining in finetuning",
    )
    parser.add_argument(
        "--mix_compression",
        action="store_true",
        help="Compress .self_attn in 4 bits, .block_sparse_moe.experts to 2.3 for mixtral.",
    )
    parser.add_argument("--load", type=str, default=None, help="Path to load quantized statistics.")
    parser.add_argument("--save", type=str, default=None, help="Path to save quantized statistics.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If true, search for previously saved layers and reuse them to save time. Requires --save path.",
    )
    parser.add_argument(
        "--on_save",
        type=str,
        default=None,
        help="Optional callback (python code string) to call after each saved layer. Example: when "
        "training on preemptible compute, upload partially quantized model and --resume later.",
    )
    parser.add_argument("--devices", metavar="N", type=str, nargs="+", default=None, help="List of devices")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model in",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for calibration data and initialization. "
        "Note that the main training is not strictly deterministic.",
    )
    parser.add_argument(
        "--skip_out_loss",
        action="store_true",
        help="Whether to skip computation of out loss.",
    )
    parser.add_argument(
        "--offload_activations",
        action="store_true",
        help="Offload activations to RAM to save GPU memory.",
    )
    parser.add_argument(
        "--true-sequential",
        action="store_true",
        help="Whether to run in true sequential model.",
    )
    parser.add_argument(
        "--num_codebooks",
        type=int,
        default=1,
        help="#Number of codebooks per layer",
    )
    parser.add_argument(
        "--nbits_per_codebook",
        type=int,
        default=16,
        help="each codebook will contain 2 ** nbits_per_codebook vectors",
    )
    parser.add_argument(
        "--out_group_size",
        type=int,
        default=1,
        help="How many output units are quantized together",
    )
    parser.add_argument(
        "--in_group_size",
        type=int,
        default=8,
        help="How many input features are quantized together",
    )
    parser.add_argument(
        "--scale_nbits",
        type=int,
        default=0,
        help="Number of bits dedicated to the learnable group-wise scale. 0 means do not use group-wise scales "
        "(still has row-wise scales), 1-15 means using per-group scales quantized to this many bits, "
        "16+ means use per-group scales but do not quantize them",
    )
    parser.add_argument(
        "--codebook_value_nbits",
        type=int,
        default=16,
        help="If below 16, quantize the values in each codebook with the specified number of bits",
    )
    parser.add_argument(
        "--codebook_value_num_groups",
        type=int,
        default=1,
        help="Split codebook vectors into this many groups for quantizations. Only used when quantized codebooks.",
    )

    parser.add_argument(
        "--init_max_iter",
        type=int,
        default=100,
        help="Number of K-Means iterations used to initialize codebooks and codes",
    )
    parser.add_argument(
        "--use_faiss",
        action="store_true",
        help="Whether to use faiss.Kmeans when initializing codebooks and codes",
    )
    parser.add_argument(
        "--init_max_points_per_centroid",
        type=int,
        default=None,
        help="During K-means initialzation, sample (this_many * 2 ^ nbits_per_codebook) points for training K-means",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=1,
        help="Keep top-(this_many) best candidates for each codebook when finding optimal codes",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1000,
        help="Maximum number of beam search rounds before the optimization is forcibly stopped.",
    )
    parser.add_argument(
        "--relative_mse_tolerance",
        type=float,
        default=None,
        help="Stop training when (current_epoch_mse / previous_epoch_mse) > (1 - relative_mse_tolerance)",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=100,
        help="Run (this many) Adam updates before every beam search round",
    )
    parser.add_argument(
        "--finetune_max_epochs",
        type=int,
        default=5,
        help="Run this many passes over training data when doing finetuning; No finetuning if set to 0.",
    )
    parser.add_argument(
        "--finetune_early_stop",
        type=int,
        default=3,
        help="Terminate finetuning if loss doesn't improve after this number of epochs.",
    )
    parser.add_argument(
        "--finetune_lr",
        type=float,
        default=1e-5,
        help="Finetuning learning rate",
    )
    parser.add_argument(
        "--finetune_batch_size",
        type=int,
        default=1,
        help="(finetuning only) train on batches of this many sequences, globally across all GPUs",
    )
    parser.add_argument(
        "--finetune_adam_beta1",
        type=float,
        default=0.9,
        help="Finetuning adam_beta1",
    )
    parser.add_argument(
        "--finetune_adam_beta2",
        type=float,
        default=0.95,
        help="Finetuning adam_beta2",
    )
    parser.add_argument("--finetune_keep_best", action="store_true")
    parser.add_argument(
        "--local_batch_size",
        type=int,
        default=None,
        help="(finetuning only) Per-device and per-forward-pass batch size used to accumulate global --batch_size",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=0,
        help="Num validation sequences",
    )
    parser.add_argument(
        "--print_frequency",
        type=int,
        default=10,
        help="Print Adam progress after each print_frequency updates",
    )
    parser.add_argument("--wandb", action="store_true", help="Whether to use wandb or store locally.")
    parser.add_argument(
        "--no_quant",
        action="store_true",
        help="Skip model quantization and immediately evaluate the loaded model",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=[None, "eager", "flash_attention_2", "sdpa"],
        help="Attention implementation.",
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        action="store_true",
        help="Whether to use fast tokenizer (some models have only fast tokenizer).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code.",
    )

    torch.set_num_threads(min(16, torch.get_num_threads()))
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    if args.devices is None:
        if torch.cuda.is_available():
            args.devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        else:
            args.devices = [torch.device("cpu")]
    else:
        args.devices = [torch.device(device_str) for device_str in args.devices]
    assert all(isinstance(device, torch.device) for device in args.devices)

    # validate val size
    if args.nsamples is not None:
        assert args.val_size < args.nsamples, "Number of validation set must be smaller than train + val"

    if args.wandb:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        args.exp_name = (
            os.environ.get("WANDB_NAME", "AQ")
            + f"_num_codebooks_{args.num_codebooks}"
            + f"_out_group_size_{args.out_group_size}"
            + f"_in_group_size_{args.in_group_size}"
            + f"_nbits_per_codebook_{args.nbits_per_codebook}"
            + f"_codebook_value_nbits_{args.codebook_value_nbits}"
            + f"_codebook_value_num_groups_{args.codebook_value_num_groups}"
            + f"_scale_nbits_{args.scale_nbits}"
            + f"_steps_per_epoch_{args.steps_per_epoch}"
            + f"_init_max_iter{args.init_max_iter}"
            + f"_{len(args.devices)}gpus"
        )
        args.group_size = args.in_group_size * args.out_group_size

        wandb.init(
            config={a: getattr(args, a) for a in dir(args) if not a.startswith("_")},
        )

    print("\n============ Load model... ============")
    model = get_model(
        args.model_path,
        args.load,
        args.dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    ).train(False)

    if not args.load and not args.no_quant:
        print("\n============ Quantizing model... ============")
        quantize_model(model, args)

    print("\n============ Evaluating perplexity... ============")
    torch.cuda.reset_peak_memory_stats()
    datasets = ["wikitext2", "c4"]
    if args.new_eval:
        datasets = ["wikitext2", "c4-new"]
    for dataset in datasets:
        testloader = get_loaders(
            dataset,
            seed=args.seed,
            model_path=args.model_path,
            seqlen=args.model_seqlen,
            eval_mode=True,
            use_fast_tokenizer=args.use_fast_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )
        args.dataset_name = dataset
        perplexity_eval(model, testloader, args)

    print(f"eval: {torch.cuda.max_memory_allocated()=:,}")
    if args.wandb:
        wandb.log({"max_cuda_mem_eval": round(torch.cuda.max_memory_allocated() / 1e9, 2)})


if __name__ == "__main__":
    main()
```

### `requirements.txt`

```text
safetensors==0.4.3
datasets==2.19.0
sentencepiece==0.2.0
torch>=2.3.0
numpy>=1.26.4
transformers==4.40.1
accelerate==0.29.3
```

### `benchmark/benchmark_generate_cpu.py`

```python
import argparse
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import time
import warnings

warnings.filterwarnings("ignore")

import torch

torch.set_num_threads(8)
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num_codebooks",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--in_group_size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--nbits_per_codebook",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=1,
        help="Number of warmup iterations.",
    )
    parser.add_argument(
        "--benchmark_iters",
        type=int,
        default=3,
        help="Number of benchmark iterations.",
    )
    parser.add_argument(
        "--input_length",
        type=int,
        default=1,
        help="Input length.",
    )
    parser.add_argument(
        "--output_length",
        type=int,
        default=128,
        help="Output length.",
    )
    args = parser.parse_args()

    device = "cpu"

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.float32)
    if args.num_codebooks is not None:
        config.aqlm["num_codebooks"] = args.num_codebooks
    if args.in_group_size is not None:
        config.aqlm["in_group_size"] = args.in_group_size
    if args.nbits_per_codebook is not None:
        config.aqlm["nbits_per_codebook"] = args.nbits_per_codebook

    real_num_layers = config.num_hidden_layers
    if "meta-llama" in args.model:
        config.num_hidden_layers = 1
    aqlm_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.float32)

    if "meta-llama" in args.model:
        aqlm_model.config.num_hidden_layers = real_num_layers
        layer = aqlm_model.model.layers[0]
        aqlm_model.model.layers = nn.ModuleList([])
        for i in range(real_num_layers):
            another_layer = type(layer)(config, i)

            another_layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data
            another_layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data
            another_layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data
            another_layer.self_attn.o_proj.weight.data = layer.self_attn.o_proj.weight.data
            another_layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data
            another_layer.mlp.down_proj.weight.data = layer.mlp.down_proj.weight.data
            another_layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data

            another_layer.self_attn.layer_idx = i
            aqlm_model.model.layers.append(another_layer)

        aqlm_model.model.config.num_hidden_layers = real_num_layers

    prompt = torch.randint(low=0, high=aqlm_model.config.vocab_size, size=(1, args.input_length), device=device)

    for i in range(args.warmup_iters + args.benchmark_iters):
        aqlm_model.generate(prompt, min_new_tokens=args.output_length, max_new_tokens=args.output_length)
        if i == args.warmup_iters - 1:
            t_s = time.perf_counter()
    t_e = time.perf_counter()

    tokens_per_second = args.benchmark_iters * args.output_length / (t_e - t_s)
    print(f"<Tokens per second> = {tokens_per_second:.3f}")
```

### `benchmark/generate_benchmark.py`

```python
import argparse
import os
import time
import warnings

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from tqdm import trange
from transformers import AutoConfig, AutoModelForCausalLM

if __name__ == "__main__":
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=1,
        help="Number of warmup iterations.",
    )
    parser.add_argument(
        "--benchmark_iters",
        type=int,
        default=10,
        help="Number of benchmark iterations.",
    )
    parser.add_argument(
        "--input_length",
        type=int,
        default=1,
        help="Input length.",
    )
    parser.add_argument(
        "--output_length",
        type=int,
        default=128,
        help="Output length.",
    )
    parser.add_argument(
        "--real_model",
        action="store_true",
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
    )

    args = parser.parse_args()


def load_model(model_name, device="cuda"):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
    ).to(device)


def load_shared_model(model_name, device="cuda"):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = config.num_hidden_layers
    config.num_hidden_layers = 1
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.float16).to(device)
    layer = model.model.layers[0]
    for i in trange(1, num_layers, desc="Copying block parameters"):
        new_layer = type(layer)(model.config, i).to(device)
        for new_layer_param, layer_param in zip(new_layer.parameters(), layer.parameters()):
            new_layer_param.data = layer_param.data
        new_layer.self_attn.layer_idx = i
        model.model.layers.append(new_layer)
    return model


if __name__ == "__main__":
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    parser = argparse.ArgumentParser(add_help=True)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    if args.real_model:
        aqlm_model = load_model(args.model, device)
    else:
        aqlm_model = load_shared_model(args.model, device)

    prompt = torch.randint(low=0, high=aqlm_model.config.vocab_size, size=(1, args.input_length), device=device)

    for i in range(args.warmup_iters + args.benchmark_iters):
        output = aqlm_model.generate(prompt, min_new_tokens=args.output_length, max_new_tokens=args.output_length)
        if i == args.warmup_iters - 1:
            torch.cuda.synchronize(device)
            t_s = time.perf_counter()
    torch.cuda.synchronize(device)
    t_e = time.perf_counter()

    tokens_per_second = args.benchmark_iters * args.output_length / (t_e - t_s)
    print(f"<Tokens per second> = {tokens_per_second:.2f}")
```

### `benchmark/matmul_benchmark.py`

```python
import argparse
import time

import torch
import torch.nn.functional as F
from aqlm.inference_kernels.cuda_kernel import CUDA_KERNEL
from aqlm.utils import _dequantize_weight, pack_int_data, unpack_int_data
from torch import nn


def benchmark(f, warmup=10, iter=10):
    for i in range(warmup + iter):
        f()
        if i == warmup - 1:
            torch.cuda.synchronize()
            tick = time.perf_counter()
    torch.cuda.synchronize()
    average_latency = (time.perf_counter() - tick) / iter
    time.sleep(1.0)
    return average_latency


MODELS = {
    "Llama 2 7B": [
        (4096, 11008),  #  gate_proj shape
    ],
    "Llama 2 13B": [
        (5120, 13824),  #  gate_proj shape
    ],
    "Llama 2 70B": [
        (8192, 28672),  #  gate_proj shape
    ],
}


if __name__ == "__main__":
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=10,
        help="Number of warmup iterations.",
    )
    parser.add_argument(
        "--benchmark_iters",
        type=int,
        default=10,
        help="Number of benchmark iterations.",
    )
    parser.add_argument(
        "--log_error",
        action="store_true",
    )
    parser.add_argument(
        "--nbits_per_codebook",
        type=int,
        default=16,
        help="Number of bits per codebook.",
    )
    parser.add_argument(
        "--num_codebooks",
        type=int,
        default=1,
        help="Number of num_codebooks.",
    )
    parser.add_argument(
        "--in_group_size",
        type=int,
        default=8,
        help="Input group size.",
    )

    args = parser.parse_args()

    for model, layers in MODELS.items():
        dense = 0
        quant = 0
        for in_features, out_features in layers:
            input = torch.randn((1, 1, in_features), dtype=torch.half, device=device)  #  [..., in_features]
            codes = pack_int_data(
                torch.randint(
                    2**args.nbits_per_codebook,
                    (out_features, in_features // args.in_group_size, args.num_codebooks),
                    device=device,
                ),  #  [num_out_groups, num_in_groups, num_codebooks]
                args.nbits_per_codebook,
            )
            codebooks = torch.randn(
                (args.num_codebooks, 2**args.nbits_per_codebook, 1, args.in_group_size),
                dtype=torch.half,
                device=device,
            )  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
            scales = torch.ones((out_features, 1, 1, 1), dtype=torch.half, device=device)  #  [num_out_groups, 1, 1, 1]

            weight = _dequantize_weight(unpack_int_data(codes, args.nbits_per_codebook), codebooks, scales).contiguous()

            output_ref = F.linear(input, weight)

            matmul = CUDA_KERNEL.code1x16_matmat if args.nbits_per_codebook == 16 else CUDA_KERNEL.code2x8_matmat

            output = matmul(input, codes, codebooks, scales, None)
            if args.log_error:
                print(
                    f"Relative error: {(torch.mean(torch.abs(output_ref - output)) / torch.mean(torch.abs(output_ref))).item():.2e}"
                )

            dense += benchmark(lambda: F.linear(input, weight, out=output_ref), args.warmup_iters, args.benchmark_iters)
            quant += benchmark(lambda: matmul(input, codes, codebooks, scales, None), args.warmup_iters, args.benchmark_iters)

        print(f"{model}: Dense forward = {dense * 1e6:.0f} mus")
        print(f"{model}: Quant forward = {quant * 1e6:.0f} mus")
        print(f"{model}: Speedup relative to dense = {(dense / quant):.3f}")
```

### `benchmark/matmul_benchmark_cpu.py`

```python
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import time

import numba
import numpy as np
import torch
import torch.nn.functional as F
from aqlm.utils import _dequantize_weight, pack_int_data, unpack_int_data
from torch import nn


def benchmark(f, warmup=10, iter=10):
    for i in range(warmup + iter):
        f()
        if i == warmup - 1:
            tick = time.perf_counter()
    average_latency = (time.perf_counter() - tick) / iter
    time.sleep(1.0)
    return average_latency


MODELS = {
    "Llama 2 7B": [
        (4096, 11008),  #  gate_proj shape
    ],
    "Llama 2 13B": [
        (5120, 13824),  #  gate_proj shape
    ],
    "Llama 2 70B": [
        (8192, 28672),  #  gate_proj shape
    ],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=10,
        help="Number of warmup iterations.",
    )
    parser.add_argument(
        "--benchmark_iters",
        type=int,
        default=1000,
        help="Number of benchmark iterations.",
    )
    parser.add_argument(
        "--log_error",
        action="store_true",
    )
    parser.add_argument(
        "--nbits_per_codebook",
        type=int,
        default=8,
        help="Number of bits per codebook.",
    )
    parser.add_argument(
        "--num_codebooks",
        type=int,
        default=2,
        help="Number of num_codebooks.",
    )
    parser.add_argument(
        "--in_group_size",
        type=int,
        default=8,
        help="Input group size.",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=1,
        help="Num threads.",
    )

    args = parser.parse_args()

    numba.set_num_threads(args.nthreads)
    torch.set_num_threads(args.nthreads)

    for model, layers in MODELS.items():
        dense = 0
        quant = 0
        for in_features, out_features in layers:
            in_group_size, num_codebooks, nbits_per_codebook, num_input_groups = (
                args.in_group_size,
                args.num_codebooks,
                args.nbits_per_codebook,
                in_features // args.in_group_size,
            )

            @numba.njit(parallel=True)
            def aqlm_gemv_lut(x, codebooks, codes_alt, scales):
                lut = x.reshape(-1, in_group_size) @ codebooks.reshape(-1, in_group_size).T
                lut = lut.reshape(-1, num_codebooks, 2**nbits_per_codebook)

                output_vec = np.zeros(out_features, dtype=x.dtype)
                for j in numba.prange(num_input_groups):
                    for i in range(out_features):
                        for c in range(num_codebooks):
                            output_vec[i] += lut[j, c, codes_alt[j, i, c]]
                output_vec *= scales.flatten()
                return output_vec

            input = torch.randn((1, in_features), dtype=torch.float32)  #  [..., in_features]
            codes = pack_int_data(
                torch.randint(
                    2**args.nbits_per_codebook, (in_features // args.in_group_size, out_features, args.num_codebooks)
                ),  #  [num_in_groups, num_out_groups, num_codebooks]
                args.nbits_per_codebook,
            )
            codebooks = torch.randn(
                (args.num_codebooks, 2**args.nbits_per_codebook, 1, args.in_group_size), dtype=torch.float32
            )  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
            scales = torch.randn((out_features, 1, 1, 1), dtype=torch.float32)  #  [num_out_groups, 1, 1, 1]

            weight = _dequantize_weight(
                unpack_int_data(torch.permute(codes, (1, 0, 2)), args.nbits_per_codebook), codebooks, scales
            ).contiguous()

            output_ref = F.linear(input, weight)
            output = aqlm_gemv_lut(input.numpy(), codebooks.numpy(), codes.numpy(), scales.numpy())
            if args.log_error:
                print(
                    f"Relative error: {(torch.mean(torch.abs(output_ref - output)) / torch.mean(torch.abs(output_ref))).item():.2e}"
                )

            dense += benchmark(lambda: F.linear(input, weight, out=output_ref), args.warmup_iters, args.benchmark_iters)
            input, codebooks, codes, scales = (
                input.numpy(),
                codebooks.squeeze(-2).numpy(),
                codes.view(torch.uint8).numpy(),
                scales.numpy(),
            )
            quant += benchmark(
                lambda: aqlm_gemv_lut(input, codebooks, codes, scales), args.warmup_iters, args.benchmark_iters
            )

        print(f"{model}: Dense forward = {dense * 1e3:.2f} ms")
        print(f"{model}: Quant forward = {quant * 1e3:.2f} ms")
        print(f"{model}: Speedup relative to dense = {(dense / quant):.3f}")
```

### `inference_lib/MANIFEST.in`

```text
include src/aqlm/inference_kernels/*.cpp
include src/aqlm/inference_kernels/*.cu
```

### `inference_lib/setup.cfg`

```ini
[metadata]
name = aqlm
version = 1.1.6
author = AQLM paper authors
author_email = vahe527887@yandex.ru
description = Efficiently run models quantized with AQLM
long_description = file: README.md
long_description_content_type = text/markdown
url = https://github.com/Vahe1994/AQLM
project_urls =
    Bug Tracker = https://github.com/Vahe1994/AQLM/issues
classifiers =
    Development Status :: 4 - Beta
    Intended Audience :: Developers
    Intended Audience :: Science/Research
    License :: OSI Approved :: MIT License
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3.8
    Programming Language :: Python :: 3.9
    Programming Language :: Python :: 3.10
    Programming Language :: Python :: 3.11
    Topic :: Scientific/Engineering
    Topic :: Scientific/Engineering :: Mathematics
    Topic :: Scientific/Engineering :: Artificial Intelligence
    Topic :: Software Development
    Topic :: Software Development :: Libraries
    Topic :: Software Development :: Libraries :: Python Modules

[options]
package_dir =
    = src
packages = find:
include_package_data = True
python_requires = >=3.8
install_requires =
    torch>=2.2.0
    transformers>=4.38.0
    accelerate>=0.27.0
[options.extras_require]
gpu =
    triton>=2.1
    ninja
cpu = 
    numba>=0.56.4
    scipy>=1.11.3
dev =
    pytest==6.2.5
    pytest-forked
    pytest-asyncio==0.16.0
    black==22.3.0
    isort==5.13.2
[options.packages.find]
where = src
```

### `inference_lib/src/aqlm/__init__.py`

```python
import aqlm.inference_kernels
from aqlm.inference import QuantizedLinear
from aqlm.inference_kernels import optimize_for_training
```

### `inference_lib/src/aqlm/inference.py`

```python
""" Core mathematics for Additive Quantization (AQ): initialization, reconstruction and beam search"""
import math
from typing import Any, Optional

import torch
import torch.nn as nn
from aqlm.inference_kernels import get_backward_pass_kernel, get_forward_pass_kernel
from aqlm.utils import get_int_dtype


class QuantizedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        in_group_size: int,
        out_group_size: int,
        num_codebooks: int,
        nbits_per_codebook: int,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        assert self.in_features % in_group_size == 0
        assert self.out_features % out_group_size == 0
        num_out_groups = out_features // out_group_size
        num_in_groups = in_features // in_group_size
        self.out_group_size, self.in_group_size = out_group_size, in_group_size
        self.num_codebooks = num_codebooks
        self.nbits_per_codebook = nbits_per_codebook
        self.codebook_size = 2**nbits_per_codebook

        # CODES & CODEBOOKS
        self.codebooks = nn.Parameter(
            torch.empty((num_codebooks, self.codebook_size, out_group_size, in_group_size), **factory_kwargs),
            requires_grad=False,
        )  # [num_codebooks, codebook_size, out_group_size, in_group_size]
        self.codes = nn.Parameter(
            torch.empty(
                (num_out_groups, num_in_groups, num_codebooks),
                device=device,
                dtype=get_int_dtype(nbits_per_codebook),
            ),
            requires_grad=False,
        )  #  [num_out_groups, num_in_groups, num_codebooks]

        # SCALES
        self.scales = nn.Parameter(
            torch.empty((num_out_groups, 1, 1, 1), **factory_kwargs), requires_grad=False
        )  #  [num_out_groups, 1, 1, 1]

        # BIAS
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # MATMUL_OPS
        self.gemv_op = None
        self.gemm_op = None
        self.use_gemv_rule = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.gemv_op is None:
            self.prepare_matmul_op(input)

        if self.use_gemv_rule(input):
            return self.gemv_op.apply(input, self.codes, self.codebooks, self.scales, self.bias)
        else:
            return self.gemm_op.apply(input, self.codes, self.codebooks, self.scales, self.bias)

    def prepare_matmul_op(self, input: torch.Tensor):
        if (
            not input.is_cuda
            and self.codebook_size == 256
            and self.codes.shape[0] == self.out_features // self.out_group_size
        ):
            self.codes.data = torch.permute(self.codes.data, (1, 0, 2)).contiguous()  #  TODO: fix this thing

        self.gemv_op = _get_autograd_matmul_op(
            get_forward_pass_kernel(self.codebooks, False),
            get_backward_pass_kernel(self.codebooks, False),
        )

        self.gemm_op = _get_autograd_matmul_op(
            get_forward_pass_kernel(self.codebooks, True),
            get_backward_pass_kernel(self.codebooks, True),
        )

        self.use_gemv_rule = lambda input: math.prod(input.shape[:-1]) <= 6


def _get_autograd_matmul_op(forward_pass_kernel, backward_pass_kernel):
    class _QuantizedMatmul(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: Any,
            input: torch.Tensor,
            codes: torch.IntTensor,
            codebooks: torch.Tensor,
            scales: torch.Tensor,
            bias: Optional[torch.Tensor],
        ) -> torch.Tensor:
            ctx.save_for_backward(
                input,
                codes,
                codebooks,
                scales,
                bias,
            )
            return forward_pass_kernel(
                input,
                codes,
                codebooks,
                scales,
                bias,
            )

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
            input, codes, codebooks, scales, bias = ctx.saved_tensors
            return (
                backward_pass_kernel(
                    grad_output,
                    codes,
                    codebooks,
                    scales,
                    bias,
                ),
                None,
                None,
                None,
                None,
            )

    return _QuantizedMatmul
```

### `inference_lib/src/aqlm/utils.py`

```python
from __future__ import annotations

import functools
import os
from typing import Callable, Iterator, Optional, Sequence

import torch
import torch.nn.functional as F


def get_int_dtype(nbits: int) -> torch.dtype:
    if nbits <= 8:
        return torch.int8
    if nbits <= 16:
        return torch.int16
    if nbits <= 32:
        return torch.int32
    if nbits <= 64:
        return torch.int64
    raise ValueError(f"No dtype available for {nbits}-bit codebooks")


@torch.inference_mode()
def pack_int_data(data: torch.IntTensor, nbits: int) -> torch.IntTensor:
    data[data >= 2 ** (nbits - 1)] -= 2**nbits
    return data.to(get_int_dtype(nbits))


@torch.inference_mode()
def unpack_int_data(data: torch.IntTensor, nbits: int) -> torch.IntTensor:
    return data.to(torch.int64) % (2**nbits)


@functools.lru_cache()
def maybe_script(fn: callable) -> callable:
    """Apply torch.jit.script to function unless one is using TPU. TPU does not support torch.jit.script."""
    using_tpu = bool(os.environ.get("TPU_NAME"))
    # this is a reserved variable that must be set to TPU address (e.g. grpc://11.22.33.44:1337) for TPU to function
    should_script = int(os.environ.get("AQ_USE_JIT", not using_tpu))
    return torch.jit.script(fn) if should_script else fn


@maybe_script
def _dequantize_weight(
    codes: torch.Tensor, codebooks: torch.Tensor, scales: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Decode float weights from quantization codes. Differentiable.
    :param codes: tensor of integer quantization codes, shape [*dims, num_out_groups, num_in_groups, num_codebooks]
    :param codebooks: tensor of vectors for each quantization code, [num_codebooks, codebook_size, out_group_size, in_group_size]
    :param scales: weight will be multiplied by this factor, must be broadcastble with [*dims, out_groups, num_in_groups, out_group_size, in_group_size]
    :return: reconstructed weight tensor of shape [*dims, num_in_groups*group_size]
    """
    num_out_groups, num_in_groups, num_codebooks = codes.shape[-3:]
    num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape
    out_features = num_out_groups * out_group_size
    in_features = num_in_groups * in_group_size
    codebook_offsets = torch.arange(
        0, num_codebooks * codebook_size, codebook_size, device=codes.device
    )  # shape: [num_codebooks]
    reconstructed_weight_flat = F.embedding_bag(
        codes.flatten(0, -2) + codebook_offsets, codebooks.flatten(0, 1).flatten(-2, -1), mode="sum"
    )  # [prod(dims) * num_out_groups * num_in_groups, out_group_size * in_group_size]

    reconstructed_weight_groupwise = reconstructed_weight_flat.view(
        list(codes.shape[:-3]) + [num_out_groups, num_in_groups, out_group_size, in_group_size]
    )
    if scales is not None:
        reconstructed_weight_groupwise = reconstructed_weight_groupwise.mul(scales)
    return reconstructed_weight_groupwise.swapaxes(-3, -2).reshape(list(codes.shape[:-3]) + [out_features, in_features])
```

### `inference_lib/src/aqlm/inference_kernels/__init__.py`

```python
from .kernel_selector import get_backward_pass_kernel, get_forward_pass_kernel, optimize_for_training
```

### `inference_lib/src/aqlm/inference_kernels/cuda_kernel.cpp`

```cpp
#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

namespace F = torch::nn::functional;


inline bool check_use_bfloat16(const torch::Tensor& input) {
  auto dtype = input.dtype();
  if (dtype == at::kHalf) {
    return false;
  } else if (dtype == at::kBFloat16) {
    return true;
  } else {
    throw c10::NotImplementedError(
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
      c10::str(
        "AQLM CUDA kernels only support float16 and bfloat16. Got ",
        dtype.name(),
        ". Please specify the correct `torch_dtype` when loading the model."
      )
    );
  }
}


template <bool use_bfloat16, size_t group_size>
void  code1x16_matvec_cuda(
  const void* A,
  const void* B,
        void* C,
  const void* codebook,
  int prob_m,
  int prob_k
);
extern template void code1x16_matvec_cuda<false, 8>(const void*, const void*, void*, const void*, int, int);
extern template void code1x16_matvec_cuda<true, 8>(const void*, const void*, void*, const void*, int, int); 
extern template void code1x16_matvec_cuda<false, 16>(const void*, const void*, void*, const void*, int, int);
extern template void code1x16_matvec_cuda<true, 16>(const void*, const void*, void*, const void*, int, int);

template <size_t group_size>
void code1x16_dequant_cuda(
  const void* A,
        void* C,
  const void* codebook,
  int prob_m,
  int prob_k
);
extern template void code1x16_dequant_cuda<8>(const void*, void*, const void*, int, int);
extern template void code1x16_dequant_cuda<16>(const void*, void*, const void*, int, int);

template <bool use_bfloat16>
void code2x8_matvec_cuda(
  const void* A,
  const void* B,
        void* C,
  const void* codebook,
  int prob_m,
  int prob_k
);
extern template void code2x8_matvec_cuda<false>(const void*, const void*, void*, const void*, int, int);
extern template void code2x8_matvec_cuda<true>(const void*, const void*, void*, const void*, int, int);

void code2x8_dequant_cuda(
  const void* A,
        void* C,
  const void* codebook,
  int prob_m,
  int prob_k,
  bool use_bfloat16
);

template <bool use_bfloat16>
void code1x8_matvec_cuda(
  const void* A,
  const void* B,
        void* C,
  const void* codebook,
  int prob_m,
  int prob_k
);
extern template void code1x8_matvec_cuda<false>(const void*, const void*, void*, const void*, int, int);
extern template void code1x8_matvec_cuda<true>(const void*, const void*, void*, const void*, int, int);

void code1x8_dequant_cuda(
  const void* A,
        void* C,
  const void* codebook,
  int prob_m,
  int prob_k,
  bool use_bfloat16
);

inline torch::Tensor scale_bias_unflatten_output(
        torch::Tensor& flat_output,
  const torch::Tensor& scales,
  const std::optional<torch::Tensor>& bias,
  const c10::IntArrayRef& input_sizes
) {
  flat_output *= scales.flatten().unsqueeze(0);
  if (bias.has_value()) {
    flat_output += bias->unsqueeze(0);
  }

  auto output_sizes = input_sizes.vec();
  output_sizes.pop_back();
  output_sizes.push_back(flat_output.size(-1));
  auto output = flat_output.reshape(output_sizes).clone();
  return output;
}

void code1x16_matvec(
  const torch::Tensor& A,
  const torch::Tensor& B,
        torch::Tensor& C,
  const torch::Tensor& codebook,
  const bool use_bfloat16
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  int prob_m = C.size(0);
  int prob_k = B.size(0);

  if (codebook.size(3) == 8) {
    if (use_bfloat16) {
      code1x16_matvec_cuda<true, 8>(A.data_ptr(), B.data_ptr(), C.data_ptr(), codebook.data_ptr(), prob_m, prob_k);
    } else {
      code1x16_matvec_cuda<false, 8>(A.data_ptr(), B.data_ptr(), C.data_ptr(), codebook.data_ptr(), prob_m, prob_k);
    }
  } else if (codebook.size(3) == 16) {
    if (use_bfloat16) {
      code1x16_matvec_cuda<true, 16>(A.data_ptr(), B.data_ptr(), C.data_ptr(), codebook.data_ptr(), prob_m, prob_k);
    } else {
      code1x16_matvec_cuda<false, 16>(A.data_ptr(), B.data_ptr(), C.data_ptr(), codebook.data_ptr(), prob_m, prob_k);
    }
  } else {
    throw c10::NotImplementedError(
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
      c10::str(
        "AQLM CUDA kernels only support codebooks with 8 or 16 features. Got ",
        codebook.size(3),
        "."
      )
    );
  }
}

torch::Tensor code1x16_matmat(
  const torch::Tensor& input,
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales,
  const std::optional<torch::Tensor>& bias
) {
  bool use_bfloat16 = check_use_bfloat16(input);
  auto input_sizes = input.sizes();
  auto out_features = codes.size(0) * codebooks.size(2);
  auto flat_input = input.reshape({-1, input.size(-1)});
  auto flat_output = torch::empty({flat_input.size(0), out_features},
    torch::TensorOptions()
      .dtype(input.dtype())
      .device(input.device())
  );

  for (int i = 0; i < flat_input.size(0); ++i) {
    auto input_vec = flat_input.index({i});
    auto output_vec = flat_output.index({i});
    code1x16_matvec(
      codes.squeeze(2),
      input_vec,
      output_vec,
      codebooks,
      use_bfloat16
    );
  }
  return scale_bias_unflatten_output(
    flat_output,
    scales,
    bias,
    input_sizes
  );
}

torch::Tensor code1x16_dequant(
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales
) {
  check_use_bfloat16(codebooks);
  auto in_features = codes.size(1) * codebooks.size(3);
  auto out_features = scales.size(0);

  auto weight = torch::empty({out_features, in_features},
    torch::TensorOptions()
      .dtype(codebooks.dtype())
      .device(codebooks.device())
  );
  if (codebooks.size(3) == 8) {
    code1x16_dequant_cuda<8>(
      codes.data_ptr(),
      weight.data_ptr(),
      codebooks.data_ptr(),
      out_features,
      in_features
    );
  } else if (codebooks.size(3) == 16) {
    code1x16_dequant_cuda<16>(
      codes.data_ptr(),
      weight.data_ptr(),
      codebooks.data_ptr(),
      out_features,
      in_features
    );
  } else {
    throw c10::NotImplementedError(
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
      c10::str(
        "AQLM CUDA kernels only support codebooks with 8 or 16 features. Got ",
        codebooks.size(3),
        "."
      )
    );
  }
  weight *= scales.index({"...", 0, 0});

  return weight;
}

int4 accumulate_sizes(const torch::Tensor& codebook_partition_sizes)
{
  int4 cumulative_sizes;
  auto cumulative_size = &cumulative_sizes.x;
  int i = 0;
  int last = 0;
  assert(codebook_partition_sizes.size(0) <= 4);
  for (; i <  codebook_partition_sizes.size(0); ++i, ++cumulative_size)
  {
    *cumulative_size = codebook_partition_sizes[i].item<int>() + last;
    last = *cumulative_size;
  }
  // fill in the rest with unreachable.
  for (; i < 4; ++i, ++cumulative_size)
  {
    *cumulative_size = last*10;
  }
  return cumulative_sizes;
}

torch::Tensor code1x16_matmat_dequant(
  const torch::Tensor& input,
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales,
  const std::optional<torch::Tensor>& bias
) {
  bool use_bfloat16 = check_use_bfloat16(input);
  auto input_sizes = input.sizes();
  auto in_features = codes.size(1) * codebooks.size(3);
  auto out_features = codes.size(0) * codebooks.size(2);
  auto flat_input = input.reshape({-1, input.size(-1)});

  auto weight = torch::empty({out_features, in_features},
    torch::TensorOptions()
      .dtype(codebooks.dtype())
      .device(codebooks.device())
  );
  if (codebooks.size(3) == 8) {
    code1x16_dequant_cuda<8>(
      codes.data_ptr(),
      weight.data_ptr(),
      codebooks.data_ptr(),
      out_features,
      in_features
    );
  } else if (codebooks.size(3) == 16) {
    code1x16_dequant_cuda<16>(
      codes.data_ptr(),
      weight.data_ptr(),
      codebooks.data_ptr(),
      out_features,
      in_features
    );
  } else {
    throw c10::NotImplementedError(
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
      c10::str(
        "AQLM CUDA kernels only support codebooks with 8 or 16 features. Got ",
        codebooks.size(3),
        "."
      )
    );
  }

  auto flat_output = F::linear(flat_input, weight);
  return scale_bias_unflatten_output(
    flat_output,
    scales,
    bias,
    input_sizes
  );
}

torch::Tensor code1x16_matmat_dequant_transposed(
  const torch::Tensor& input,
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales,
  const std::optional<torch::Tensor>& bias
) {
  check_use_bfloat16(codebooks);
  auto input_sizes = input.sizes();
  auto in_features = codes.size(1) * 8;
  auto out_features = scales.size(0);
  auto scaled_input = (input.reshape({-1, input.size(-1)}) * scales.flatten().unsqueeze(0)).reshape(input_sizes);

  auto weight = torch::empty({out_features, in_features},
    torch::TensorOptions()
      .dtype(codebooks.dtype())
      .device(codebooks.device())
  );
  if (codebooks.size(3) == 8) {
    code1x16_dequant_cuda<8>(
      codes.data_ptr(),
      weight.data_ptr(),
      codebooks.data_ptr(),
      out_features,
      in_features
    );
  } else if (codebooks.size(3) == 16) {
    code1x16_dequant_cuda<16>(
      codes.data_ptr(),
      weight.data_ptr(),
      codebooks.data_ptr(),
      out_features,
      in_features
    );
  } else {
    throw c10::NotImplementedError(
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
      c10::str(
        "AQLM CUDA kernels only support codebooks with 8 or 16 features. Got ",
        codebooks.size(3),
        "."
      )
    );
  }

  torch::Tensor bias_2{};
  if (bias.has_value()) {
    bias_2 = bias.value();
  }

  return F::linear(scaled_input, weight.transpose(0, 1), bias_2);
}

void code2x8_matvec(
  const torch::Tensor& A,
  const torch::Tensor& B,
        torch::Tensor& C,
  const torch::Tensor& codebook,
  bool use_bfloat16
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  int prob_m = C.size(0);
  int prob_k = B.size(0);
  if (use_bfloat16) {
    code2x8_matvec_cuda<true>(
      A.data_ptr(),
      B.data_ptr(),
      C.data_ptr(),
      codebook.data_ptr(),
      prob_m,
      prob_k
    );
  } else {
    code2x8_matvec_cuda<false>(
      A.data_ptr(),
      B.data_ptr(),
      C.data_ptr(),
      codebook.data_ptr(),
      prob_m,
      prob_k
    );
  }
}

torch::Tensor code2x8_matmat(
  const torch::Tensor& input,
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales,
  const std::optional<torch::Tensor>& bias
) {
  bool use_bfloat16 = check_use_bfloat16(input);
  auto input_sizes = input.sizes();
  auto out_features = codes.size(0) * codebooks.size(2);
  auto flat_input = input.reshape({-1, input.size(-1)});
  auto flat_output = torch::empty({flat_input.size(0), out_features},
    torch::TensorOptions()
      .dtype(input.dtype())
      .device(input.device())
  );

  for (int i = 0; i < flat_input.size(0); ++i) {
    auto input_vec = flat_input.index({i});
    auto output_vec = flat_output.index({i});
    code2x8_matvec(
      codes.squeeze(2),
      input_vec,
      output_vec,
      codebooks,
      use_bfloat16
    );
  }
  return scale_bias_unflatten_output(
    flat_output,
    scales,
    bias,
    input_sizes
  );
}

torch::Tensor code2x8_dequant(
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales
) {
  auto use_bfloat16 = check_use_bfloat16(codebooks);
  auto in_features = codes.size(1) * 8;
  auto out_features = scales.size(0);

  auto weight = torch::empty({out_features, in_features},
    torch::TensorOptions()
      .dtype(codebooks.dtype())
      .device(codebooks.device())
  );
  code2x8_dequant_cuda(
    codes.data_ptr(),
    weight.data_ptr(),
    codebooks.data_ptr(),
    out_features,
    in_features,
    use_bfloat16
  );
  weight *= scales.index({"...", 0, 0});

  return weight;
}

torch::Tensor code2x8_matmat_dequant(
  const torch::Tensor& input,
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales,
  const std::optional<torch::Tensor>& bias
) {
  bool use_bfloat16 = check_use_bfloat16(input);
  auto input_sizes = input.sizes();
  auto in_features = codes.size(1) * 8;
  auto out_features = codes.size(0) * codebooks.size(2);
  auto flat_input = input.reshape({-1, input.size(-1)});

  auto weight = torch::empty({out_features, in_features},
    torch::TensorOptions()
      .dtype(codebooks.dtype())
      .device(codebooks.device())
  );
  code2x8_dequant_cuda(
    codes.data_ptr(),
    weight.data_ptr(),
    codebooks.data_ptr(),
    out_features,
    in_features,
    use_bfloat16
  );

  auto flat_output = F::linear(flat_input, weight);
  return scale_bias_unflatten_output(
    flat_output,
    scales,
    bias,
    input_sizes
  );
}

torch::Tensor code2x8_matmat_dequant_transposed(
  const torch::Tensor& input,
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales,
  const std::optional<torch::Tensor>& bias
) {
  auto use_bfloat16 = check_use_bfloat16(codebooks);
  auto input_sizes = input.sizes();
  auto in_features = codes.size(1) * 8;
  auto out_features = scales.size(0);
  auto scaled_input = (input.reshape({-1, input.size(-1)}) * scales.flatten().unsqueeze(0)).reshape(input_sizes);

  auto weight = torch::empty({out_features, in_features},
    torch::TensorOptions()
      .dtype(codebooks.dtype())
      .device(codebooks.device())
  );
  code2x8_dequant_cuda(
    codes.data_ptr(),
    weight.data_ptr(),
    codebooks.data_ptr(),
    out_features,
    in_features,
    use_bfloat16
  );

  torch::Tensor bias_2{};
  if (bias.has_value()) {
    bias_2 = bias.value();
  }

  return F::linear(input, weight.transpose(0, 1), bias_2);
}

void code1x8_matvec(
  const torch::Tensor& A,
  const torch::Tensor& B,
        torch::Tensor& C,
  const torch::Tensor& codebook,
  bool use_bfloat16
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  int prob_m = C.size(0);
  int prob_k = B.size(0);
  if (use_bfloat16) {
    code1x8_matvec_cuda<true>(
      A.data_ptr(),
      B.data_ptr(),
      C.data_ptr(),
      codebook.data_ptr(),
      prob_m,
      prob_k
    );
  } else {
    code1x8_matvec_cuda<false>(
      A.data_ptr(),
      B.data_ptr(),
      C.data_ptr(),
      codebook.data_ptr(),
      prob_m,
      prob_k
    );
  }
}

torch::Tensor code1x8_matmat(
  const torch::Tensor& input,
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales,
  const std::optional<torch::Tensor>& bias
) {
  bool use_bfloat16 = check_use_bfloat16(input);
  auto input_sizes = input.sizes();
  auto out_features = codes.size(0) * codebooks.size(2);
  auto flat_input = input.reshape({-1, input.size(-1)});
  auto flat_output = torch::empty({flat_input.size(0), out_features},
    torch::TensorOptions()
      .dtype(input.dtype())
      .device(input.device())
  );

  for (int i = 0; i < flat_input.size(0); ++i) {
    auto input_vec = flat_input.index({i});
    auto output_vec = flat_output.index({i});
    code1x8_matvec(
      codes.squeeze(2),
      input_vec,
      output_vec,
      codebooks,
      use_bfloat16
    );
  }
  return scale_bias_unflatten_output(
    flat_output,
    scales,
    bias,
    input_sizes
  );
}

torch::Tensor code1x8_dequant(
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales
) {
  auto use_bfloat16 = check_use_bfloat16(codebooks);
  auto in_features = codes.size(1) * 8;
  auto out_features = scales.size(0);

  auto weight = torch::empty({out_features, in_features},
    torch::TensorOptions()
      .dtype(codebooks.dtype())
      .device(codebooks.device())
  );
  code1x8_dequant_cuda(
    codes.data_ptr(),
    weight.data_ptr(),
    codebooks.data_ptr(),
    out_features,
    in_features,
    use_bfloat16
  );
  weight *= scales.index({"...", 0, 0});

  return weight;
}

torch::Tensor code1x8_matmat_dequant(
  const torch::Tensor& input,
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales,
  const std::optional<torch::Tensor>& bias
) {
  bool use_bfloat16 = check_use_bfloat16(input);
  auto input_sizes = input.sizes();
  auto in_features = codes.size(1) * 8;
  auto out_features = codes.size(0) * codebooks.size(2);
  auto flat_input = input.reshape({-1, input.size(-1)});

  auto weight = torch::empty({out_features, in_features},
    torch::TensorOptions()
      .dtype(codebooks.dtype())
      .device(codebooks.device())
  );
  code1x8_dequant_cuda(
    codes.data_ptr(),
    weight.data_ptr(),
    codebooks.data_ptr(),
    out_features,
    in_features,
    use_bfloat16
  );

  auto flat_output = F::linear(flat_input, weight);
  return scale_bias_unflatten_output(
    flat_output,
    scales,
    bias,
    input_sizes
  );
}

torch::Tensor code1x8_matmat_dequant_transposed(
  const torch::Tensor& input,
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales,
  const std::optional<torch::Tensor>& bias
) {
  auto use_bfloat16 = check_use_bfloat16(codebooks);
  auto input_sizes = input.sizes();
  auto in_features = codes.size(1) * 8;
  auto out_features = scales.size(0);
  auto scaled_input = (input.reshape({-1, input.size(-1)}) * scales.flatten().unsqueeze(0)).reshape(input_sizes);

  auto weight = torch::empty({out_features, in_features},
    torch::TensorOptions()
      .dtype(codebooks.dtype())
      .device(codebooks.device())
  );
  code1x8_dequant_cuda(
    codes.data_ptr(),
    weight.data_ptr(),
    codebooks.data_ptr(),
    out_features,
    in_features,
    use_bfloat16
  );

  torch::Tensor bias_2{};
  if (bias.has_value()) {
    bias_2 = bias.value();
  }

  return F::linear(input, weight.transpose(0, 1), bias_2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("code1x16_matmat", &code1x16_matmat, "1x16 (2bit) codebook matrix-matrix product through matvec.");
  m.def("code1x16_dequant", &code1x16_dequant, "1x16 (2bit) codebook dequantization.");
  m.def("code1x16_matmat_dequant", &code1x16_matmat_dequant, "1x16 (2bit) codebook matrix-matrix dequantization product.");
  m.def("code1x16_matmat_dequant_transposed", &code1x16_matmat_dequant_transposed, "1x16 (2bit) codebook matrix-matrix dequantization product for backward pass.");
  m.def("code2x8_matmat", &code2x8_matmat, "2x8 (2bit) codebook matrix-matrix product.");
  m.def("code2x8_dequant", &code2x8_dequant, "2x8 (2bit) codebook dequantization.");
  m.def("code2x8_matmat_dequant", &code2x8_matmat_dequant, "2x8 (2bit) codebook matrix-matrix dequantization product.");
  m.def("code2x8_matmat_dequant_transposed", &code2x8_matmat_dequant_transposed, "2x8 (2bit) codebook matrix-matrix dequantization product for backward pass.");
  m.def("code1x8_matmat", &code1x8_matmat, "1x8 (1bit) codebook matrix-matrix product.");
  m.def("code1x8_dequant", &code1x8_dequant, "1x8 (1bit) codebook dequantization.");
  m.def("code1x8_matmat_dequant", &code1x8_matmat_dequant, "1x8 (1bit) codebook matrix-matrix dequantization product.");
  m.def("code1x8_matmat_dequant_transposed", &code1x8_matmat_dequant_transposed, "1x8 (1bit) codebook matrix-matrix dequantization product for backward pass.");
}
```

### `inference_lib/src/aqlm/inference_kernels/cuda_kernel.py`

```python
import os
from typing import Optional

import torch
from torch.utils.cpp_extension import load

CUDA_FOLDER = os.path.dirname(os.path.abspath(__file__))
CUDA_KERNEL = load(
    name="codebook_cuda",
    sources=[os.path.join(CUDA_FOLDER, "cuda_kernel.cpp"), os.path.join(CUDA_FOLDER, "cuda_kernel.cu")],
)

torch.library.define(
    "aqlm::code1x16_matmat", "(Tensor input, Tensor codes, Tensor codebooks, Tensor scales, Tensor bias) -> Tensor"
)

torch.library.impl("aqlm::code1x16_matmat", "default", CUDA_KERNEL.code1x16_matmat)


@torch.library.impl_abstract("aqlm::code1x16_matmat")
def code1x16_matmat_meta(input, codes, codebooks, scales, bias):
    return torch.empty(input.shape[:-1] + (codes.shape[0],), device=input.device, dtype=input.dtype)


torch.library.define(
    "aqlm::code1x16_matmat_dequant",
    "(Tensor input, Tensor codes, Tensor codebooks, Tensor scales, Tensor bias) -> Tensor",
)

torch.library.impl("aqlm::code1x16_matmat_dequant", "default", CUDA_KERNEL.code1x16_matmat_dequant)


@torch.library.impl_abstract("aqlm::code1x16_matmat_dequant")
def code1x16_matmat_dequant_meta(input, codes, codebooks, scales, bias):
    return torch.empty(input.shape[:-1] + (codes.shape[0],), device=input.device, dtype=input.dtype)


torch.library.define(
    "aqlm::code1x16_matmat_dequant_transposed",
    "(Tensor input, Tensor codes, Tensor codebooks, Tensor scales, Tensor bias) -> Tensor",
)

torch.library.impl(
    "aqlm::code1x16_matmat_dequant_transposed", "default", CUDA_KERNEL.code1x16_matmat_dequant_transposed
)


@torch.library.impl_abstract("aqlm::code1x16_matmat_dequant_transposed")
def code1x16_matmat_dequant_transposed_meta(input, codes, codebooks, scales, bias):
    return torch.empty(
        input.shape[:-1] + (codes.shape[1] * codebooks.shape[3],), device=input.device, dtype=input.dtype
    )


torch.library.define(
    "aqlm::code2x8_matmat", "(Tensor input, Tensor codes, Tensor codebooks, Tensor scales, Tensor bias) -> Tensor"
)

torch.library.impl("aqlm::code2x8_matmat", "default", CUDA_KERNEL.code2x8_matmat)


@torch.library.impl_abstract("aqlm::code2x8_matmat")
def code2x8_matmat_meta(input, codes, codebooks, scales, bias):
    return torch.empty(input.shape[:-1] + (codes.shape[0],), device=input.device, dtype=input.dtype)


torch.library.define(
    "aqlm::code2x8_matmat_dequant",
    "(Tensor input, Tensor codes, Tensor codebooks, Tensor scales, Tensor bias) -> Tensor",
)

torch.library.impl("aqlm::code2x8_matmat_dequant", "default", CUDA_KERNEL.code2x8_matmat_dequant)


@torch.library.impl_abstract("aqlm::code2x8_matmat_dequant")
def code2x8_matmat_dequant_meta(input, codes, codebooks, scales, bias):
    return torch.empty(input.shape[:-1] + (codes.shape[0],), device=input.device, dtype=input.dtype)


torch.library.define(
    "aqlm::code2x8_matmat_dequant_transposed",
    "(Tensor input, Tensor codes, Tensor codebooks, Tensor scales, Tensor bias) -> Tensor",
)

torch.library.impl("aqlm::code2x8_matmat_dequant_transposed", "default", CUDA_KERNEL.code2x8_matmat_dequant_transposed)


@torch.library.impl_abstract("aqlm::code2x8_matmat_dequant_transposed")
def code2x8_matmat_dequant_transposed_meta(input, codes, codebooks, scales, bias):
    return torch.empty(
        input.shape[:-1] + (codes.shape[1] * codebooks.shape[3],), device=input.device, dtype=input.dtype
    )


torch.library.define(
    "aqlm::code1x8_matmat", "(Tensor input, Tensor codes, Tensor codebooks, Tensor scales, Tensor bias) -> Tensor"
)

torch.library.impl("aqlm::code1x8_matmat", "default", CUDA_KERNEL.code1x8_matmat)


@torch.library.impl_abstract("aqlm::code1x8_matmat")
def code1x8_matmat_meta(input, codes, codebooks, scales, bias):
    return torch.empty(input.shape[:-1] + (codes.shape[0],), device=input.device, dtype=input.dtype)


torch.library.define(
    "aqlm::code1x8_matmat_dequant",
    "(Tensor input, Tensor codes, Tensor codebooks, Tensor scales, Tensor bias) -> Tensor",
)

torch.library.impl("aqlm::code1x8_matmat_dequant", "default", CUDA_KERNEL.code1x8_matmat_dequant)


@torch.library.impl_abstract("aqlm::code1x8_matmat_dequant")
def code1x8_matmat_dequant_meta(input, codes, codebooks, scales, bias):
    return torch.empty(input.shape[:-1] + (codes.shape[0],), device=input.device, dtype=input.dtype)


torch.library.define(
    "aqlm::code1x8_matmat_dequant_transposed",
    "(Tensor input, Tensor codes, Tensor codebooks, Tensor scales, Tensor bias) -> Tensor",
)

torch.library.impl("aqlm::code1x8_matmat_dequant_transposed", "default", CUDA_KERNEL.code1x8_matmat_dequant_transposed)


@torch.library.impl_abstract("aqlm::code1x8_matmat_dequant_transposed")
def code1x8_matmat_dequant_transposed_meta(input, codes, codebooks, scales, bias):
    return torch.empty(
        input.shape[:-1] + (codes.shape[1] * codebooks.shape[3],), device=input.device, dtype=input.dtype
    )
```

### `inference_lib/src/aqlm/inference_kernels/dequantization.py`

```python
from typing import Optional

import torch
import torch.nn.functional as F
from aqlm.utils import _dequantize_weight, unpack_int_data
from torch import nn


def dequantize_gemm(
    input: torch.Tensor,  #  [..., in_features]
    codes: torch.IntTensor,  #  [num_out_groups, num_in_groups, num_codebooks]
    codebooks: torch.Tensor,  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
    scales: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    dequantized_weight = _dequantize_weight(
        unpack_int_data(codes, codebooks.shape[1].bit_length() - 1),
        codebooks,
        scales,
    )
    return F.linear(input, dequantized_weight, bias)
```

### `inference_lib/src/aqlm/inference_kernels/kernel_selector.py`

```python
import warnings
from contextlib import contextmanager
from typing import Callable, Optional

import torch


@contextmanager
def optimize_for_training():
    """
    WARNING: `optimize_for_training` is deprecated. The optimization now happens automatically at runtime.
    OBSOLETE: Use this context manager during model initialization (e.g. `.from_pretrained(...)`) to select inference kernels optimized for larger batch sizes
    """
    warnings.warn("`optimize_for_training` is deprecated. The optimization now happens automatically at runtime.")
    try:
        yield
    finally:
        return


def get_forward_pass_kernel(
    codebooks: torch.Tensor,
    optimize_for_training: bool,
) -> Callable[[torch.Tensor, torch.IntTensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
    num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape

    if (optimize_for_training, codebooks.device.type, num_codebooks, codebook_size, out_group_size) == (
        False,
        "cuda",
        1,
        65536,
        1,
    ) and in_group_size in [8, 16]:
        from .cuda_kernel import CUDA_FOLDER

        return torch.ops.aqlm.code1x16_matmat
    elif (optimize_for_training, codebooks.device.type, num_codebooks, codebook_size, out_group_size,) == (
        True,
        "cuda",
        1,
        65536,
        1,
    ) and in_group_size in [8, 16]:
        from .cuda_kernel import CUDA_FOLDER

        return torch.ops.aqlm.code1x16_matmat_dequant
    elif (
        optimize_for_training,
        codebooks.device.type,
        num_codebooks,
        codebook_size,
        out_group_size,
        in_group_size,
    ) == (False, "cuda", 2, 256, 1, 8):
        from .cuda_kernel import CUDA_FOLDER

        return torch.ops.aqlm.code2x8_matmat
    elif (
        optimize_for_training,
        codebooks.device.type,
        num_codebooks,
        codebook_size,
        out_group_size,
        in_group_size,
    ) == (False, "cuda", 1, 256, 1, 8):
        from .cuda_kernel import CUDA_FOLDER

        return torch.ops.aqlm.code1x8_matmat
    elif (
        optimize_for_training,
        codebooks.device.type,
        num_codebooks,
        codebook_size,
        out_group_size,
        in_group_size,
    ) == (True, "cuda", 2, 256, 1, 8):
        from .cuda_kernel import CUDA_FOLDER

        return torch.ops.aqlm.code2x8_matmat_dequant
    elif (
        optimize_for_training,
        codebooks.device.type,
        num_codebooks,
        codebook_size,
        out_group_size,
        in_group_size,
    ) == (True, "cuda", 1, 256, 1, 8):
        from .cuda_kernel import CUDA_FOLDER

        return torch.ops.aqlm.code1x8_matmat_dequant
    elif (optimize_for_training, codebooks.device.type, out_group_size) == (False, "cuda", 1):
        from .triton_kernel import triton_matmul

        return triton_matmul
    elif (codebooks.device.type, codebook_size, out_group_size) == ("cpu", 256, 1):
        from .numba_kernel import numba_gemm_lut

        return numba_gemm_lut
    else:
        from .dequantization import dequantize_gemm

        return dequantize_gemm


def get_backward_pass_kernel(
    codebooks: torch.Tensor,
    optimize_for_training: bool,
) -> torch.Tensor:
    num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape

    if (optimize_for_training, codebooks.device.type, num_codebooks, codebook_size, out_group_size,) == (
        True,
        "cuda",
        1,
        65536,
        1,
    ) and in_group_size in [8, 16]:
        from .cuda_kernel import CUDA_FOLDER

        return torch.ops.aqlm.code1x16_matmat_dequant_transposed
    elif (
        optimize_for_training,
        codebooks.device.type,
        num_codebooks,
        codebook_size,
        out_group_size,
        in_group_size,
    ) == (True, "cuda", 2, 256, 1, 8):
        from .cuda_kernel import CUDA_FOLDER

        return torch.ops.aqlm.code2x8_matmat_dequant_transposed
    elif (
        optimize_for_training,
        codebooks.device.type,
        num_codebooks,
        codebook_size,
        out_group_size,
        in_group_size,
    ) == (True, "cuda", 1, 256, 1, 8):
        from .cuda_kernel import CUDA_FOLDER

        return torch.ops.aqlm.code1x8_matmat_dequant_transposed
    else:
        forward_pass_kernel = get_forward_pass_kernel(
            codebooks=codebooks.transpose(2, 3), optimize_for_training=optimize_for_training
        )

        def _backward_pass_kernel(
            grad_output: torch.Tensor,  #  [..., in_features]
            codes: torch.IntTensor,  #  [num_out_groups, num_in_groups, num_codebooks]
            codebooks: torch.Tensor,  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
            scales: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
            bias: Optional[torch.Tensor],
        ) -> torch.Tensor:
            return forward_pass_kernel(
                grad_output.contiguous(),
                codes.transpose(0, 1).contiguous(),
                codebooks.transpose(2, 3).contiguous(),
                scales.transpose(0, 1).transpose(2, 3).contiguous(),
                None,
            )

        return _backward_pass_kernel
```

### `inference_lib/src/aqlm/inference_kernels/numba_kernel.py`

```python
from typing import Optional

import numba
import numpy as np
import torch

COMPILED_KERNELS = {}


def numba_gemm_lut(
    input: torch.Tensor,  #  [..., in_features]
    codes: torch.IntTensor,  #  [num_out_groups, num_in_groups, num_codebooks]
    codebooks: torch.Tensor,  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
    scales: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    input_shape = input.shape
    input = input.reshape(-1, input_shape[-1])

    device, dtype = codebooks.device, codebooks.dtype
    num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape
    in_features = input.shape[1]
    num_input_groups = in_features // in_group_size
    out_features = codes.shape[1] * out_group_size
    assert input.ndim == 2
    assert scales.shape == (out_features // out_group_size, 1, 1, 1)
    assert in_features % in_group_size == 0
    assert codebook_size == 2**8
    assert codes.dtype == torch.int8
    assert (
        input.dtype == torch.float32 and codebooks.dtype == torch.float32
    ), f"please load the model with `torch_dtype=torch.float32`, as {input.dtype} is not supported for CPU"

    kernel_key = (in_group_size, out_features, in_features, num_codebooks)
    if kernel_key not in COMPILED_KERNELS:

        @numba.njit(parallel=True)
        def numba_gemv_lut_(x, codebooks, codes_alt, scales):
            lut = x.reshape(-1, in_group_size) @ codebooks.reshape(-1, in_group_size).T
            lut = lut.reshape(-1, num_codebooks, codebook_size)

            output_vec = np.zeros(out_features, dtype=x.dtype)
            for j in numba.prange(num_input_groups):
                for i in range(out_features):
                    for c in range(num_codebooks):
                        output_vec[i] += lut[j, c, codes_alt[j, i, c]]
            output_vec *= scales.flatten()
            return output_vec

        COMPILED_KERNELS[kernel_key] = numba_gemv_lut_
    compiled_kernel = COMPILED_KERNELS[kernel_key]

    output = torch.empty(input.shape[0], out_features, device=device, dtype=dtype)
    for i in range(input.shape[0]):
        output[i] = torch.as_tensor(
            compiled_kernel(
                input[i].numpy(),
                codebooks.numpy(),
                codes.view(torch.uint8).numpy(),
                scales.numpy(),
            )
        )
    if bias is not None:
        output += bias
    return output.reshape(input_shape[:-1] + (-1,))
```

### `inference_lib/src/aqlm/inference_kernels/triton_kernel.py`

```python
import math
from typing import Optional

import torch
import triton
import triton.language as tl
from torch.autograd import Function


@triton.autotune(
    configs=[
        triton.Config({"UNUSED": 1}, num_stages=num_stages, num_warps=num_warps)
        for num_stages in (1, 2, 3, 4, 5)
        for num_warps in (1, 2, 4, 8)
    ],
    key=[
        "in_features",
        "out_features",
        "num_codebooks",
        "codebook_size",
        "out_group_size",
        "in_group_size",
        "num_input_groups",
        "num_input_groups_next_power_of_2",
        "compute_in_fp32",
        "has_output_scale",
        "has_bias",
    ],
)
@triton.jit
def _aqlm_gemv_simple(
    input_vec_ptr,
    output_vec_ptr,
    codes_ptr,
    codebooks_ptr,
    scales_ptr,
    bias_ptr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    num_codebooks: tl.constexpr,
    codebook_size: tl.constexpr,
    out_group_size: tl.constexpr,
    in_group_size: tl.constexpr,
    num_input_groups: tl.constexpr,
    num_input_groups_next_power_of_2: tl.constexpr,
    compute_in_fp32: tl.constexpr,
    has_output_scale: tl.constexpr,
    has_bias: tl.constexpr,
    UNUSED: tl.constexpr,
):
    # variables ending with "_i" mean "for i-th output unit"
    pid = tl.program_id(axis=0)  # [0, 1, ... {num_out_groups-1}]

    # Stage 1: load input data
    input_vec = tl.load(
        input_vec_ptr
        + tl.arange(0, num_input_groups_next_power_of_2)[:, None, None, None] * in_group_size
        + tl.arange(0, in_group_size)[None, None, None, :],
        mask=tl.arange(0, num_input_groups_next_power_of_2)[:, None, None, None] < num_input_groups,
    )
    # [in_features//in_group_size, 1, 1, group_size]
    # Note: we could simply load input_vec then reshape
    #     input_vec = tl.load(input_vec_ptr + tl.arange(0, in_features))  # [in_features]
    #     input_vec = tl.view(input_vec, [num_input_groups, 1, in_group_size])
    #     , but this does not work because tl.view may reorder elements arbitrarily; see its docstring
    dtype = input_vec.dtype

    # Stage 2: load integer codes for the active row
    # [in_features // in_group_size, num_codebooks]
    codes_i_ptrs = (
        codes_ptr
        + pid * num_input_groups * num_codebooks
        + tl.arange(0, num_input_groups_next_power_of_2)[:, None] * num_codebooks
        + tl.arange(0, num_codebooks)[None, :]
    )
    codes_i_mask_1d = tl.arange(0, num_input_groups_next_power_of_2) < num_input_groups

    codes_i = tl.load(codes_i_ptrs, mask=codes_i_mask_1d[:, None])  # [in_features//in_group_size, num_codebooks]
    codes_i = codes_i.to(tl.int32)
    codes_i = (codes_i) + (codes_i < 0) * codebook_size  # aka 2 ** nbits_per_codebook
    # ^-- (because codes are int16 tensors that contain uint data)

    # The following alternative does not work:
    #     codes_i = codes_i.to(tl.int32) % codebook_size # aka 2 ** nbits_per_codeboo

    # shift codes_i so that codebooks after 0th point to correct indices in codebooks_ptr
    codes_i += tl.arange(0, num_codebooks)[None, :] * codebook_size  # aka 2 ** nbits_per_codebook
    # ^-- [in_group_size, num_codebooks]

    # Stage 3: convert codes to pointers to every individual (activated) weight in codebooks
    # [in_features // in_group_size, num_codebooks, out_group_size, in_group_size]
    out_group_ix = tl.arange(0, out_group_size)[None, None, :, None]
    in_group_ix = tl.arange(0, in_group_size)[None, None, None, :]
    weight_i_ptrs = (
        codebooks_ptr
        + codes_i[:, :, None, None] * out_group_size * in_group_size
        + out_group_ix * in_group_size
        + in_group_ix
    )

    # Stage 4: reconstruct weights, multiply by inputs and write out
    weights_i = tl.load(weight_i_ptrs, mask=codes_i_mask_1d[:, None, None, None], other=0)
    if compute_in_fp32:
        weights_i = weights_i.to(tl.float32)
        input_vec = input_vec.to(tl.float32)
    # ^-- [in_features // in_group_size, num_codebooks, out_group_size, in_group_size]

    output_i = weights_i * input_vec  #  [in_features // in_group_size, num_codebooks, out_group_size, in_group_size]

    if out_group_size == 1:
        output_i = tl.sum(output_i)  #  []
    else:
        output_i = tl.sum(output_i, axis=1)  #  [in_features // in_group_size, out_group_size, in_group_size]
        output_i = tl.sum(output_i, axis=2)  #  [in_features // in_group_size, out_group_size]
        output_i = tl.sum(output_i, axis=0)  #  [out_group_size]

    if has_output_scale:
        output_i *= tl.load(scales_ptr + pid).to(weights_i.dtype)  # scalar
    if has_bias:
        output_i += tl.load(bias_ptr + pid).to(weights_i.dtype)

    if out_group_size == 1:
        tl.store(output_vec_ptr + pid, output_i.to(dtype))
    else:
        tl.store(output_vec_ptr + pid * out_group_size + tl.arange(0, out_group_size), output_i.to(dtype))


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** math.ceil(math.log2(x))


def aqlm_gemm_stupid(
    input: torch.Tensor,
    codes_i16: torch.ShortTensor,
    codebooks: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor],
    compute_in_fp32: bool = True,
):
    device, dtype = codebooks.device, codebooks.dtype
    num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape
    in_features = input.shape[1]
    out_features = codes_i16.shape[0] * out_group_size
    num_input_groups = codes_i16.shape[1]
    assert input.ndim == 2
    assert in_features % in_group_size == 0
    assert codebooks.shape[1] < 2**32

    if scales.shape == (out_features // out_group_size, 1, 1, 1):
        has_output_scales = True
    elif scales.shape == (1, in_features // in_group_size, 1, 1) and in_group_size == 1:
        has_output_scales = False
        input *= scales.squeeze()
    else:
        raise NotImplementedError(f"Can't do Triton AQLM matmul with scales of shape {scales.shape}")

    if not input.is_contiguous():
        raise ValueError("Input tensor must be contiguous")

    output = torch.empty(input.shape[0], out_features, device=device, dtype=dtype)
    for i in range(input.shape[0]):
        # 1D launch kernel where each block computes output unit
        grid = lambda META: (out_features // out_group_size,)
        _aqlm_gemv_simple[grid](
            input[i],
            output[i],
            codes_i16,
            codebooks,
            scales,
            bias,
            in_features,
            out_features,
            num_codebooks,
            codebook_size,
            out_group_size,
            in_group_size,
            num_input_groups,
            next_power_of_2(num_input_groups),
            compute_in_fp32,
            has_output_scales,
            bias is not None,
        )

    return output


def triton_matmul(
    input: torch.Tensor,
    codes: torch.IntTensor,
    codebooks: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor],
    compute_in_fp32: bool = True,
) -> torch.Tensor:
    input_shape = input.shape
    input = input.reshape(-1, input_shape[-1])

    return aqlm_gemm_stupid(
        input,
        codes,
        codebooks,
        scales,
        bias,
        compute_in_fp32,
    ).reshape(input_shape[:-1] + (-1,))
```

### `src/__init__.py`

```python

```

### `src/aq.py`

```python
""" Core mathematics for Additive Quantization (AQ): initialization, reconstruction and beam search"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tqdm.auto import trange

from src.beam_search_l2 import beam_search_optimal_codes as beam_search_minimize_weight_mse
from src.beam_search_xtx import beam_search_optimal_codes as beam_search_minimize_activation_mse
from src.kmeans import find_nearest_cluster, fit_faiss_kmeans, fit_kmeans, fit_kmeans_1d
from src.utils import IntCodes, _dequantize_weight, ellipsis, is_signed


class QuantizedLinear(nn.Module):
    def __init__(self, quantized_weight: QuantizedWeight, bias: Optional[nn.Parameter]):
        super().__init__()
        self.out_features, self.in_features = quantized_weight.out_features, quantized_weight.in_features
        self.quantized_weight = quantized_weight
        self.bias = bias
        self.use_checkpoint = False

    def _forward(self, input: torch.Tensor):
        return F.linear(input, self.quantized_weight(), self.bias)

    def forward(self, input: torch.Tensor):
        if getattr(self, "use_checkpoint", False) and torch.is_grad_enabled():
            return checkpoint(
                self._forward, input, use_reentrant=False, preserve_rng_state=False, determinism_check="none"
            )
        return self._forward(input)


class QuantizedWeight(nn.Module):
    EPS = 1e-9

    def __init__(
        self,
        *,
        reference_weight: torch.Tensor,
        in_group_size: int,
        out_group_size: int,
        num_codebooks: int,
        nbits_per_codebook: int = 8,
        codebook_value_nbits: int = 16,
        codebook_value_num_groups: int = 1,
        scale_nbits: int = 0,
        straight_through_gradient: Optional[bool] = None,
        code_dtype: torch.dtype = torch.int32,
        **init_kwargs,
    ):
        super().__init__()
        self.out_features, self.in_features = reference_weight.shape
        assert self.in_features % in_group_size == 0
        assert self.out_features % out_group_size == 0
        if nbits_per_codebook > torch.iinfo(code_dtype).bits - is_signed(code_dtype):
            raise ValueError(f"Code dtype cannot store {nbits_per_codebook} bits; please specify code_dtype manually")

        self.out_group_size, self.in_group_size = out_group_size, in_group_size
        self.num_codebooks = num_codebooks
        self.nbits_per_codebook = nbits_per_codebook
        self.codebook_size = codebook_size = 2**nbits_per_codebook
        self.codebook_value_nbits = codebook_value_nbits
        self.codebook_value_num_groups = codebook_value_num_groups
        self.codebook_value_clusters = None

        self.scales = self.scales_clusters = self.scales_indices = None
        if straight_through_gradient is None and scale_nbits > 0:
            straight_through_gradient = scale_nbits >= 6
        self.straight_through_gradient = straight_through_gradient
        self.scale_nbits = scale_nbits

        with torch.no_grad():
            weight_groupwise = reference_weight.reshape(
                self.out_features // out_group_size, out_group_size, self.in_features // in_group_size, in_group_size
            ).swapaxes(
                1, 2
            )  # [num_out_groups, num_in_groups, out_group_size, in_group_size]

            if scale_nbits > 0:
                scales = weight_groupwise.norm(dim=(2, 3), keepdim=True) + self.EPS
            else:
                scales = weight_groupwise.flatten(1, -1).norm(dim=-1).view(-1, 1, 1, 1) + self.EPS
            # shape [num_out_groups, num_in_groups, 1, 1] if scale_nbits > 0 else [num_out_groups, num_in_groups, 1, 1]

            self.scales_are_lossless = scale_nbits == 0 or scale_nbits >= 16 or (2**scale_nbits >= scales.shape[1])
            if self.scales_are_lossless or self.straight_through_gradient:
                # ^-- this checks if scales can be preserved losslessly
                self.scales = nn.Parameter(scales, requires_grad=True)
            else:
                scales_clusters, scales_indices, _ = fit_kmeans_1d(scales.flatten(1, -1), k=2**scale_nbits)
                self.scales_clusters = nn.Parameter(scales_clusters, requires_grad=True)
                self.scales_indices = nn.Parameter(scales_indices, requires_grad=False)

            weight_for_init = (weight_groupwise / scales).swapaxes(1, 2).reshape_as(reference_weight)
            del weight_groupwise

        codes, codebooks = init_aq_kmeans(
            weight_for_init,
            num_codebooks=num_codebooks,
            out_group_size=out_group_size,
            in_group_size=in_group_size,
            codebook_size=self.codebook_size,
            **init_kwargs,
        )

        self.codebooks = nn.Parameter(
            codebooks, requires_grad=True
        )  # [num_codebooks, codebook_size, out_group_size, in_group_size]
        self.codes: Optional[nn.Parameter] = nn.Parameter(
            codes.to(code_dtype), requires_grad=False
        )  # [num_out_groups, num_in_groups, num_codebooks]
        self.codes_storage: Optional[IntCodes] = None  # storage for FSDP compatibility

    def get_codes(self) -> torch.IntTensor:
        """Get a non view to codes, regardless of how codes are stored"""
        assert (self.codes is None) != (self.codes_storage is None), "must have either .codes or storage, but not both"
        codes = self.codes if self.codes is not None else self.codes_storage()
        if torch.iinfo(codes.dtype).bits < 32:
            codes = codes.to(torch.int32)  # cast to int32 to allow indexing if codes are int16 or uint8
        return codes

    def set_codes(self, new_codes: torch.Tensor, selection: Union[slice, ellipsis, torch.Tensor] = ..., **kwargs):
        """Update codes[selection] to new_codes, regardless of their dtype and whether they are wrapped as storage"""
        assert (self.codes is None) != (self.codes_storage is None), "must have either .codes or storage, but not both"
        codes_ptr = self.codes if self.codes is not None else self.codes_storage()
        codes_ptr[selection].copy_(new_codes, **kwargs)

    def wrap_codes_for_fsdp_(self, **kwargs):
        """Make this module compatible with FullyShardedDataParallel; modifies state dict in-place"""
        assert self.codes is not None and self.codes_storage is None
        self.codes_storage, self.codes = IntCodes(self.codes, **kwargs), None

    def unwrap_codes_(self):
        """Undo the effect of wrap_codes_for_fsdp_; modifies state dict in-place"""
        assert self.codes is None and self.codes_storage is not None
        self.codes, self.codes_storage = nn.Parameter(self.codes_storage(), requires_grad=False), None

    def get_codebooks(self) -> torch.Tensor:
        """Get quantization codebooks or reconstruct them from second level quantization (see codebook_values_nbits)"""
        if self.codebook_value_nbits >= 16:
            return self.codebooks
        elif 0 < self.codebook_value_nbits < 16:
            with torch.no_grad():
                codebooks_dimshuffle = (
                    self.codebooks.reshape(
                        self.num_codebooks,
                        self.codebook_value_num_groups,
                        self.codebook_size // self.codebook_value_num_groups,
                        self.out_group_size,
                        self.in_group_size,
                    )
                    .permute(0, 1, 3, 4, 2)
                    .flatten(0, -2)
                )
                self.codebook_value_clusters, _unused, reconstructed_codebooks_dimshuffle = fit_kmeans_1d(
                    codebooks_dimshuffle,
                    k=2**self.codebook_value_nbits,
                    initial_clusters=self.codebook_value_clusters,
                )
                reconstructed_codebooks = (
                    reconstructed_codebooks_dimshuffle.view(
                        self.num_codebooks,
                        self.codebook_value_num_groups,
                        self.out_group_size,
                        self.in_group_size,
                        self.codebook_size // self.codebook_value_num_groups,
                    )
                    .permute(0, 1, 4, 2, 3)
                    .reshape_as(self.codebooks)
                )
            if torch.is_grad_enabled():
                reconstructed_codebooks = reconstructed_codebooks + (self.codebooks - self.codebooks.detach())
            return reconstructed_codebooks
        raise NotImplementedError(f"{self.codebook_value_nbits}-bit codebook values are not supported")

    def get_scales(self) -> torch.Tensor:
        """Get per-channel or per-group quantization scales or reconstruct those scales based on scales_nbits"""
        if self.scale_nbits == 0 or self.scales_are_lossless:
            return self.scales  # scales are not quantized or the quantization is lossless
        elif self.straight_through_gradient:
            with torch.no_grad():
                self.scales_clusters, _, dequantized_scales = fit_kmeans_1d(
                    self.scales.flatten(1, -1), k=2**self.scale_nbits, initial_clusters=self.scales_clusters
                )
                dequantized_scales = dequantized_scales.reshape_as(self.scales)
            if torch.is_grad_enabled() and self.scales.requires_grad:
                dequantized_scales = dequantized_scales + (self.scales - self.scales.detach())
            return dequantized_scales
        else:  # train scale codebook only
            return self.scales_clusters.gather(1, self.scales_indices)[:, :, None, None]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.out_features, self.in_features

    def forward(self, selection: Union[slice, ellipsis, torch.Tensor] = ...):
        """
        Differentably reconstruct the weight (or parts thereof) from compressed components
        :param selection: By default, reconstruct the entire weight. If selection is specified, this method will instead
            reconstruct a portion of weight for the corresponding output dimensions (used for parallelism).
            The indices / slices must correspond to output channels (if out_group_size==1) or groups (if > 1).
            Formally, the indices must be in range [ 0 , self.out_features // self.out_group_size )

        """
        weight = _dequantize_weight(self.get_codes()[selection], self.get_codebooks(), self.get_scales()[selection])
        return weight

    @torch.no_grad()
    def beam_search_update_codes_(
        self,
        *,
        XTX: Optional[torch.Tensor] = None,
        reference_weight: torch.Tensor,
        selection: Union[slice, ellipsis, torch.LongTensor] = ...,
        **kwargs,
    ) -> torch:
        """
        Update own codes in-place via beam search so as to minimize squared errors. Return the updated codes.
        :param reference_weight: original weight matrix that is being quantized, shape: [out_features, in_features]
        :param XTX: pairwise products of input features matmul(X.transpose(), X), shape: [in_features, in_features]
          - if XTX is divided by dataset size, this function will return *mean* squared error
          - if XTX is not specified, this function minimizes squared error between weights, as if XTX was identity

        :note: if selection is specified, reference_weight must instead be [num_selected_out_features, in_features]
        :param selection:  By default, this function updates all codes, If selection specified, it will instead
            update only the codes for a portion of output dimensions (used for parallelism).
            The indices / slices must correspond to output channels (if out_group_size==1) or groups (if > 1).
            Formally, the indices must be in range [ 0 , self.out_features // self.out_group_size )
        :param beam_size: consider up to this many best encoding combinations (this param is passed through via kwargs)
        :param kwargs: any additional keyword arguments are forwarded to beam_search_optimal_codes function
        :returns: the updated codes, in the same shape as self.get_codes()[selection]
        """
        codebooks = self.get_codebooks()
        prev_codes = self.get_codes()[selection]
        scales = self.get_scales()[selection]
        if XTX is not None:
            new_codes = beam_search_minimize_activation_mse(
                XTX=XTX,
                reference_weight=reference_weight,
                codebooks=codebooks,
                prev_codes=prev_codes,
                scales=scales,
                **kwargs,
            )
        else:
            new_codes = beam_search_minimize_weight_mse(
                reference_weight=reference_weight, codebooks=codebooks, prev_codes=prev_codes, scales=scales, **kwargs
            )
        self.set_codes(new_codes, selection)
        return new_codes

    def estimate_nbits_per_parameter(self) -> float:
        """Calculate the effective number of bits per original matrix parameters"""
        num_parameters = self.out_features * self.in_features
        group_size = self.out_group_size * self.in_group_size
        num_out_groups = self.out_features // self.out_group_size
        num_in_groups = self.in_features // self.in_group_size

        matrix_store = num_parameters // group_size * self.num_codebooks * self.nbits_per_codebook

        codebooks_store = self.num_codebooks * self.codebook_size * group_size * self.codebook_value_nbits
        if self.codebook_value_nbits < 16:
            codebooks_store += (
                2**self.codebook_value_nbits * self.num_codebooks * self.codebook_value_num_groups * group_size * 16
            )

        if self.scale_nbits >= 16 or 2**self.scale_nbits >= num_in_groups:  # group-wise scales in 16 bit
            scale_store = self.scale_nbits * num_out_groups * num_in_groups
        elif 0 < self.scale_nbits < 16:  # use scale quantization codebooks
            scale_store = self.scale_nbits * num_out_groups * num_in_groups
            scale_store += num_out_groups * 2**self.scale_nbits * 16
        elif self.scale_nbits == 0:  # no group-wise scales; use global 1d scales instead
            scale_store = num_out_groups * 16
        else:
            assert False

        return (matrix_store + codebooks_store + scale_store) / num_parameters

    def extra_repr(self) -> str:
        return f"{self.out_features=}, {self.in_features=}, bits_per_parameter={self.estimate_nbits_per_parameter()}"


@torch.no_grad()
def init_aq_kmeans(
    reference_weight: torch.Tensor,
    *,
    num_codebooks: int,
    out_group_size: int,
    in_group_size: int,
    codebook_size: int,
    verbose: bool = False,
    use_faiss: bool = False,
    max_points_per_centroid: Optional[int] = None,
    max_iter: int = 1000,
    devices: Optional[List[torch.device]] = None,
    **kwargs,
):
    """
    Create initial codes and codebooks using residual K-means clustering of weights
    :params reference_weight, num_codebooks, out_group_size, in_group_size, nbits, verbose: same as in QuantizedWeight
    :params use_faiss  whether to use faiss implementation of kmeans or pure torch
    :params max_point_per_centorid maximum data point per cluster
    :param kwargs: any additional params are forwarded to fit_kmeans
    """
    out_features, in_features = reference_weight.shape
    num_out_groups = out_features // out_group_size
    num_in_groups = in_features // in_group_size
    weight_residue = (
        reference_weight.reshape(num_out_groups, out_group_size, num_in_groups, in_group_size)
        .clone()
        .swapaxes(-3, -2)
        .reshape(num_out_groups * num_in_groups, out_group_size * in_group_size)
    )
    codebooks = []
    codes = []

    if max_points_per_centroid is not None:
        print("Clustering:", max_points_per_centroid * codebook_size, "points from", weight_residue.shape[0])

    for _ in trange(num_codebooks, desc="initializing with kmeans") if verbose else range(num_codebooks):
        if use_faiss:
            codebook_i, codes_i, reconstructed_weight_i = fit_faiss_kmeans(
                weight_residue,
                k=codebook_size,
                max_iter=max_iter,
                gpu=(weight_residue.device.type == "cuda"),
                max_points_per_centroid=max_points_per_centroid,
            )
        else:
            chosen_ids = None
            if max_points_per_centroid is not None:
                chosen_ids = torch.randperm(weight_residue.shape[0], device=weight_residue.device)[
                    : max_points_per_centroid * codebook_size
                ]
            codebook_i, _, _ = fit_kmeans(
                weight_residue if chosen_ids is None else weight_residue[chosen_ids, :],
                k=codebook_size,
                max_iter=max_iter,
                devices=devices,
                **kwargs,
            )
            codes_i, reconstructed_weight_i = find_nearest_cluster(weight_residue, codebook_i, devices=devices)

        codes_i = codes_i.reshape(num_out_groups, num_in_groups, 1)
        codebook_i = codebook_i.reshape(1, codebook_size, out_group_size, in_group_size)
        weight_residue -= reconstructed_weight_i
        codes.append(codes_i)
        codebooks.append(codebook_i)
        del reconstructed_weight_i
    codebooks = torch.cat(codebooks, dim=0)
    codes = torch.cat(codes, dim=-1)
    return codes, codebooks
```

### `src/beam_search_l2.py`

```python
"""Beam search that minimizes ||Wref - Wq||^2 w.r.t. Wq"""
import math
import random
import time
from typing import List, Optional

import torch
import torch.nn.functional as F

from src.utils import _dequantize_weight, maybe_script


@torch.inference_mode
def beam_search_optimal_codes(
    reference_weight: torch.Tensor,
    codebooks: torch.Tensor,
    prev_codes: torch.Tensor,
    scales: Optional[torch.Tensor],
    beam_size: int,
    stochastic_rounding_tau: float = 0.0,
    chunk_size_bytes: int = 2**32,
    dim_rng: Optional[random.Random] = None,
    force_update: bool = False,
    max_update_fraction: float = 1.0,
    code_selection_temperature: float = 0,
    trust_ratio: Optional[float] = None,
) -> torch.Tensor:
    """
    Update codes using beam search to minimize L2 error in code values (regardless of activations)
    :param reference_weight: a target for L2 error, [out_features, in_features]
    :param codebooks: look-up tables of codes, shape: [num_codebooks, codebook_size, out_group_size, in_group_size]
    :param prev_codes: previous-best integer weight codes, shape: [num_output_groups, num_input_groups, num_codebooks]
    :param scales: weight will be multiplied by this factor, shape = [num_output_groups, num_input_groups or 1, 1, 1]
    :param dim_rng: a source of randomness to (optionally) shuffle the order in which the beam search runs
      None = update dimensions and codebooks in their natural order (0, 1, ..., n)
      random.Random(optional_seed) = shuffle dimensions at random, optionally using the specified seed

    :param beam_size: consider up to this many best encoding combinations
    :param stochastic_rounding_tau: if positive, each time the algorithm chooses a code, it will have a probability
        of replacing it with the second-best choice. If the two best codes increase the error by delta1 and delta2,
        then the probability of choosing each code is P_i = delta_i ^ -1/tau / (sum_j_in_choices delta_j ^ -1/tau).
        Note that if there is a code that has zero error, the algorithm will choose allways choose such a code
    :param chunk_size_bytes: process this many candidates at a time; reduce to save memory
    :param force_update: if True, the algorithm will force codes to change even if code is optimal in terms
     of mean squared error. By default, the algorithm forces *all* weights to update this way, which may change weights
     too much. To limit the numer of updated weights, set max_code_change and trust_ratio.
    :param max_update_fraction: the maximum portion of discrete code groups that *can* be updated;
        By default, all codes can be updated. If < 1, only this portion of all code groups is allowed to update.
        The algorithm selects the codes for update based on the difference between de-quantized and reference_weight.
        If there are multiple codebooks, changing any one code responsible for the group counts as code group changed.
        Note that small max_code_change also speeds up computation since not all codes need beam search.
        If the number of weights do not divide evenly, the algoritm will round the number of updates up.
    :param code_selection_temperature: only used if max_code_change > 1; by default, prioritize updating the codes with
        the largest delta = ||(reference_weight - quantized_weight) * mask_only_weights_that_depend_on_this_code||_2 .
        If temperature > 0, the updated codes are instead *sampled* at random, proportionally to delta^(1/temperature) .
    :param trust_ratio: if not None, the algorithm only admits code changes as long as they do not change too much.
        Formally, ||new_quantized_weight - prev_quantized_weight|| / ||prev_quantized_weight|| <= trust_ratio
        If this is not true, the algorithm will reset some of the new quantized weights to their old values until the
        constraint becomes satisfied. The algorithm still prioritizes changes to weights with largest delta (see above).
        If code_change_temperature > 0, the algorithm instead samples which weights to change with the same probability.
        The algorithm will always allow changing exactly *one* code in excess of trust ratio to ensure that at least
        one weight is updated. If both this and max_code_change is set, both these constraints are enforced.
    :return: the best quantization codes found within constraints, same shape as prev_codes

    """
    assert 0 < max_update_fraction <= 1 and (trust_ratio is None or trust_ratio > 0)
    # reshape references, codes and codebooks so they are no longer group-wise
    num_output_groups, num_input_groups, num_codebooks = prev_codes.shape
    _num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape

    flat_unscaled_reference = reference_weight.reshape(
        num_output_groups, out_group_size, num_input_groups, in_group_size
    ).permute(
        0, 2, 1, 3
    )  # [num_output_groups, num_input_groups, out_group_size, in_group_size]
    if scales is not None:
        flat_unscaled_reference = flat_unscaled_reference / scales
        # divide by scales; the resulting problem is equivalent to multiplying dequantized weight
    flat_unscaled_reference = flat_unscaled_reference.flatten(2, 3).flatten(0, 1)
    flat_prev_codes = prev_codes.flatten(0, -2)
    flat_codebooks = codebooks.flatten(-2, -1).detach()
    dim_order = list(range(num_codebooks))
    if dim_rng is not None:
        dim_rng.shuffle(dim_order)

    def _update_flat_codes(_flat_reference, _flat_codes):
        """update _flat_codes [num_groups, num_codebooks] to approximate _flat_reference [num_groups, group_size]"""
        if num_codebooks == 1 and beam_size == 1 and stochastic_rounding_tau == 0 and not force_update:
            # a faster algorithm for a special case of one codebook
            return _greedy_find_best_codes(
                reference=_flat_reference,
                codebook=flat_codebooks[0],
                chunk_size_values=chunk_size_bytes // _flat_reference[0, 0].nbytes,
                code_dtype=prev_codes.dtype,
            )
        else:
            return _beam_search_update_codes_groupwise(
                reference=_flat_reference,
                codebooks=flat_codebooks,
                codes=_flat_codes,
                beam_size=beam_size,
                stochastic_rounding_tau=stochastic_rounding_tau,
                force_update=force_update,
                chunk_size_values=chunk_size_bytes // _flat_reference[0, 0].nbytes,
                dim_order=dim_order,
            )

    def _groupwise_squared_norms(delta: torch.Tensor):
        """
        Given a matrix delta [out_features, in_features], compute a tensor [num_output_groups, num_input_groups] that
        contains the squared sum of elements of delta from each tile of (out_group_size, in_group_size) values.
        """
        return (
            delta.view(delta.shape[0] // out_group_size, out_group_size, delta.shape[1] // in_group_size, in_group_size)
            .square()
            .sum(dim=(1, 3))
        )

    flat_indices_to_update = prev_dequantized_weight = None
    if max_update_fraction < 1 or trust_ratio is not None:
        # precompute ordered code indices to be used for constraints on the number of updates
        prev_dequantized_weight = _dequantize_weight(prev_codes, codebooks, scales)
        num_codes_to_update = int(math.ceil(max_update_fraction * num_output_groups * num_input_groups))
        difference_with_reference_squared_norms = _groupwise_squared_norms(reference_weight - prev_dequantized_weight)
        # ^-- [num_output_groups, num_input_groups]
        if code_selection_temperature > 0:
            flat_indices_to_update = torch.pow(
                difference_with_reference_squared_norms.flatten(),
                0.5 / code_selection_temperature,
                # note: temperature is multuplied by 0.5 because sampling is proportional to norms without square
            ).multinomial(num_samples=num_codes_to_update, replacement=False)
        else:
            flat_indices_to_update = torch.topk(
                difference_with_reference_squared_norms.flatten(), k=num_codes_to_update, largest=True, sorted=True
            ).indices

    if max_update_fraction == 1:
        flat_new_codes = _update_flat_codes(flat_unscaled_reference, flat_prev_codes)
    else:
        flat_new_codes = flat_prev_codes.index_put(  # note: this is an out-of-place op that does not modify prev codes
            (flat_indices_to_update[:, None], torch.arange(num_codebooks, device=codebooks.device)[None, :]),
            _update_flat_codes(
                flat_unscaled_reference[flat_indices_to_update], flat_prev_codes[flat_indices_to_update]
            ),
        )

    if trust_ratio is not None:
        assert isinstance(flat_indices_to_update, torch.Tensor) and isinstance(prev_dequantized_weight, torch.Tensor)
        new_dequantized_weight = _dequantize_weight(flat_new_codes.view_as(prev_codes), codebooks, scales)
        weight_change_squared_norms = _groupwise_squared_norms(new_dequantized_weight - prev_dequantized_weight)
        # ^-- shape: [num_output_groups, num_input_groups]

        flat_ordered_weight_change_squared_norms = weight_change_squared_norms.flatten()[flat_indices_to_update]
        flat_ordered_cumulative_norms = flat_ordered_weight_change_squared_norms.cumsum(0).sqrt()
        # [num_codes_to_update]

        num_codes_selected = 1 + torch.searchsorted(
            flat_ordered_cumulative_norms, trust_ratio * prev_dequantized_weight.norm(), side="left"
        )
        truncated_flat_indices_to_update = flat_indices_to_update[:num_codes_selected]  # sorted most to least important
        flat_new_codes = flat_prev_codes.index_put(  # <-- note: this is an out-of-place operation
            (truncated_flat_indices_to_update[:, None], torch.arange(num_codebooks, device=codebooks.device)[None, :]),
            flat_new_codes[truncated_flat_indices_to_update],
        )
    return flat_new_codes.view_as(prev_codes)


@maybe_script
def _beam_search_update_codes_groupwise(
    reference: torch.Tensor,
    codebooks: torch.Tensor,
    codes: torch.Tensor,
    *,
    beam_size: int,
    stochastic_rounding_tau: float,
    chunk_size_values: int,
    dim_order: Optional[List[int]],
    force_update: bool,
) -> torch.Tensor:
    """
    :param reference: [num_groups, group_size]
    :param codes: [num_groups, num_codebooks]
    :param codebooks: [num_codebooks, codebook_size, group_size]
    :returns: [num_groups, num_codebooks]
    """
    if stochastic_rounding_tau > 0:
        assert beam_size >= 2, "with stochastic rounding, we need at least 2 hypotheses to choose from"

    prev_codes = codes
    device = reference.device
    num_groups, group_size = reference.shape
    num_codebooks, codebook_size, group_size = codebooks.shape
    codebook_offsets = torch.arange(0, num_codebooks * codebook_size, codebook_size, device=device)  # [num_codebooks]
    original_dequantized_vectors = F.embedding_bag(
        codes + codebook_offsets, codebooks.flatten(0, 1), mode="sum"
    )  # [num_groups, group_size]
    if dim_order is None:
        dim_order = list(range(num_codebooks))

    code_norms_sq = codebooks.square().sum(-1)  # [num_codebooks, codebook_size]
    beam_codes = codes.clone().unsqueeze(1)  # [num_groups, current_beam_size, num_codebooks]
    residue = (reference - original_dequantized_vectors).view(num_groups, 1, group_size)
    # shape: [num_groups, current_beam_size, group_size]
    direction = residue.clone().view(num_groups, group_size) if force_update else torch.empty(0)

    for i, codebook_index in enumerate(dim_order):
        current_beam_size = residue.shape[1]
        is_last_step = i == len(dim_order) - 1
        # ^-- [num_groups, current_beam_size, group_size]
        residue = residue + F.embedding(beam_codes[..., codebook_index], codebooks[codebook_index, ...])
        if beam_size > 1 or stochastic_rounding_tau > 0:
            residue_norms_sq = residue.square().sum(-1).unsqueeze(-1)  # [num_groups, current beam size, 1]
        else:
            residue_norms_sq = torch.empty(0, device=device)  # when doing greedy search, these are const

        if not is_last_step:
            target_num_candidates = beam_size + int(stochastic_rounding_tau > 0)
        else:
            target_num_candidates = 2 if stochastic_rounding_tau > 0 or force_update else 1

        flat_best_indices = torch.empty(num_groups, target_num_candidates, device=device, dtype=codes.dtype)
        chunk_size_rows = chunk_size_values // (codebook_size * current_beam_size) // 32
        for chunk_start in range(0, num_groups, chunk_size_rows):
            chunk_end = min(chunk_start + chunk_size_rows, num_groups)
            scores = torch.matmul(residue[chunk_start:chunk_end], codebooks[codebook_index].T)
            if beam_size > 1 or stochastic_rounding_tau > 0:
                scores = residue_norms_sq[chunk_start:chunk_end] - 2 * scores + code_norms_sq[codebook_index]
            else:
                scores = -2 * scores + code_norms_sq[codebook_index]  # residue norms are const(j)
            # ^-- [num_groups_chunk, beam_size, codebook_size]

            flat_best_losses_chunk, flat_best_indices_chunk = torch.topk(
                scores.flatten(1, 2),
                k=target_num_candidates,
                largest=False,
                sorted=is_last_step or beam_size > 1 or stochastic_rounding_tau > 0,
            )  # [num_groups_chunk, target_num_candidates]

            if stochastic_rounding_tau > 0:
                errors = flat_best_losses_chunk.relu().sqrt()  # non-squared errors
                scores = torch.pow(errors / errors.sum(-1, keepdim=True), -1 / stochastic_rounding_tau)
                # ^-- [num_groups_chunk, beam_size + 1]
                keep_prob = scores[:, :-1] / (scores[:, :-1] + scores[:, 1:])  # [num_groups, k_best]
                keep_prob = torch.where(torch.isinf(scores[:, :-1]), 1.0, keep_prob)
                keep = torch.less_equal(torch.rand_like(keep_prob), keep_prob)
                flat_best_indices_chunk = torch.where(
                    keep, flat_best_indices_chunk[:, :-1], flat_best_indices_chunk[:, 1:]
                )

            flat_best_indices[chunk_start:chunk_end] = flat_best_indices_chunk

        arange_num_groups = torch.arange(num_groups, device=device)
        best_hypo_source_ids = flat_best_indices // codebook_size
        best_hypo_codes = flat_best_indices % codebook_size
        beam_codes = beam_codes[arange_num_groups[:, None], best_hypo_source_ids, :]
        beam_codes[:, :, codebook_index] = best_hypo_codes.to(beam_codes.dtype)
        # ^-- [num_groups, beam_size, num_codebooks]

        if not is_last_step:
            residue = residue - F.embedding(beam_codes[..., codebook_index], codebooks[codebook_index, ...])

    if force_update:
        assert beam_codes.shape[1] == 2
        best_codes = beam_codes[:, 0, :]
        second_best_codes = beam_codes[:, 1, :]
        best_code_changed = torch.ne(best_codes, prev_codes).any(dim=-1)
        return torch.where(best_code_changed.unsqueeze(-1), best_codes, second_best_codes)
    else:
        return beam_codes[:, 0, :]


@maybe_script
def _greedy_find_best_codes(
    reference: torch.Tensor, codebook: torch.Tensor, chunk_size_values: int, code_dtype: torch.dtype
) -> torch.Tensor:
    """
    :param reference: [num_groups, group_size]
    :param codebook: [codebook_size, group_size]
    :param chunk_size_values: how many values can be materialized in memory simultaneously
    :parma code_dtype the dtype of optimal codes returned by this function
    :returns: codes [num_groups, 1]
    """
    codebook_t = codebook.T.contiguous()
    chunk_size = chunk_size_values // len(codebook)
    codebook_norms_sq = codebook.square().sum(dim=-1)
    new_codes = torch.empty((len(reference),), dtype=code_dtype, device=reference.device)
    for chunk_start in range(0, len(reference), chunk_size):
        new_codes[chunk_start : chunk_start + chunk_size] = torch.addmm(
            codebook_norms_sq[None], reference[chunk_start : chunk_start + chunk_size], codebook_t, alpha=-2
        ).argmin(-1)
    return new_codes.unsqueeze(-1)


def _find_optimal_codebooks(
    reference: torch.Tensor,
    codebooks: torch.Tensor,
    codes: torch.Tensor,
) -> torch.Tensor:
    num_samples = len(reference)
    num_codebooks, codebook_size, group_size = codebooks.shape

    # compute optimal codebooks via linsolve
    codebook_offsets = torch.arange(num_codebooks, device=codes.device) * codebook_size
    code_indicators = torch.sparse_coo_tensor(
        indices=torch.stack(
            [
                torch.arange(num_samples * num_codebooks, device=codes.device) // num_codebooks,
                (codes + codebook_offsets).flatten(),
            ],
            0,
        ),
        values=torch.ones(num_samples * num_codebooks, device=codes.device),
        size=(num_samples, num_codebooks * codebook_size),
    )
    cooc = (code_indicators.T @ code_indicators).coalesce()
    rhs = code_indicators.T @ reference

    try:
        cooc = cooc.to_dense()
        cooc[torch.arange(len(cooc)), torch.arange(len(cooc))].clamp_min_(1.0)
        optimal_codebooks = (torch.linalg.lstsq(cooc, rhs)).solution.reshape(num_codebooks, codebook_size, group_size)
    except Exception as e:
        print(f"Linsolve failed with {e}")
        optimal_codebooks = codebooks
    return optimal_codebooks
```

### `src/beam_search_xtx.py`

```python
""" Beam search that minimizes ||XWref - XWq||^2 w.r.t. Wq codes """
import random
from typing import Optional, Tuple

import torch
from torch.nn import functional as F
from tqdm.asyncio import trange

from src.utils import _dequantize_weight, maybe_script


@torch.inference_mode()
def beam_search_optimal_codes(
    *,
    XTX: torch.Tensor,
    reference_weight: torch.Tensor,
    codebooks: torch.Tensor,
    prev_codes: torch.IntTensor,
    scales: Optional[torch.Tensor],
    beam_size: int,
    dim_rng: Optional[random.Random] = None,
    sparsity_regularizer: float = 0,
    verbose: bool,
):
    """
    :param XTX: pairwise products of input features matmul(X.transpose(), X), shape: [in_features, in_features]
    :note: if XTX is divided by dataset size, this function will return *mean* squared error
    :param reference_weight: original weight matrix that is being quantized, shape: [out_features, in_features]
    :param codebooks: look-up tables of codes, shape: [num_codebooks, codebook_size, out_group_siz, in_group_size]
    :param prev_codes: previous-best integer weight codes, shape: [num_out_groups, num_in_groups, num_codebooks]
    :param scales: weight will be multiplied by this factor, shape = [num_out_groups, num_in_groups or 1, 1, 1]
    :param dim_rng: a source of randomness to (optionally) shuffle the order in which the beam search runs
      None = update dimensions and codebooks in their natural order (0, 1, ..., n)
      random.Random(optional_seed) = shuffle dimensions at random, optionally using the specified seed

    :param beam_size: consider up to this many best encoding combinations
    :param sparsity_regularizer: subtract this value from beam search objective each time you have a zero code somewhere
    :param verbose: if True, draw a progressbar and periodically print best loss
    :return: best quantization codes found, same shape as prev_codes

    :intuition: the beam search needs to produce weight codes that minimize MSE error
    - the codes are of shape [out_features / out_group_size, in_features / in_group_size, num_codebooks]

    Out of those three dimensions, out_features is "independent", i.e. changing code in
    one output feature does not increase the MSE error for another feature. Therefore,
    beam search for different output features can run in independently in parallel.

    Neither (in_features / in_group_size) nor (num_codebooks) dimension are independent:
    - changing the encoding for one feature can compensate the error from encoding another, OBC-style
    - for a single weight group, changing code in one codebook can affect the optimal choice in another codebook
    Therefore, beam search must go in a double loop over (in_features/in_group_size) and (num_codebooks) dimensions

    This leaves one choice: which dimension used for outer loop, and which one goes is in the inner loop?
    Due to the nature of beam search, interactions between dimensions of inner loop will be explored better.
    We chose to use (in_features/in_group_size) in the outer loop and (num_codebooks) for the inner loop.
    This is based on an intuition from GPTQ: you can get decent performance by quantizing each input unit ...
    ... greedily --- GPTQ does not change quantizations for previously quantized features and works fine.
    Therefore, we believe that we can also use a greedy approach to compensate error between input features.
    In turn, we believe that the codes used to encode the same weights (additively) are more inter-dependent.
    This should be treated as an educated guess with no proof and no ablation (as of the time of writing).

    """
    num_out_groups, num_in_groups, num_codebooks = prev_codes.shape
    num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape
    in_features = num_in_groups * in_group_size
    out_features = num_out_groups * out_group_size
    assert reference_weight.shape == (out_features, in_features)
    prev_weight = _dequantize_weight(prev_codes, codebooks, scales)

    # initialize all beam codes as previous codes - so they can be updated during beam search
    beam_codes = prev_codes.unsqueeze(0)
    # beam_codes shape: [current beam_size, num_out_groups, num_in_groups, num_codebooks], initial beam_size = 1
    beam_weights = prev_weight.unsqueeze(0)
    # beam_weights shape: [current beam_size, out_features, in_features], initial beam size = 1

    beam_losses = (
        _channelwise_squared_error(XTX, prev_weight, reference_weight)
        .reshape(1, num_out_groups, out_group_size)
        .sum(-1)
    )
    # beam_losses shape: [current beam_size, num_out_groups], initial beam_size = 1
    if sparsity_regularizer != 0:
        beam_losses = beam_losses - sparsity_regularizer * (prev_codes == 0).sum(dim=(-1, -2))[None, :]

    if verbose:
        progressbar = trange(num_in_groups * num_codebooks)

    def _make_range(n: int) -> list:
        seq = list(range(n))
        if dim_rng is not None:
            dim_rng.shuffle(seq)
        return seq

    for input_group_index in _make_range(num_in_groups):
        for codebook_index in _make_range(num_codebooks):
            ### part 1: compute losses for every possible candidate for one given codebook and input group.
            # Currently, we compute errors for all output features in parallel in a vectorized fashion.
            best_losses, best_indices = _beam_search_squared_errors(
                XTX=XTX,
                reference_weight=reference_weight,
                codebooks=codebooks,
                scales=scales,
                beam_losses=beam_losses,
                beam_codes=beam_codes,
                beam_weights=beam_weights,
                input_group_index=input_group_index,
                codebook_index=codebook_index,
                k_best=beam_size,
                sparsity_regularizer=sparsity_regularizer,
            )  # [current beam_size, codebook_size, num_out_groups]

            # part 2: select beam_size new best codes and re-arrange beam to account for the fact that ...
            # ... sometimes two or more top candidates originate from the same source in previous beam
            beam_codes, beam_weights, beam_losses = _beam_search_select_best(
                beam_codes=beam_codes,
                beam_weights=beam_weights,
                codebooks=codebooks,
                scales=scales,
                input_group_index=input_group_index,
                codebook_index=codebook_index,
                best_losses=best_losses,
                best_indices=best_indices,
                beam_size=beam_size,
            )

            if verbose:
                progressbar.update()
                if (input_group_index * num_codebooks + codebook_index) % verbose != 0:
                    continue  # if update is an integer, compute metrics every (this many) beam search steps
                best_loss = beam_losses.min(0).values.sum().item() / out_features
                info = f"in_group {input_group_index} / {num_in_groups} "
                info += f"| codebook {codebook_index} / {num_codebooks} "
                if sparsity_regularizer == 0:
                    info += f"| loss {best_loss:.10f}"
                else:  # un-regularize to restore MSE loss, report sparsity rate
                    num_zero_codes = (beam_codes[0] == 0).sum().item()
                    best_loss = best_loss + sparsity_regularizer / out_features * num_zero_codes
                    sparsity = num_zero_codes / prev_codes.numel()
                    info += f"| loss {best_loss:.5f} | sparse {sparsity * 100:.1f}% |"

                progressbar.desc = info
    return beam_codes[0]


@maybe_script
def _beam_search_squared_errors(
    XTX: torch.Tensor,
    reference_weight: torch.Tensor,
    codebooks: torch.Tensor,
    scales: Optional[torch.Tensor],
    beam_losses: torch.Tensor,
    beam_codes: torch.Tensor,
    beam_weights: torch.Tensor,
    input_group_index: int,
    codebook_index: int,
    k_best: int,
    sparsity_regularizer: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute MSE or sum-of-squared-error losses for all possible ways to replace quantization codes for one input group
     and one codebook. Works in parallel for all output-dimension groups.

    :param XTX: pairwise products of input features matmul(X.transpose(), X), shape: [in_features, in_features]
    :note: if both XTX *and* beam_loses are divided by dataset size, this function will return mean squared error
    :param reference_weight: original weight matrix that is being quantized, shape: [out_features, in_features]
    :param codebooks: look-up tables of codes, shape: [num_codebooks, codebook_size, out_group_size, in_group_size]
    :param scales: weight will be multiplied by this factor, [num_out_groups, num_in_groups, 1, 1]

    :param beam_losses: sum-of-squared-error for each hypothesis in beam and for each output channel;
        shape: [beam_size, num_out_groups]
    :param beam_codes: a tensor with best weight codes, shape: [beam_size, num_out_groups, num_in_groups, num_codebooks]
    :param beam_weights: a tensor with de-quantized beam_codes, shape: [beam_size, out_features, in_features]
    :param input_group_index: an index of one group of in_features that is being re-encoded
    :param codebook_index: an index of one codebook for that group of features that is being re-encoded
    :return: tuple(Tensor, Tensor) of 3d tensor of shape = [beam_size, k_best, num_out_groups].
        First one is float tensor of losses of k_best lowest square errors for each beam and out_group
        Second one is int64 tensor of indices of k_best lowest square errors for each beam and out_group

    :note: The code computes MSE using the square-of-difference expansion
     ||X@W.T - sum_i X@(Bi@Ci).T||^2 = ||X@W.T||^2 - 2 <X@W.T, sum_i X@(Bi@Ci).T> + ||sum_i X@Bi@Ci||^2
    where X[nsamples,in_features] is calibration data, W[out_features, in_features] is the reference weight,
       C[num_codebooks, codebook_size, in_features] are learned codebooks (Ci has shape [codebook_size, out_features])
       B[num_codebooks, out_features, codebook_size] are one-hot encoded indices (quantization codes)
    The formula above uses a single group per output "neuron" and a single group.
    The algorithm below generalizes the formula for multiple groups and codebooks.

    Furthermore, the algorithm does not compute the entire formula. Instead, it begins from some baseline loss
    and computes the change in loss from changing a single code to every possible altearnative code.
    When computing the changed loss, the algorithm only computes the few affected parts of the loss formula above.
    """
    num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape
    beam_size, num_out_groups, num_in_groups, num_codebooks = beam_codes.shape
    out_features = num_out_groups * out_group_size

    input_group_slice = slice(input_group_index * in_group_size, (input_group_index + 1) * in_group_size)

    prev_codes_part = beam_codes[:, :, input_group_index, codebook_index]  # [beam_size, num_out_groups]

    if scales is not None:
        scales_part = scales[:, input_group_index % scales.shape[1], :, :]  # [num_out_groups, 1, 1]
    else:
        scales_part = torch.empty(0, device=XTX.device)
    prev_part_dequantized = F.embedding(prev_codes_part, codebooks[codebook_index].flatten(-2, -1)).view(
        beam_size, out_features, in_group_size
    )  # previous codes de-quantized

    prev_weight_part = prev_part_dequantized
    if scales is not None:
        prev_weight_part = (
            prev_weight_part.view(beam_size, num_out_groups, out_group_size, in_group_size)
            .mul(scales_part)
            .view(beam_size, out_features, in_group_size)
        )

    cand_weights = codebooks[codebook_index]  # [codebook_size, out_group_size, in_group_size], all replacement codes

    delta_weight_without_part = reference_weight - beam_weights
    delta_weight_without_part[:, :, input_group_slice] += prev_weight_part

    # dWTXTX is equivalent to < X @ (W - \sum BiCi except current codebook), X @ SOMETHING >
    dWTXTXg = delta_weight_without_part @ XTX[..., input_group_slice]  # [beam_size, out_features, in_group_size]
    # below: use torch.matmul to compute broadcasted batch matrix multiplication; see matmul docs

    XnewBkC_norms_sq = torch.bmm(
        (cand_weights.flatten(0, 1) @ XTX[input_group_slice, input_group_slice]).view(
            codebook_size, 1, out_group_size * in_group_size
        ),
        cand_weights.view(codebook_size, out_group_size * in_group_size, 1),
    ).reshape(
        codebook_size, 1
    )  # [codebook_size, num_out_groups]
    if scales is not None:
        XnewBkC_norms_sq = XnewBkC_norms_sq.mul(scales_part.square().reshape(1, num_out_groups))

    best_losses = torch.empty(
        (beam_size, k_best, num_out_groups), dtype=XTX.dtype, device=XTX.device
    )  # shape: [beam_size, k_best, num_out_groups]
    best_indices = torch.empty(
        (beam_size, k_best, num_out_groups),
        dtype=torch.int64,
        device=XTX.device,
    )
    for beam_id in range(beam_size):
        dot_products = (
            torch.einsum(
                "mg,og->mo",
                cand_weights.reshape(codebook_size, out_group_size * in_group_size),
                dWTXTXg[beam_id].view(num_out_groups, out_group_size * in_group_size),
            )
            .sub_(
                torch.einsum(
                    "og,og->o",
                    prev_part_dequantized[beam_id].reshape(num_out_groups, out_group_size * in_group_size),
                    dWTXTXg[beam_id].view(num_out_groups, out_group_size * in_group_size),
                ).view(1, num_out_groups)
            )
            .view(codebook_size, num_out_groups)
        )
        if scales is not None:
            dot_products = dot_products.mul_(scales_part.reshape(1, num_out_groups))

        XoldBkC_norms_sq = torch.bmm(
            (prev_weight_part[beam_id] @ XTX[input_group_slice, input_group_slice]).view(
                num_out_groups, 1, out_group_size * in_group_size
            ),
            prev_weight_part[beam_id].view(num_out_groups, out_group_size * in_group_size, 1),
        ).reshape(1, num_out_groups)

        # finally, combine them to get MSE
        candidate_squared_errors = (
            beam_losses[beam_id, None, :] - 2 * dot_products + XnewBkC_norms_sq - XoldBkC_norms_sq
        )  # shape: [codebook_size, num_out_groups]

        if sparsity_regularizer != 0:
            candidate_squared_errors += sparsity_regularizer * (prev_codes_part[beam_id] == 0).to(XTX.dtype)[None, :]
            candidate_squared_errors[0, :] -= sparsity_regularizer

        best_beam_squared_errors, best_beam_indices = torch.topk(
            candidate_squared_errors, k_best, dim=0, largest=False, sorted=False
        )
        best_losses[beam_id] = best_beam_squared_errors
        best_indices[beam_id] = best_beam_indices

    return best_losses, best_indices


@maybe_script
def _beam_search_select_best(
    beam_codes: torch.Tensor,
    beam_weights: torch.Tensor,
    codebooks: torch.Tensor,
    scales: Optional[torch.Tensor],
    input_group_index: int,
    codebook_index: int,
    best_losses: torch.Tensor,
    best_indices: torch.Tensor,
    beam_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Select top-:beam_size: and reorder beam accordingly, return new beam
    :param beam_codes: a tensor with best weight codes, shape: [beam_size, num_out_groups, num_in_groups, num_codebooks]
    :param beam_weights: a tensor with de-quantized beam_codes, shape: [beam_size, out_features, in_features]
    :param codebooks: a tensor with look-up tables of codes, shape: [num_codebooks, codebook_size, out_group_size, in_group_size]
    :param scales: weight will be multiplied by this factor, [num_out_groups, num_in_groups, 1, 1]

    :param input_group_index: an index of one group of in_features that is being re-encoded
    :param codebook_index: an index of one codebook for that group of features that is being re-encoded
    :param best_losses: a 3d tensor of losses of k_best lowest square errors for each beam and out group,
        shape = [beam_size, k_best, num_out_groups]
    :param best_indices: a 3d tensor of indices of k_best lowest square errors for each beam and out group,
        shape = [beam_size, k_best, num_out_groups]
    :param beam_size: how many top hypotheses should be selected

    :returns: new (beam_codes, beam_weights, beam_losses)
    """
    dtype = best_losses.dtype
    device = best_losses.device
    _prev_beam_size, k_best, num_out_groups = best_losses.shape
    _prev_beam_size, out_features, in_features = beam_weights.shape
    _prev_beam_size, num_out_groups, num_in_groups, num_codebooks = beam_codes.shape
    flat_best = best_losses.flatten(0, 1).topk(dim=0, k=beam_size, largest=False)
    best_hypo_source_ids = flat_best.indices // k_best
    arange_out_groups = torch.arange(num_out_groups, device=device)
    best_hypo_codes = best_indices.flatten(0, 1)[flat_best.indices, arange_out_groups].reshape(
        beam_size, num_out_groups
    )
    # ^-- shape: [beam_size, num_out_groups]

    # reorder beam codes and weights
    new_beam_codes = torch.full(
        size=(len(best_hypo_codes), num_out_groups, num_in_groups, num_codebooks),
        fill_value=-1,
        dtype=beam_codes.dtype,
        device=device,
    )  # [beam_size, num_out_groups, num_in_groups, num_codebooks]
    new_beam_weights = torch.empty(len(best_hypo_codes), out_features, in_features, dtype=dtype, device=device)

    for beam_index in range(len(best_hypo_codes)):
        new_beam_codes[beam_index, :, ...] = beam_codes[best_hypo_source_ids[beam_index, :], arange_out_groups, ...]
        new_beam_codes[beam_index, :, input_group_index, codebook_index] = best_hypo_codes[beam_index, :]
        new_beam_weights[beam_index, :, :] = _dequantize_weight(new_beam_codes[beam_index, ...], codebooks, scales)

    # Note: the code above can be further accelerated by 1) vectorzing loop and ...
    # ... 2) updating new_beam_weights only for the chosen input group
    return new_beam_codes, new_beam_weights, flat_best.values


@maybe_script
def _channelwise_squared_error(XTX: torch.Tensor, weight: torch.Tensor, reference_weight: torch.Tensor):
    """
    Compute per-channel squared error between X @ weight_or_weights and X @ reference_weight
    :param XTX: pairwise products of input features matmul(X.transpose(), X), shape: [in_features, in_features]
    :note: if XTX is divided by dataset size, this function will return *mean* squared error
    :param weight: predicted/reconstructed weights of shape [*dims, out_features, in_features]
    :param reference_weight: reference weight of shape [out_features, in_features]
    :return: per-channel squared errors of shape [*dims, out_features]
    """
    XW_norm_square = torch.matmul(weight[..., :, None, :], (weight @ XTX)[..., :, :, None]).flatten(-3)
    XWreference_norm_square = torch.bmm(reference_weight[:, None, :], (reference_weight @ XTX)[:, :, None]).flatten(-3)
    dot_product = torch.matmul((reference_weight @ XTX)[:, None, :], weight[..., :, :, None]).flatten(-3)
    return XW_norm_square - 2 * dot_product + XWreference_norm_square
```

### `src/configurable_adam.py`

```python
import math
from contextlib import contextmanager
from typing import Iterable, Optional, Tuple, Union

import torch

from src.utils import maybe_script

NO_DATA = torch.empty(0)


class ConfigurableAdamW(torch.optim.Optimizer):
    r"""
    A version of Adam optimizer that supports custom parameter dtypes, amsgrad, lamb or rmsprop on per-group basis.
    Adam and Amsgrad based on https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py
    Lamb flag based on https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
    This was tested to match Adam and Lamb exactly for torch 2.3.0 (when compute_dtypes are all None)
    :param exp_avg_dtype: dtype for storing first moments; only created if betas[0] != 0; defaults to param dtype
    :param exp_avg_sq_dtype: dtype for storing second moments; only created if betas[1] != 0; defaults to param dtype
    :param v_hat_max_dtype: dtype for storing maximum v_hat; only created if amsgrad=True; defaults to param dtype
    :param exp_avg_device: device for storing exp_avg buffers; only created if betas[0]!=0; defaults to param.device
    :param exp_avg_sq_device: device for storing exp_avg_sq only created if betas[1]!=0; defaults to param.device
    :param v_hat_max_device: device for storing v_hat buffers; only created if amsgrad=True; defaults to param.device
    :note: if any of these devices are CPU, they will be prefetched for optimizer step using pinned memory
    :param compute_dtype: dtype for optimizer step computation; defaults to param dtype
    """

    def __init__(
        self,
        params: Iterable[Union[torch.Tensor, dict]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0,
        debias: Optional[bool] = None,
        amsgrad: bool = False,
        lamb: bool = False,
        clamp_value: Optional[float] = None,
        compute_dtype: Optional[torch.dtype] = None,
        exp_avg_dtype: Optional[torch.dtype] = None,
        exp_avg_sq_dtype: Optional[torch.dtype] = None,
        v_hat_max_dtype: Optional[torch.dtype] = None,
        exp_avg_device: torch.device = None,
        exp_avg_sq_device: torch.device = None,
        v_hat_max_device: torch.device = None,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            debias=debias,
            amsgrad=amsgrad,
            lamb=lamb,
            clamp_value=clamp_value,
            compute_dtype=compute_dtype,
            exp_avg_dtype=exp_avg_dtype,
            exp_avg_sq_dtype=exp_avg_sq_dtype,
            v_hat_max_dtype=v_hat_max_dtype,
            exp_avg_device=exp_avg_device,
            exp_avg_sq_device=exp_avg_sq_device,
            v_hat_max_device=v_hat_max_device,
        )
        super().__init__(params, defaults)

    def _maybe_init_state(self, param: torch.Tensor, group: dict) -> dict:
        state = self.state[param]
        if "step" not in state:
            state["step"] = 0
        if group["betas"][0] != 0 and "exp_avg" not in state:
            pin_memory = group["exp_avg_device"] == torch.device("cpu")
            state["exp_avg"] = torch.zeros_like(
                param,
                dtype=group["exp_avg_dtype"],
                memory_format=torch.preserve_format,
                device=group["exp_avg_device"],
                pin_memory=pin_memory,
            )
        if group["betas"][1] not in (0, 1) and "exp_avg_sq" not in state:
            pin_memory = group["exp_avg_sq_device"] == torch.device("cpu")
            state["exp_avg_sq"] = torch.zeros_like(
                param,
                dtype=group["exp_avg_sq_dtype"],
                memory_format=torch.preserve_format,
                device=group["exp_avg_sq_device"],
                pin_memory=pin_memory,
            )
        if group["amsgrad"] and "v_hat_max" not in state:
            pin_memory = group["v_hat_max_device"] == torch.device("cpu")
            state["v_hat_max"] = torch.zeros_like(
                param,
                dtype=group["v_hat_max_dtype"],
                memory_format=torch.preserve_format,
                device=group["v_hat_max_device"],
                pin_memory=pin_memory,
            )
        return state

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        r"""Performs a single optimization step.
        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, p, state in self.iterate_groups_with_prefetch():
            assert p.grad is not None
            assert not p.grad.is_sparse, f"{self} does not support sparse gradients"
            grad = p.grad.data

            state["step"] += 1
            beta1, beta2 = group["betas"]
            compute_dtype = group.get("compute_dtype") or p.dtype

            if not group["lamb"] and group["weight_decay"] != 0:
                p.data = p.data.mul_(1 - group["lr"] * group["weight_decay"])
                # adam weight decay is not scaled by bias correction

            # Decay the first and second moment running average coefficient
            update = _inner_adam_step_and_update_statistics(
                p,
                grad,
                state.get("exp_avg", p),
                state.get("exp_avg_sq", p),
                state.get("v_hat_max", p),
                beta1,
                beta2,
                group["eps"],
                group["amsgrad"],
                compute_dtype,
            )

            if group["lamb"] and group["weight_decay"] != 0:
                update = update.add(p, alpha=group["weight_decay"])
                # lamb weight decay is later multiplied by -lr * trust_ratio * bias_correction

            update_scale = -group["lr"]
            # below: to save compute, we update scalar coefficient to account for debias/lamb/.. and multiply once
            if group["debias"] if group["debias"] is not None else (not group["lamb"]):
                # if not specified, default to True for Adam, False for Lamb
                mt_debias = 1.0 / (1 - beta1 ** state["step"]) if beta1 != 0 else 1
                vt_debias = 1.0 / math.sqrt(1 - beta2 ** state["step"]) if beta2 != 0 else 1
                bias_correction = mt_debias / vt_debias
                update_scale *= bias_correction

            if group["lamb"]:
                weight_norm = torch.norm(p.data.to(compute_dtype))
                update_norm = torch.norm(update)
                # note: lamb does not count debiasing when computing trust ratio
                if group["clamp_value"] is not None:
                    weight_norm = torch.clamp_max_(weight_norm, group["clamp_value"])
                if weight_norm == 0 or update_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / update_norm
                update_scale *= trust_ratio

            p.data.add_(update, alpha=update_scale)
        return loss

    def iterate_groups_with_prefetch(self):
        """Iterate parameters and optimizer states; skip parameters that do not require grad"""
        flat_params = [
            (group, param) for group, param in _get_flat_param_groups(self.param_groups) if param.grad is not None
        ]

        active_group, active_param = flat_params[0]
        active_state = self._maybe_init_state(active_param, active_group)
        active_state_fetched = _fetch_state_to_device(active_state, active_param.device)

        for next_group, next_param in flat_params[1:] + [(active_group, active_param)]:
            next_state = self._maybe_init_state(next_param, next_group)
            next_state_fetched = _fetch_state_to_device(next_state, next_param.device)

            yield active_group, active_param, active_state_fetched

            _commit_state_updates(active_state, active_state_fetched)

            active_group, active_param, active_state, active_state_fetched = (
                next_group,
                next_param,
                next_state,
                next_state_fetched,
            )


@maybe_script
def _inner_adam_step_and_update_statistics(
    p: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    v_hat_max: torch.Tensor,
    beta1: float,
    beta2: float,
    eps: float,
    amsgrad: bool,
    compute_dtype: torch.dtype,
):
    grad = grad.to(compute_dtype, copy=True)
    stored_exp_avg, stored_exp_avg_sq, stored_v_hat_max = exp_avg, exp_avg_sq, v_hat_max
    if beta1 != 0:
        exp_avg = exp_avg.to(compute_dtype).lerp(grad, 1 - beta1)
        stored_exp_avg.copy_(exp_avg, non_blocking=True)
        update = exp_avg
    else:
        update = grad.clone()

    if beta2 == 1:
        pass
    else:
        if beta2 == 0:
            exp_avg_sq = grad.square()
        else:
            exp_avg_sq = exp_avg_sq.to(compute_dtype).lerp(grad.square(), (1 - beta2))
            stored_exp_avg_sq.copy_(exp_avg_sq, non_blocking=True)
        if amsgrad:
            exp_avg_sq = torch.maximum(exp_avg_sq, v_hat_max, out=exp_avg_sq)
            stored_v_hat_max.copy_(exp_avg_sq, non_blocking=True)

        update /= exp_avg_sq.sqrt().add(eps)

    return update


def _get_flat_param_groups(param_groups):
    return [(group, param) for group in param_groups for param in group["params"]]


def _fetch_state_to_device(state, device):
    fetchable_state_keys = {"exp_avg", "exp_avg_sq", "v_hat_max"}.intersection(state.keys())
    fetched_states = {state_key: state[state_key].to(device, non_blocking=True) for state_key in fetchable_state_keys}
    return state | fetched_states


def _commit_state_updates(offloaded_states, fetched_states):
    fetched_keys = {"exp_avg", "exp_avg_sq", "v_hat_max"}
    for state_key in offloaded_states:
        if state_key not in fetched_keys:
            offloaded_states[state_key] = fetched_states[state_key]
        elif offloaded_states[state_key] is not fetched_states[state_key]:
            offloaded_states[state_key].copy_(fetched_states[state_key], non_blocking=True)
```

### `src/datautils.py`

```python
import os
import random
from itertools import chain
from typing import Optional, Sequence

import numpy as np
import torch
import torch.distributed
from datasets import load_dataset
from torch import nn
from tqdm import trange
from tqdm.auto import tqdm
from transformers import AutoTokenizer


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


def get_loaders(
    name,
    nsamples=128,
    seed=0,
    seqlen=2048,
    eval_mode=False,
    model_path=None,
    use_fast_tokenizer=False,
    trust_remote_code=None,
):
    """
    Loads and prepares data for a Transformers model.
    Args:
        name (str): The name of the dataset to load.
        This can be one of 'wikitext2', 'c4', 'ptb','pajama' for datasets loaded from Huggingface datasets,
        or 'none' for cases where a dataset is not needed, like RTN. It can also accept data path to custom file.
        nsamples (int, optional): The number of samples to load from the dataset. Defaults to 128.
        seed (int, optional): The random seed value for data shuffling and splitting. Defaults to 0.
        seqlen (int, optional): The maximum sequence length for input tokenization. Defaults to 2048.
        model_path (str, optional): The path to the pretrained model weights or full model name.
            used to detect llama to call proper tokenizer.
            see https://github.com/huggingface/transformers/issues/22222#issuecomment-1488578722 for reasons.
        eval_mode (bool, optional). defines slice selection for 'wikitext2', 'c4', 'ptb' datasets.
        leave False for train slice.
        use_fast_tokenizer: whether to use fast tokenizer
        trust_remote_code: whether to trust remote code
    Returns:
        data (torch.utils.data.DataLoader or iterable): Data iterable for the dataset.
    Note:
        the popular decapoda-research Llama models have errors in tokenizer config, specifically
        incorrect token ids for BOS, EOS. This gets corrected to ensure compatibility with transformers
        of versions 4.29 and above.
    """
    set_seed(seed)

    # for pre-tokenized datasets

    if name.lower() == "none":
        print("Not loading any dataset. (OK if you use no compression or methods like RTN.)")
        return None
    elif os.path.isfile(name):
        try:
            data = torch.load(name)[:nsamples]
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Failed to load custom data from {name}.",
                "Check data path or use one of [c4, wikitext2, ptb, pajama, none]",
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=use_fast_tokenizer, trust_remote_code=trust_remote_code
        )

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
            raise ValueError(
                f"Failed to load data from {name}.",
                "Check dataset name or path or use one of [c4, wikitext2, ptb, pajama, none]",
            )

    if hasattr(data, "input_ids"):
        data = data.input_ids

    print(f"Loaded data from {name}; {len(data)=} sequences")
    return data


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
```

### `src/finetune.py`

```python
"""Utilities for internal **block-wise** finetuning used during initial AQLM calibration"""
from __future__ import annotations

import warnings
from argparse import Namespace
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.scatter_gather import Gather

from aq_engine import replace_parameter_
from src.utils import iterate_minibatches


@torch.enable_grad()
def finetune_groupwise(
    *,
    layer: nn.Module,
    train_inps: Sequence[torch.Tensor],
    train_outs: Sequence[torch.Tensor],
    args: Namespace,
    valid_inps: Sequence[torch.Tensor] = None,
    valid_outs: Sequence[torch.Tensor] = None,
    verbose: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Fine-tune a module with pre-quantized linear layers so as to minimize MSE between layer-wise inps/outs

    :param layer: a trainable module where linear layers are replaced by QuantizedLinear instances
    :param inps: a list of tensors of input activations, [nsamples_per_device, seq_len, hidden_size]
    :param outs: a list of tensors of previous output activations, [nsamples_per_device, seq_len, hidden_size]
    :param args: quantization hyperparameters from main.py
    :param kwargs: additional keyword arguments to be passed into layer on each forward
    """
    assert isinstance(args.devices, (list, tuple)) and len(args.devices) >= 1, f"Found devices = {args.devices}"
    assert isinstance(train_inps, (list, tuple)) and isinstance(train_inps, (list, tuple))
    assert len(train_inps) == len(train_outs) == len(args.devices)
    for i in range(len(args.devices)):
        assert isinstance(train_inps[i], torch.Tensor) and isinstance(train_outs[i], torch.Tensor)
        if not args.offload_activations:
            assert train_inps[i].device == train_outs[i].device == args.devices[i], (
                train_inps[i].device,
                train_outs[i].device,
                args.devices,
            )
        else:
            assert train_inps[i].device == train_outs[i].device == torch.device("cpu")
            assert train_inps[i].is_pinned() and train_outs[i].is_pinned()

    # replicate non-trainable parameters to each GPU
    replicas = kwargs_by_device = None
    if len(args.devices) > 1:
        replicas = torch.nn.parallel.replicate(layer, args.devices)
        replicas[0] = layer
        kwargs_by_device = []
        for device in args.devices:
            kwargs_by_device.append(
                {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
            )

    # initialize trainable parameters on main device; prepare to send them to replicas
    differentiable_parameters_by_name = {name: param for name, param in layer.named_parameters() if param.requires_grad}
    param_names, differentiable_parameters = zip(*differentiable_parameters_by_name.items())
    differentiable_parameters = nn.ParameterList(differentiable_parameters)
    for param in differentiable_parameters:
        param.grad = torch.zeros_like(param)
    if replicas:
        replacement_tables = _make_parameter_replacement_tables(layer, replicas, param_names, differentiable_parameters)

    print(f"Fine-tuning {sum(param.numel() for param in differentiable_parameters)} parameters")
    opt = torch.optim.Adam(
        differentiable_parameters, lr=args.finetune_lr, betas=(args.finetune_adam_beta1, args.finetune_adam_beta2)
    )

    assert args.finetune_batch_size % len(args.devices) == 0, "batch_size must be divisible by the number of GPUs"

    num_samples_per_device = len(train_inps[0])
    local_batch_size = args.local_batch_size
    if local_batch_size is None:
        local_batch_size = args.finetune_batch_size // len(args.devices)

    assert all(len(inps_tensor) == num_samples_per_device for inps_tensor in train_inps)
    assert args.finetune_batch_size % (local_batch_size * len(args.devices)) == 0, ""
    num_accumulation_steps = args.finetune_batch_size // (local_batch_size * len(args.devices))
    assert num_samples_per_device % local_batch_size * num_accumulation_steps == 0, (
        num_samples_per_device,
        local_batch_size,
    )
    train_batches_per_epoch = num_samples_per_device // local_batch_size
    train_batch_iterators = [
        iterate_minibatches(train_inps[i], train_outs[i], batch_size=local_batch_size, device=args.devices[i])
        for i in range(len(args.devices))
    ]

    run_validation = False
    if valid_inps and valid_outs:
        run_validation = True
        num_valid_samples_per_device = len(valid_inps[0])
        valid_batches_per_epoch = num_valid_samples_per_device // local_batch_size
        valid_batch_iterators = [
            iterate_minibatches(valid_inps[i], valid_outs[i], batch_size=local_batch_size, device=args.devices[i])
            for i in range(len(args.devices))
        ]

    if run_validation:
        # evaluate before training
        layer.eval()
        loss_numerator = loss_denominator = 0
        with torch.no_grad():
            for _ in range(valid_batches_per_epoch):
                if len(args.devices) == 1:
                    loss = _compute_mse_on_batch(layer, valid_batch_iterators[0], **kwargs)
                else:
                    loss = _compute_mse_parallel(
                        args.devices,
                        replicas,
                        differentiable_parameters,
                        replacement_tables,
                        valid_batch_iterators,
                        kwargs_by_device,
                    )
                loss_numerator += loss.item()
                loss_denominator += 1
        valid_loss_epoch = loss_numerator / loss_denominator
        print(f"Evaluation before training.")
        print(f"valid loss={valid_loss_epoch:.2e}\t")
        best_loss = valid_loss_epoch
        best_parameters_by_name = deepcopy(differentiable_parameters_by_name)
        worse_count = 0

    steps_accumulated = 0
    for epoch in range(args.finetune_max_epochs):
        layer.train()
        # train epoch
        loss_numerator = loss_denominator = 0
        for _ in range(train_batches_per_epoch):
            if len(args.devices) == 1:
                loss = _compute_mse_on_batch(layer, train_batch_iterators[0], **kwargs)
            else:
                loss = _compute_mse_parallel(
                    args.devices,
                    replicas,
                    differentiable_parameters,
                    replacement_tables,
                    train_batch_iterators,
                    kwargs_by_device,
                )

            (loss / num_accumulation_steps).backward()
            steps_accumulated += 1

            if not torch.isfinite(loss).item():
                raise ValueError(f"Fine-tuning loss is {loss}")

            if steps_accumulated >= num_accumulation_steps:
                opt.step()
                opt.zero_grad()
                steps_accumulated = 0

            loss_numerator += loss.item()
            loss_denominator += 1
        train_loss_epoch = loss_numerator / loss_denominator
        if run_validation:
            layer.eval()
            # val epoch
            loss_numerator = loss_denominator = 0
            with torch.no_grad():
                for _ in range(valid_batches_per_epoch):
                    if len(args.devices) == 1:
                        loss = _compute_mse_on_batch(layer, valid_batch_iterators[0], **kwargs)
                    else:
                        loss = _compute_mse_parallel(
                            args.devices,
                            replicas,
                            differentiable_parameters,
                            replacement_tables,
                            valid_batch_iterators,
                            kwargs_by_device,
                        )
                    loss_numerator += loss.item()
                    loss_denominator += 1
            valid_loss_epoch = loss_numerator / loss_denominator
        # log losses in the end of the epoch
        if verbose:
            print("-" * 10)
            print(f"epoch={epoch}")
            print(f"train loss={train_loss_epoch:.2e}\t")
            if run_validation:
                print(f"valid loss={valid_loss_epoch:.2e}\t")

        if run_validation:
            if valid_loss_epoch < best_loss:
                print(f"new best loss {valid_loss_epoch:.2e} on epoch {epoch}")
                best_loss = valid_loss_epoch
                best_parameters_by_name = deepcopy(differentiable_parameters_by_name)
                worse_count = 0
            else:
                worse_count += 1
                if worse_count >= args.finetune_early_stop:
                    break

    if run_validation:
        layer.load_state_dict(best_parameters_by_name, strict=False)

    return layer


def _make_parameter_replacement_tables(
    layer: nn.Module, replicas: Sequence[nn.Module], param_names: Sequence[str], parameters: nn.ParameterList
) -> Sequence[List[Sequence[Tuple[nn.Module, str]]]]:
    """
    Prepare auxiliary data structures for quickly copying parameters to replicas for data-parallel training.

    """
    assert len(param_names) == len(parameters)
    assert len(replicas) > 1
    assert replicas[0] is layer

    parameters_by_name = dict(zip(param_names, parameters))

    param_to_name = {param: name for name, param in parameters_by_name.items()}
    param_occurences = defaultdict(list)  # param_name -> List [ Tuple [submodule name, attr name] ]
    for submodule_name, submodule in layer.named_modules():
        for attr_name, param in submodule.named_parameters(recurse=False):  # immediate params (excluding children)
            if param in param_to_name:
                param_name = param_to_name[param]
                param_occurences[param_name].append((submodule_name, attr_name))
    assert len(param_occurences) == len(parameters), "internal error: not all parameters were found"

    replacement_tables = []
    for replica in replicas:
        replacement_table = list()  # for each master param -> List[ Tuple[replica submodule, attr name] ]
        replica_modules_by_name: Dict[str, nn.Module] = dict(replica.named_modules())

        for param_name, master_param in zip(param_names, parameters):
            param_replacements = list()
            for submodule_name, attr_name in param_occurences[param_name]:
                param_replacements.append((replica_modules_by_name[submodule_name], attr_name))
            replacement_table.append(param_replacements)
        replacement_tables.append(replacement_table)
    return replacement_tables


def _compute_mse_on_batch(
    layer: nn.Module, batch_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]], **kwargs
) -> torch.Tensor:
    """
    Compute the activation MSE error between transformer layers
    :param
    """
    inps_batch, outs_batch = next(batch_iter)
    inps_batch = inps_batch.to(dtype=torch.float32)
    outs_batch = outs_batch.to(dtype=torch.float32)

    if inps_batch.shape[0] != 1:  # replicate kwargs to match the batch size
        for name, value in list(kwargs.items()):
            if isinstance(value, torch.Tensor) and value.shape[0] == 1:
                if name not in ("attention_mask", "position_ids"):
                    warnings.warn(f"Tiling an unexpected kwarg {name} over batch size; make sure this is valid.")
                repeats = [len(inps_batch)] + [1 for _ in range(value.ndim - 1)]
                kwargs[name] = value.tile(*repeats)

    outs_prediction, *_unused = layer(inps_batch, **kwargs)
    assert outs_prediction.shape == outs_batch.shape
    return F.mse_loss(outs_prediction, outs_batch)


def _compute_mse_parallel(
    devices: Sequence[torch.device],
    replicas: Sequence[nn.Module],
    parameters_to_replicate: nn.ParameterList,
    replacement_tables: Sequence[List[Sequence[Tuple[nn.Module, str]]]],
    batch_iterators: Sequence[Iterator[Tuple[torch.Tensor, torch.Tensor]]],
    kwargs_by_device: Sequence[Dict[str, Any]],
) -> torch.Tensor:
    """Compute MSE in parallel over multiple GPUs, each GPU processes a portion of samples"""
    replicated_parameters = torch.nn.parallel.replicate(parameters_to_replicate, devices, detach=False)
    funcs_by_replica = [_compute_mse_on_batch for _ in replicas]
    inputs_by_replica = []
    for i in range(len(devices)):
        if i != 0:  # no overrides needed for master module
            for replacement_param, replacement_table in zip(replicated_parameters[i], replacement_tables[i]):
                for (replica_submodule, attr_name) in replacement_table:
                    replace_parameter_(replica_submodule, attr_name, replacement_param)
        inputs_by_replica.append((replicas[i], batch_iterators[i]))
    mse_components = torch.nn.parallel.parallel_apply(
        funcs_by_replica, inputs_by_replica, kwargs_by_device, devices=devices
    )
    return Gather.apply(devices[0], 0, *(mse.view(1) for mse in mse_components)).mean()
```

### `src/kmeans.py`

```python
import itertools
from typing import List, Optional, Tuple

import torch

from src.utils import maybe_script


@maybe_script
def _kmeans_greedy_init(data: torch.Tensor, k: int) -> torch.Tensor:
    """Get initial clusters by iteratively choosing a vector that is the farthest from already selected clusters"""
    clusters = torch.zeros(k, data.shape[1], device=data.device)
    running_min_distances = torch.full((data.shape[0],), torch.inf, device=data.device, dtype=data.dtype)
    data_norm_squared = data.norm(p=2, dim=1).square()

    for i in range(k):
        clusters[i] = data[running_min_distances.argmax()]
        distances_to_cluster_i = data_norm_squared - 2 * data @ clusters[i] + clusters[i].norm().square()
        running_min_distances = torch.minimum(running_min_distances, distances_to_cluster_i, out=running_min_distances)
    return clusters


@maybe_script
def fit_kmeans(
    data: torch.Tensor,
    k: int,
    max_iter: int = 1000,
    check_every: int = 10,
    rtol: float = 1e-06,
    atol: float = 1e-08,
    greedy_init: bool = False,
    block_size_vals: int = 2**30,
    devices: Optional[List[torch.device]] = None,
):
    """
    :param data: [nsamples, dim]
    :param k: number of centroids
    :param max_iter: run at most this many iterations
    :param check_every: check for convergence (allclose(new_centroids, old_centroids)) once in this many steps
    :param rtol: early stopping relative tolerance for centroids
    :param atol: early stopping absolute tolerance for centroids
    :param greedy_init: if True, init by greedily selecting the point that is farthest from any cluster
        if False (default), initialize with random points using pytorch global RNG
    :param block_size_vals: how many dot products to compute at a time
    :param devices: if specified, run kmeans in data-parallel mode across these devices
    :return: (clusters float[k, dim], data_indices int[nsamples], reconstructed_data: float[nsamples, dim])
    """
    if devices is None:
        devices = [data.device]

    if greedy_init:
        clusters = _kmeans_greedy_init(data, k)
    else:
        clusters = data[torch.randperm(data.shape[0])[:k], :]  # [k, dim]

    block_size = block_size_vals // k
    shard_size = (len(data) - 1) // len(devices) + 1
    data = [
        data[gi * shard_size : (gi + 1) * shard_size].to(devices[gi], non_blocking=True) for gi in range(len(devices))
    ]
    nearest_indices = [torch.empty(len(data[gi]), dtype=torch.int64, device=devices[gi]) for gi in range(len(devices))]
    clusters = [clusters.to(device, non_blocking=True) for device in devices]

    for i in range(max_iter):
        for block_start in range(0, shard_size, block_size):
            for gi in range(len(devices)):
                nearest_indices[gi][block_start : block_start + block_size] = torch.addmm(
                    torch.bmm(clusters[gi][:, None, :], clusters[gi][:, :, None]).flatten(),
                    data[gi][block_start : block_start + block_size],
                    clusters[gi].T,
                    beta=-0.5,
                ).argmax(1)
            # note: the above formula equals to - 0.5 || data[:, None, :] - clusters[None, :, :] || ^ 2 + const

        if len(devices) == 1:
            new_clusters = [
                clusters[0]
                .clone()
                .index_reduce_(dim=0, index=nearest_indices[0], source=data[0], reduce="mean", include_self=False)
            ]
        else:
            cluster_sums = [
                torch.zeros_like(clusters[gi])
                .index_add(dim=0, index=nearest_indices[gi], source=data[gi])
                .to(devices[0], non_blocking=True)
                for gi in range(len(devices))
            ]
            cluster_counts = [
                torch.bincount(nearest_indices[gi], minlength=k).to(devices[0], non_blocking=True)
                for gi in range(len(devices))
            ]
            for gi in range(1, len(devices)):
                cluster_sums[0] += cluster_sums[gi]
                cluster_counts[0] += cluster_counts[gi]

            new_clusters = [cluster_sums[0] / cluster_counts[0].unsqueeze(1).clamp_min(1)]
            new_clusters[0] += (cluster_counts[0].unsqueeze(1) == 0) * clusters[0]
            for gi in range(1, len(devices)):
                new_clusters.append(new_clusters[0].to(devices[gi], non_blocking=True))

        if i % check_every == 0:
            if torch.allclose(new_clusters[0], clusters[0], rtol=rtol, atol=atol):
                break
        clusters = new_clusters
    for block_start in range(0, shard_size, block_size):
        for gi in range(len(devices)):
            nearest_indices[gi][block_start : block_start + block_size] = torch.addmm(
                torch.bmm(clusters[gi][:, None, :], clusters[gi][:, :, None]).flatten(),
                data[gi][block_start : block_start + block_size],
                clusters[gi].T,
                beta=-0.5,
            ).argmax(1)

    clusters = clusters[0]
    nearest_indices = torch.cat([nearest_indices[gi].to(devices[0]) for gi in range(len(devices))], dim=0)
    reconstructed_data = clusters[nearest_indices]
    return clusters, nearest_indices, reconstructed_data


def fit_faiss_kmeans(
    data: torch.Tensor,
    k: int,
    *,
    max_iter: int = 1000,
    gpu: bool = True,
    max_points_per_centroid: Optional[int] = None,
    verbose: bool = True,
):
    """
    :param data: [nsamples, dim]
    :param k: number of centroids
    :param max_iter: run at most this many iterations
    :param gpu: if True, run kmeans on (all available) GPUs; if False, run on CPU
    :param max_points_per_centroid: if specified, train kmeans on a random subset of (this_many * k) points
    :param verbose: if True, faiss.kmeans will print status to stdout

    :return: (clusters float[k, dim], data_indices int[nsamples], reconstructed_data: float[nsamples, dim])
    """
    try:
        import faiss
    except ModuleNotFoundError:
        raise RuntimeError("Faiss is not installed. Please install it before running this function.")

    d = data.shape[1]
    if max_points_per_centroid is not None:
        kmeans = faiss.Kmeans(
            d, k, niter=max_iter, verbose=verbose, gpu=gpu, max_points_per_centroid=max_points_per_centroid
        )
    else:
        kmeans = faiss.Kmeans(d, k, niter=max_iter, verbose=verbose, gpu=gpu)
    kmeans.train(data.cpu())
    clusters = kmeans.centroids
    nearest_indices = kmeans.index.search(data.cpu(), 1)[1][:, 0]
    clusters, nearest_indices = torch.from_numpy(clusters).to(data.device), torch.from_numpy(nearest_indices).to(
        data.device
    )
    reconstructed_data = clusters[nearest_indices]

    return clusters, nearest_indices, reconstructed_data


@maybe_script
def find_nearest_cluster(data, clusters, block_size_vals: int = 2**30, devices: Optional[List[torch.device]] = None):
    """Find nearest clusters for each batch of data and return their indices"""
    if devices is None:
        devices = [data.device]
    block_size = block_size_vals // len(clusters)
    shard_size = (len(data) - 1) // len(devices) + 1
    data = [
        data[gi * shard_size : (gi + 1) * shard_size].to(devices[gi], non_blocking=True) for gi in range(len(devices))
    ]
    nearest_indices = [torch.empty(len(data[gi]), dtype=torch.int64, device=devices[gi]) for gi in range(len(devices))]
    clusters = [clusters.to(device, non_blocking=True) for device in devices]

    for block_start in range(0, shard_size, block_size):
        for gi in range(len(devices)):
            nearest_indices[gi][block_start : block_start + block_size] = torch.addmm(
                torch.bmm(clusters[gi][:, None, :], clusters[gi][:, :, None]).flatten(),
                data[gi][block_start : block_start + block_size],
                clusters[gi].T,
                beta=-0.5,
            ).argmax(1)
    clusters = clusters[0]
    nearest_indices = torch.cat([nearest_indices[gi].to(devices[0]) for gi in range(len(devices))], dim=0)
    reconstructed_data = clusters[nearest_indices]
    return nearest_indices, reconstructed_data


def fit_kmeans_1d(
    groupwise_data: torch.Tensor,
    k: int,
    max_iter: int = -1,
    offset_rate: float = 0,
    verbose: bool = False,
    initial_clusters: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    optimized batch k-means for 1d datapoint using sort
    :param groupwise_data: stuff to be compressed, shape: [num_groups, group_size]
    :param k: the number of centroids to find
    :param max_iter: run for at most this many kmeans iterations (-1 = run until convergence)
    :param offset_rate: if greater than 0, skip this percentage of smallest/largest elements for initialization
    :param verbose: print mse and early stopping info
    :param kwargs: optionally provide rtol=... and atol=... for early stopping;
    :note: if rtol/atol is speficied, these tolerances are measured between cluster centroids from subsequent steps
    :returns: (clusters, indices, restored_data)
        - clusters are centroids of shape
        - indices are integers [0, k) in the same shape as data; they denote the index of the nearest centroid
        - restored_data is a floating point tensor in the same shape as data; they are dequantized(quantized(data))
    :note: to reconstruct clusters manually, call clusters.gather(-1, indices)
    :TODO[aqlm]: torch.jit.script / torch.compile
    """
    assert groupwise_data.ndim == 2
    assert 0 <= offset_rate < 0.5

    # step 2: pre-sort data and initialize kmeans with uniform percentiles
    sorted_data, groupwise_sort_indices = groupwise_data.sort(dim=1)
    groupwise_ranks_1based = groupwise_sort_indices.argsort(-1).add_(1)
    del groupwise_sort_indices

    # ^-- [num_groups, group_size]; sorted by group_size
    sorted_cumsum = torch.cat([torch.zeros_like(sorted_data[:, :1]), sorted_data.cumsum(dim=1)], dim=1)
    # ^-- [num_groups, group_size + 1]; sorted by group_size + 1
    if initial_clusters is not None:
        clusters = initial_clusters
    else:
        offset = int((sorted_data.shape[1] - 1) * offset_rate)
        init_indices = torch.linspace(offset, sorted_data.shape[1] - 1 - offset, k, dtype=torch.int64)
        clusters = sorted_data[:, init_indices]  # shape: [num_groups, k]

    # step 3: run kmeans
    def _groupwise_find_border_indices(clusters, sorted_data):
        borders = (clusters[:, 1:] + clusters[:, :-1]) / 2
        column = clusters[:, :1]
        borders = torch.cat(
            [torch.full_like(column, float("-inf")), borders, torch.full_like(column, float("inf"))], dim=1
        )
        border_indices = torch.searchsorted(sorted_data, borders, side="left")
        return border_indices

    for i in itertools.count():
        border_indices = _groupwise_find_border_indices(clusters, sorted_data)
        sum_by_cluster = torch.diff(sorted_cumsum.gather(1, border_indices), dim=1)
        count_by_cluster = torch.diff(border_indices, dim=1)
        new_cluster_centers = torch.where(
            count_by_cluster > 0,
            sum_by_cluster / count_by_cluster,
            sorted_data.gather(1, border_indices[:, :-1].clamp_max(sorted_data.shape[1] - 1)),
        )
        if torch.allclose(new_cluster_centers, clusters, **kwargs):
            if verbose:
                print(f"Early stopping after {i} iterations")
            break
        clusters = new_cluster_centers
        if max_iter > 0 and i >= max_iter:
            break

    # step 4: determine the final clustering
    border_indices = _groupwise_find_border_indices(clusters, sorted_data)
    groupwise_cluster_indices = torch.searchsorted(border_indices[:, 1:], groupwise_ranks_1based, side="left")
    groupwise_restored_data = clusters.gather(1, groupwise_cluster_indices)
    # [num_groups, k]

    if verbose:
        sorted_cumsum_squares = torch.cat(
            [torch.zeros_like(sorted_data[:, :1]), sorted_data.square().cumsum(dim=1)], dim=1
        )
        sum_by_cluster = torch.diff(sorted_cumsum.gather(1, border_indices), dim=1)
        sum_squares_by_cluster = torch.diff(sorted_cumsum_squares.gather(1, border_indices), dim=1)
        count_by_cluster = torch.diff(border_indices, dim=1).clamp_min(1)
        mse_l2 = (groupwise_restored_data - groupwise_data).square().mean()
        mse_approx = sum_squares_by_cluster - 2 * sum_by_cluster * clusters + count_by_cluster * clusters.square()
        mse_approx = mse_approx.sum(0) / count_by_cluster.sum(0)
        print(f"mse: {mse_l2.mean().item()} , dot-based estimate: {mse_approx.mean().item()}")

    return clusters, groupwise_cluster_indices, groupwise_restored_data
```

### `src/memory_efficient_loss.py`

```python
"""
Utility functions for computing a KL divergence loss without materializing all logits / logprobs simultaneously
"""
import itertools
from typing import Callable, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

T = TypeVar("T")


def compute_kl_divergence_loss_values(
    *,
    student_hidden_states: torch.Tensor,
    student_lm_head: nn.Module,
    teacher_hidden_states: torch.Tensor,
    teacher_lm_head: nn.Module,
    max_tokens_per_chunk: int = 256,
    checkpoint_last_chunk: bool = True,
    **checkpoint_kwargs,
) -> torch.Tensor:
    """
    Compute token-wise KL divergence loss without materializing all logits/logprobs simultaneously
    :param student_hidden_states: input hidden states for student head, [batch_size, sequence_length, student_dim]
    :param student_lm_head: a token-wise layer (e.g. nn.Linear) mapping from student_dim to logits [vocabulary_size]
    :param teacher_hidden_states: input hidden states for teacher head, [batch_size, sequence_length, teacher_dim]
    :param teacher_lm_head: a token-wise layer (e.g. nn.Linear) mapping from teacher_dim to logits [vocabulary_size]
    :note: teacher is applied to hidden states without no_grad. If required, set requires_grad=False on teacher manually
    :param max_tokens_per_chunk: materialize logits logprobs for at most this many tokens at a time
    :param checkpoint_kwargs: additional arguments passed to checkpoint (e.g. use_reentrant or determinism_check)
    :param checkpoint_last_chunk: if False, do not apply gradient checkpointing to the very last chunk of inputs
        since they are the first ones to be re-materialized anyway. Useful if loss is backpropagated immediately.
    :returns: token-wise KL loss values of shape [batch_size, sequence_length]
    """
    assert student_hidden_states.requires_grad or teacher_hidden_states.requires_grad or not torch.is_grad_enabled()
    assert teacher_hidden_states.shape[:-1] == student_hidden_states.shape[:-1]
    flat_student_hidden_states = student_hidden_states.flatten(0, -2)
    flat_teacher_hidden_states = teacher_hidden_states.flatten(0, -2)
    total_tokens = flat_teacher_hidden_states.shape[0]

    loss_values_by_chunk = []
    for chunk_start in range(0, total_tokens, max_tokens_per_chunk):
        is_last_chunk = chunk_start + max_tokens_per_chunk >= total_tokens
        loss_values_by_chunk.append(
            maybe_checkpoint(
                _compute_kl_div_from_flat_hidden_states,
                flat_student_hidden_states[chunk_start : chunk_start + max_tokens_per_chunk],
                student_lm_head,
                flat_teacher_hidden_states[chunk_start : chunk_start + max_tokens_per_chunk],
                teacher_lm_head,
                checkpoint_enabled=torch.is_grad_enabled() and (checkpoint_last_chunk or not is_last_chunk),
                **checkpoint_kwargs,
            )
        )
    return torch.cat(loss_values_by_chunk).reshape(*student_hidden_states.shape[:2])


def _compute_kl_div_from_flat_hidden_states(
    flat_student_hidden_states: torch.Tensor,
    student_lm_head: nn.Module,
    flat_teacher_hidden_states: torch.Tensor,
    teacher_lm_head: nn.Module,
) -> torch.Tensor:
    student_logprobs = F.log_softmax(student_lm_head(flat_student_hidden_states), dim=-1)
    teacher_logprobs = F.log_softmax(teacher_lm_head(flat_teacher_hidden_states), dim=-1)
    return F.kl_div(input=student_logprobs, target=teacher_logprobs, log_target=True, reduction="none").sum(-1)


def maybe_checkpoint(func: Callable[[...], T], *inputs, checkpoint_enabled: bool, **checkpoint_kwargs) -> T:
    """Execute function normally or with checkpointing, depending on checkpoint_enabled. Forward **checkpoint_kwargs"""
    return func(*inputs) if checkpoint_enabled else checkpoint(func, *inputs, **checkpoint_kwargs)


def test_kl_divergence(
    teacher_hidden_size=2048,
    student_hidden_size=1024,
    batch_size=2,
    seq_length=450,
    vocab_size=10_000,
    max_tokens_per_chunk=128,
):
    """Verify correctness of compute_kl_divergence_loss_values"""

    teacher_lm_head = nn.Linear(teacher_hidden_size, vocab_size)
    student_lm_head = nn.Linear(student_hidden_size, vocab_size)

    teacher_hidden_states = torch.randn(batch_size, seq_length, teacher_hidden_size)
    student_hidden_states = torch.randn(batch_size, seq_length, student_hidden_size, requires_grad=True)

    ref_loss_values = F.kl_div(
        input=F.log_softmax(student_lm_head(student_hidden_states), dim=-1),
        target=F.log_softmax(teacher_lm_head(teacher_hidden_states), dim=-1),
        log_target=True,
        reduction="none",
    ).sum(-1)

    for use_reentrant, checkpoint_last_chunk, determinism_check in itertools.product(
        (True, False), (True, False), ("default", "none")
    ):
        loss_values = compute_kl_divergence_loss_values(
            student_hidden_states=student_hidden_states,
            student_lm_head=student_lm_head,
            teacher_hidden_states=teacher_hidden_states,
            teacher_lm_head=teacher_lm_head,
            max_tokens_per_chunk=max_tokens_per_chunk,
            checkpoint_last_chunk=checkpoint_last_chunk,
            use_reentrant=use_reentrant,
            determinism_check=determinism_check,
        )
        assert loss_values.shape == (batch_size, seq_length)
        assert torch.allclose(loss_values, ref_loss_values)
```

### `src/modelutils.py`

```python
import math
import os
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import transformers
from accelerate import dispatch_model
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision
from transformers import AutoConfig, AutoModelForCausalLM

from src.aq import QuantizedWeight

MODEL_ERROR_MSG = "Unsupported model type {} - only 'llama', 'Yi', 'opt', 'falcon', 'phi3' are supported"
FALCON_TYPES = ("falcon", "refinedweb", "refinedwebmodel")
LLAMA_LIKE = ("llama", "Yi", "mistral", "mixtral", "gemma", "cohere", "qwen2")


@contextmanager
def suspend_nn_inits():
    def skip(*args, **kwargs):
        pass

    saved_inits = torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_  # saving
    torch.nn.init.kaiming_uniform_ = torch.nn.init.uniform_ = torch.nn.init.normal_ = skip  # replacing
    try:
        yield
    finally:
        torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = saved_inits  # restoring


def dispatch_quantized_model(model):
    num_devices = torch.cuda.device_count()
    device_map = {"model.embed_tokens": 0, "model.norm": num_devices - 1, "lm_head": 0}
    num_layers = len(get_layers(model))
    layers_per_device = math.ceil(num_layers / num_devices)
    for layer_id in range(num_layers):
        device_id = layer_id // layers_per_device
        device_map[f"model.layers.{layer_id}"] = device_id
    model = dispatch_model(model, device_map)
    # for some reason dispatch doesn't put this modules on needed device
    model.model.embed_tokens = model.model.embed_tokens.to("cuda:0")
    model.lm_head = model.lm_head.to("cuda:0")
    return model


def get_model(
    model_path, load_quantized=None, dtype="auto", device_map=None, attn_implementation=None, trust_remote_code=False
):
    if dtype == "auto":
        dtype = (
            AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code).torch_dtype or "auto"
        )  # force transformers 4.29.2 to follow the same rules as 4.30.x
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    model_kwargs = {}
    # this argument is avaialbe only for transformers >= 4.38.0
    if transformers.__version__ >= "4.38.0":
        model_kwargs["attn_implementation"] = attn_implementation

    with suspend_nn_inits():
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            # defer distribution if loading quantized
            device_map=None if load_quantized else device_map,
            low_cpu_mem_usage=True,
            local_files_only=True,
            **model_kwargs,
        )
        if load_quantized:
            print("Initializing model with random weights...")
            print("Loading quantized model ...")
            model = load_quantized_model(model, load_quantized)
            if device_map == "auto":
                assert model.config.model_type in LLAMA_LIKE, "Dispatching is implemented only for Llama-like models."
                model = dispatch_quantized_model(model)
        else:
            print("Loading pretrained model ...")

    print("Model loaded sucсessfully ...")

    return model


def is_model_for_causal_lm(model: nn.Module):
    assert isinstance(model, transformers.PreTrainedModel)
    assert len(model.base_model_prefix) > 0 and hasattr(model, model.base_model_prefix)
    assert model.get_output_embeddings() is not None
    return True


def get_model_head_with_norm(model):
    head = torch.nn.ModuleList()
    if model.config.model_type in (*LLAMA_LIKE, "phi3"):
        if model.model.norm is not None:
            head.append(model.model.norm)
        head.append(model.lm_head)
    elif model.config.model_type.lower() in FALCON_TYPES:
        if model.transformer.ln_f is not None:
            head.append(model.transformer.ln_f)
        head.append(model.lm_head)
    elif model.config.model_type == "opt":
        if model.model.decoder.final_layer_norm is not None:
            head.append(model.model.decoder.final_layer_norm)
        if model.model.decoder.project_out is not None:
            head.append(model.model.decoder.project_out)
        head.append(model.lm_head)
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))
    return head


def get_lm_logits(inps_, model):
    if model.config.model_type in (*LLAMA_LIKE, "phi3"):
        hidden_states = inps_.unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
    elif model.config.model_type.lower() in FALCON_TYPES:
        hidden_states = inps_.unsqueeze(0)
        if model.transformer.ln_f is not None:
            hidden_states = model.transformer.ln_f(hidden_states)
        lm_logits = model.lm_head(hidden_states)
    elif model.config.model_type == "opt":
        hidden_states = inps_.unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))
    return lm_logits


def get_layers(model):
    if model.config.model_type in (*LLAMA_LIKE, "phi3"):
        return model.model.layers
    elif model.config.model_type.lower() in FALCON_TYPES:
        return model.transformer.h
    elif model.config.model_type == "opt":
        return model.model.decoder.layers
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))


def find_sublayers(module, layers=(nn.Conv2d, nn.Linear)):
    res = {}
    for name, layer in module.named_modules():
        if isinstance(layer, layers):
            res[name] = layer
    return res


def get_sequential_groups(model):
    if model.config.model_type in LLAMA_LIKE:
        assert "mixtral" not in model.config.model_type.lower()  # check that this is not mixtral
        return [
            ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
            ["self_attn.o_proj"],
            ["mlp.up_proj", "mlp.gate_proj"],
            ["mlp.down_proj"],
        ]
    elif model.config.model_type.lower() in FALCON_TYPES:
        return [
            ["self_attention.query_key_value"],
            ["self_attention.dense"],
            ["mlp.dense_h_to_4h"],
            ["mlp.dense_4h_to_h"],
        ]
    elif model.config.model_type == "opt":
        return [
            ["self_attn.q_proj"],
            ["self_attn.k_proj"],
            ["self_attn.v_proj"],
            ["self_attn.out_proj"],
            ["fc1"],
            ["fc2"],
        ]
    elif model.config.model_type == "phi3":
        return [["self_attn.qkv_proj"], ["self_attn.o_proj"], ["mlp.gate_up_proj"], ["mlp.down_proj"]]
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))


def read_quant_weight_from_file(load_path, block_i, layer_name, device):
    return torch.load(load_path + "/" + str(block_i) + "/" + layer_name, map_location=device)


def load_linear_layers(layer, quant_layer, model):
    layer_ident = {}
    for submodule in layer.modules():
        for child_name, child_module in submodule.named_children():
            print(child_name, "child_name", layer_ident)
            if isinstance(child_module, (nn.Conv2d, nn.Linear)) or "norm" in child_name:
                if child_name in layer_ident:
                    layer_ident[child_name] += 1
                else:
                    layer_ident[child_name] = 1
                quant_count = 0
                print("Finding to dequantize ", child_name)
                for quant_submodule in quant_layer.modules():
                    for quant_child_name, quant_child_module in quant_submodule.named_children():
                        if quant_child_name == child_name:
                            quant_count += 1
                            if quant_count != layer_ident[child_name]:
                                continue
                            print(quant_child_name, quant_child_module)
                            if ("gate" in child_name.lower()) and ("mixtral" in model.config.model_type.lower()):
                                print("gate", child_name)
                                child_module.weight.data = quant_child_module.weight.data.to(
                                    child_module.weight.dtype
                                ).to(child_module.weight.device)
                                continue
                            if "norm" in child_name and not isinstance(child_module, (nn.Conv2d, nn.Linear)):
                                print("norm", child_name)
                                child_module.weight.data = quant_child_module.weight.data.to(
                                    child_module.weight.dtype
                                ).to(child_module.weight.device)
                            else:
                                print(child_name)
                                child_module.weight.data = (
                                    quant_child_module.quantized_weight()
                                    .data.to(child_module.weight.dtype)
                                    .to(child_module.weight.device)
                                )
                            # Bias is not taked into account
    return layer


def load_dequantized_model(model, load_path):
    """Load quantized model by dequantizing it"""
    layers = get_layers(model)
    for layer_index in range(len(layers)):
        print("layer", layer_index)
        layer = layers[layer_index]
        quant_layer = torch.load(os.path.join(load_path, str(layer_index) + ".pth"), map_location="cpu")
        for module in quant_layer.modules():
            if isinstance(module, QuantizedWeight):
                if not hasattr(module, "codes_storage"):
                    module.codes_storage = None  # backwards compatibility
        layers[layer_index] = load_linear_layers(layer, quant_layer, model)
    model.load_state_dict(torch.load(os.path.join(load_path, "not_quantized_weights.pt")), strict=False)
    return model


def load_quantized_model(model, load_path):
    """Load quantized model"""

    for layer_index in range(len(model.model.layers)):
        model.model.layers[layer_index] = torch.load(
            os.path.join(load_path, str(layer_index) + ".pth"),
            map_location=model.model.layers[layer_index].input_layernorm.weight.device,
        )
        for module in model.model.layers[layer_index].modules():
            if isinstance(module, QuantizedWeight):
                if not hasattr(module, "codes_storage"):
                    module.codes_storage = None  # backwards compatibility

    model.load_state_dict(torch.load(os.path.join(load_path, "not_quantized_weights.pt")), strict=False)
    return model


def save_not_quantized_weights(model: nn.Module, save_dir: str):
    already_saved_weights = set()
    for layer in get_layers(model):
        for param in layer.parameters():
            already_saved_weights.add(param)
    not_quantized_weights = {
        name: param for name, param in model.named_parameters() if param not in already_saved_weights
    }
    torch.save(not_quantized_weights, os.path.join(save_dir, "not_quantized_weights.pt"))


def save_quantized_model(model: transformers.PreTrainedModel, save_dir: str):
    """Save dequantized model state in the same format as returned by AQLM calibration (main.py)"""
    os.makedirs(save_dir, exist_ok=True)
    for layer_index, layer in enumerate(get_layers(model)):
        layer_save_path = os.path.join(save_dir, f"{layer_index}.pth")
        torch.save(layer, layer_save_path)
    save_not_quantized_weights(model, save_dir)


def get_layers_prefix(config: transformers.PretrainedConfig) -> str:
    if config.model_type in ("llama", "mistral", "mixtral", "gemma"):
        return "model.layers"
    raise NotImplementedError(f"Can't get layers prefix for {config.model_type}")
```

### `src/pv_optimizer.py`

```python
"""Module containing utilities for straight-through fine-tuning of language models"""
import random
from enum import Enum, auto
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed
import torch.nn as nn
from torch.optim.optimizer import StateDict

from src.aq import QuantizedWeight
from src.configurable_adam import ConfigurableAdamW
from src.pv_utils import YourQuantizedWeightIsInAnotherRank, print_runtime_stats


class ParameterRole(Enum):
    QUANTIZED_PARAMETER = auto()  # entire quantized weight, in a de-quantized form
    QUANTIZED_REPRESENTATION_PARAMETER = auto()  # part of quantized weight inner parameters, e.g. codebooks or scales
    NON_QUANTIZED_PARAMETER = auto()


class StraightThroughAdamW(ConfigurableAdamW):
    """
    A wrapper for a PyTorch optimizer that can perform updates on quantized and/or de-quantized parameters
    :param update_non_quantized_params: how to update parameters that are not directly linked to a QuantizedWeight.
        This may include biases, embeddings/heads, normalization layers or parts of the model that were not quantized.
        This should be either None (do not update) or a dictionary of optimizer kwargs. In the latter case, these
        keyword arguments will be used when configuring optimizer for this specific parameter group.
    :param update_codebooks_and_scales: how to update continuous params of QuantizedWeight: codebooks and scales.
        This should be either None (do not update) or a dictionary of optimizer kwargs. In the latter case, these
        keyword arguments will be used when configuring optimizer for this specific parameter group.
    :param update_codes: how to update codes in each QuantizedWeight with beam search and straight-through grad.
        This should be either None (do not update codes) or a dictionary of hyperparameter, similarly to above.
    :param delta_decay: determines whether to use straight-through estimation, direct optimization or a mixture thereof
        - if delta_decay == 1, do not use straight-through estimation. In this regime, the optimizer first updates
         de-quantized weights as though they were continuous, then uses modified weights to update codes, codebooks and
         scales; at the end of each step, the optimizer overwrites de-quantized weights to a de-quantization of the
         possibly updated quantized representations (codes, codebooks, scales).
        - if delta_decay == 0, use standard straight-through estimation. In this regime, the optimizer creates
        an internal set of straight-through buffers in the shape of de-quantized weights. The optimizer trains these
        buffers as though they were continuous; the quantized weights are then updated to minimize the L2 distance to
        these straight-through buffers; finally, the optimizer updates de-quantized weights from the quantized versions.
        - if delta_decay is between 0 and 1, use penalized straight-through estimation. The optimizer acts as though
        using standard straight-through estimation (see delta_decay == 0), but after every step, the straight-through
        buffers are set to (1 - delta_decay) * straight_through_buffer + delta_decay * quantized_weight.

    :param max_code_change_per_step: max portion of discrete code groups that can be updated; only affects codes
    :param code_trust_ratio: the maximum relative change to quantized weights per step, as a fraction of weight norm;
        see details in src/beam_search_l2.py, and in particular, beam_search_optimal_codes docstring.
    :param code_selection_temperature: if max_code_change or code_trust_ratio is set, the optimizer will by default
        prioritize updating codes with the largest delta = ||dequantized_weight_after_sgd_step - quantized_weight||_2 .
        If code_selection_temperature is above 0, it will instead sample codes randomly in proportion to the same
        delta ^ (1 / temperature). If temperature is very high, the optimizer will choose codes uniformly at random.
    :param force_code_update: if True, beam search will force codes to change even if code is optimal in
        terms of mean squared error. By default, the algorithm forces *all* weights to update this way, which may change
        weights too much. To limit the numer of updated weights, set max_code_change and trust_ratio.
    :param stochastic_rounding_tau: if above 0, use stochastic rounding with this temperature. See aq.py

    :param beam_size: beam search width used only when updating codes. See beam_size in aq.py

    :param straight_through_buffer_dtype: use this dtype when accumulating updates to de-quantized weight matrices
        Used only if delta_decay != 1.

    """

    def __init__(
        self,
        named_dequantized_params: Dict[str, nn.Parameter],
        named_quantized_params: Dict[str, Union[QuantizedWeight, YourQuantizedWeightIsInAnotherRank]],
        *,
        update_non_quantized_parameters: Optional[dict] = None,
        update_codebooks_and_scales: Optional[dict] = None,
        update_codes: Optional[dict] = None,
        beam_size: int,
        delta_decay: float = 1,
        max_code_change_per_step: float,
        code_trust_ratio: Optional[float] = None,
        code_selection_temperature: float = 0,
        force_code_update: bool = False,
        stochastic_rounding_tau: float = 0,
        straight_through_buffer_dtype: Optional[torch.dtype] = None,
        verbose: bool = False,
        **kwargs,
    ):
        assert 0 <= delta_decay <= 1
        assert all(
            isinstance(qw, (QuantizedWeight, YourQuantizedWeightIsInAnotherRank))
            for qw in named_quantized_params.values()
        )
        assert all(name in named_dequantized_params for name in named_quantized_params), "param names mismatch"

        self.sharded = not all(isinstance(qw, QuantizedWeight) for qw in named_quantized_params.values())
        self.is_straight_through = delta_decay != 1
        if verbose and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            print(end=f"PV optimizer init:\n\tAre quantized weights sharded? : {self.sharded}.\n")
            print(end=f"\tOptimizing {('without', 'with')[self.is_straight_through]} straight-through buffers\n")
        param_groups, all_optimized_params = self._select_optimized_parameters(
            named_dequantized_params=named_dequantized_params,
            named_quantized_params=named_quantized_params,
            update_non_quantized_parameters=update_non_quantized_parameters,
            update_codebooks_and_scales=update_codebooks_and_scales,
            update_codes=update_codes,
            straight_through_buffer_dtype=straight_through_buffer_dtype,
        )

        super().__init__(param_groups, **kwargs)
        self.ordered_quantized_weight_names = tuple(sorted(named_quantized_params.keys()))
        self.optimized_param_to_name = {param: name for name, param in all_optimized_params.items()}
        self.quantized_weights_by_name = {
            name: qw
            for name, qw in named_quantized_params.items()
            if isinstance(qw, (QuantizedWeight, YourQuantizedWeightIsInAnotherRank))
        }
        self.straight_through_buffer_by_name = (
            {
                name: all_optimized_params[name]
                for name in self.quantized_weights_by_name.keys()
                if name in all_optimized_params
            }
            if self.is_straight_through
            else {}
        )
        self.dequantized_weights_by_name = {
            name: param for name, param in named_dequantized_params.items() if name in named_quantized_params
        }
        if self.sharded:
            self.sharded_param_sizes_by_rank = _get_sharded_param_sizes_by_rank(named_dequantized_params)
            self.target_rank_by_name = {
                name: qw.rank if isinstance(qw, YourQuantizedWeightIsInAnotherRank) else torch.distributed.get_rank()
                for name, qw in self.quantized_weights_by_name.items()
            }

        self.should_update_non_quantized_parameters = update_non_quantized_parameters is not None
        self.should_update_codebooks_and_scales = update_codebooks_and_scales is not None
        self.should_update_codes = update_codes is not None

        self.delta_decay = delta_decay
        self.max_code_change_per_step = max_code_change_per_step
        self.code_trust_ratio = code_trust_ratio
        self.force_code_update = force_code_update
        self.code_selection_temperature = code_selection_temperature
        self.stochastic_rounding_tau = stochastic_rounding_tau
        self.beam_size = beam_size
        self.verbose = verbose

    def _select_optimized_parameters(
        self,
        named_dequantized_params,
        named_quantized_params,
        straight_through_buffer_dtype,
        update_non_quantized_parameters: Optional[dict],
        update_codebooks_and_scales: Optional[dict],
        update_codes: Optional[dict],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, nn.Parameter]]:
        """Choose which version of parameter to optimize: the parameter itself or a straight-through buffer"""
        non_quantized_params, quantized_params, quantized_representation_params = dict(), dict(), dict()
        for name, param in named_dequantized_params.items():
            if name not in named_quantized_params or isinstance(named_quantized_params[name], torch.Tensor):
                non_quantized_params[name] = param
            elif isinstance(named_quantized_params[name], QuantizedWeight):
                quantized_weight = named_quantized_params[name]
                if self.is_straight_through:  # create an accumulator for optimizer updates; sharded alongside FSDP
                    with torch.no_grad():
                        dequantized_weight = quantized_weight()
                    dequantized_weight = nn.Parameter(
                        dequantized_weight.to(dtype=straight_through_buffer_dtype),
                        requires_grad=dequantized_weight.requires_grad,
                    )
                else:
                    dequantized_weight = param
                quantized_params[name] = dequantized_weight
                for subparam_name, subparam in quantized_weight.named_parameters():
                    full_name = f"{name}.{subparam_name}"
                    assert full_name not in quantized_representation_params, full_name
                    quantized_representation_params[full_name] = subparam
            elif isinstance(named_quantized_params[name], YourQuantizedWeightIsInAnotherRank):
                assert self.sharded  # running sharded optimizer, this weight should be optimized by another rank
            else:
                raise RuntimeError(f"Unxpected quantized param type {type(named_quantized_params[name])}")

        total_params = len(set(non_quantized_params) | set(quantized_params) | set(quantized_representation_params))
        assert total_params == len(non_quantized_params) + len(quantized_params) + len(quantized_representation_params)
        param_groups = []
        all_optimized_params = dict()
        if update_non_quantized_parameters is not None:
            all_optimized_params.update(non_quantized_params)
            param_groups.append(
                dict(
                    params=list(non_quantized_params.values()),
                    role=ParameterRole.NON_QUANTIZED_PARAMETER,
                    **update_non_quantized_parameters,
                )
            )
        if update_codebooks_and_scales is not None:
            all_optimized_params.update(quantized_representation_params)
            param_groups.append(
                dict(
                    params=list(quantized_representation_params.values()),
                    role=ParameterRole.QUANTIZED_REPRESENTATION_PARAMETER,
                    **update_codebooks_and_scales,
                )
            )
        if update_codes is not None:
            all_optimized_params.update(quantized_params)
            param_groups.append(
                dict(params=list(quantized_params.values()), role=ParameterRole.QUANTIZED_PARAMETER, **update_codes)
            )
        assert len(param_groups) > 0, (
            "Please set at least one of update_codes, update_codebooks_and_scales " "or update_non_quantized_parameters"
        )
        return param_groups, all_optimized_params

    def step(self, *args, **kwargs):
        with print_runtime_stats("_propagate_grads_to_optimized_parameters", enabled=self.verbose):
            self._propagate_grads_to_optimized_parameters()
        with print_runtime_stats("super().step", enabled=self.verbose):
            original_output = super().step(*args, **kwargs)
        with print_runtime_stats("_optimize_quantized_weights", enabled=self.verbose):
            self._optimize_quantized_weights()
        with print_runtime_stats("_update_dequantized_weights", enabled=self.verbose):
            self._update_dequantized_weights()
        return original_output

    def _aggregate_gradients_for_dequantized_weights(self):
        """collect full parameter gradients from fsdp-sharded parameters, return dict[name -> grad]"""
        grad_shards_by_name = dict()

        for name in self.ordered_quantized_weight_names:
            if self.dequantized_weights_by_name[name].grad is None:
                assert self.dequantized_weights_by_name[name].numel() == 0
                self.dequantized_weights_by_name[name].grad = torch.zeros_like(self.dequantized_weights_by_name[name])
            grad = self.dequantized_weights_by_name[name].grad
            assert grad is not None, name
            grad_shards_by_name[name] = grad

        if self.sharded:
            aggregated_grads_by_name = _aggregate_tensors_by_name(
                grad_shards_by_name,
                self.sharded_param_sizes_by_rank,
                self.target_rank_by_name,
                name_order=self.ordered_quantized_weight_names,
            )
        else:
            aggregated_grads_by_name = grad_shards_by_name

        aggregated_grads_by_name = {
            name: grad.view(self.quantized_weights_by_name[name].shape)
            for name, grad in aggregated_grads_by_name.items()
        }
        if self.verbose:
            for name, grad in aggregated_grads_by_name.items():
                print(end=f"aggregated grad norm for {name}: {grad.norm().item()}\n")
        return aggregated_grads_by_name

    def _aggregate_dequantized_weights(self):
        """collect full (possibly optimizer-updated) dequantized weights"""
        if not self.sharded:
            return self.dequantized_weights_by_name
        dequantized_flat_param_shards = {
            name: param.data.flatten() for name, param in self.dequantized_weights_by_name.items()
        }
        flat_aggregated_params_by_name = _aggregate_tensors_by_name(
            dequantized_flat_param_shards,
            self.sharded_param_sizes_by_rank,
            self.target_rank_by_name,
            name_order=self.ordered_quantized_weight_names,
        )
        aggregated_params_by_name = {
            name: param.view(self.quantized_weights_by_name[name].shape)
            for name, param in flat_aggregated_params_by_name.items()
        }
        return aggregated_params_by_name

    @torch.no_grad()
    def _propagate_grads_to_optimized_parameters(self):
        """Ensure that every optimized parameter receives gradient"""
        aggregated_grads_by_name = self._aggregate_gradients_for_dequantized_weights()
        for param_group in self.param_groups:
            for param in param_group["params"]:
                name = self.optimized_param_to_name[param]
                if param_group["role"] == ParameterRole.QUANTIZED_PARAMETER:
                    if self.is_straight_through:
                        assert param is self.straight_through_buffer_by_name[name]
                        # pass gradients to straight-through update buffer or (possibly offloaded) quantized parameter
                        grad_wrt_dequantized_parameter = aggregated_grads_by_name[name]
                        assert grad_wrt_dequantized_parameter.shape == param.shape
                        param.grad = grad_wrt_dequantized_parameter.to(dtype=param.dtype, device=param.device)
                    else:
                        assert len(self.straight_through_buffer_by_name) == 0, self.straight_through_buffer_by_name
                        assert param.grad is not None
                elif param_group["role"] == ParameterRole.NON_QUANTIZED_PARAMETER:
                    assert name not in self.dequantized_weights_by_name and name not in self.quantized_weights_by_name
                elif param_group["role"] == ParameterRole.QUANTIZED_REPRESENTATION_PARAMETER:
                    assert name not in self.dequantized_weights_by_name
                    assert self.should_update_codebooks_and_scales
                    # gradients w.r.t quantized representation parameters are computed below via backprop
                else:
                    raise RuntimeError(f"Unexpected param role: {param_group['role']}")

        if self.should_update_codebooks_and_scales:
            # propagate gradients from dequantized weights to quantization parameters so they can be updated in step;
            # if sharded, every rank propagates gradients only for the QuantizedWeight instances owned by this rank
            with torch.enable_grad():
                for name, quantized_weight in self.quantized_weights_by_name.items():
                    if isinstance(quantized_weight, QuantizedWeight):
                        quantized_weight.forward().backward(aggregated_grads_by_name[name])

    @torch.no_grad()
    def _optimize_quantized_weights(self):
        """Update discrete state representations to approximate straight through buffers"""
        # note: if sharded, this only updates the subset of quantized weights that are assigned to local rank
        remaining_quantized_weights = {
            name: qw for name, qw in self.quantized_weights_by_name.items() if isinstance(qw, QuantizedWeight)
        }
        if self.is_straight_through:
            reference_weights_by_name = self.straight_through_buffer_by_name
        else:
            reference_weights_by_name = self._aggregate_dequantized_weights()

        for param_group in self.param_groups:
            if param_group["role"] == ParameterRole.QUANTIZED_PARAMETER:
                for param in param_group["params"]:
                    # param is either a dequantized weight or a special straight-through buffer (if is_straight_through)
                    name = self.optimized_param_to_name[param]
                    quantized_weight = remaining_quantized_weights.pop(name)
                    reference_weight = reference_weights_by_name[name]
                    assert reference_weight.shape == quantized_weight.shape, (
                        reference_weight.shape,
                        quantized_weight.shape,
                    )
                    assert isinstance(quantized_weight, QuantizedWeight)

                    prev_codes = quantized_weight.get_codes().clone()  # [num_output_groups, num_input_groups]
                    new_codes = quantized_weight.beam_search_update_codes_(
                        reference_weight=reference_weight,
                        beam_size=self.beam_size,
                        stochastic_rounding_tau=self.stochastic_rounding_tau,
                        max_update_fraction=self.max_code_change_per_step,
                        force_update=self.force_code_update,
                        code_selection_temperature=self.code_selection_temperature,
                        trust_ratio=self.code_trust_ratio,
                        dim_rng=random.Random(None),
                    )  # note: this updates quantized_weight codes in-place
                    if self.delta_decay != 0 and self.is_straight_through:
                        self.straight_through_buffer_by_name[name][...] = (
                            self.delta_decay * quantized_weight() + (1 - self.delta_decay) * reference_weight
                        )
                        # if not is_straight_throuh, param will be properly updated in _update_dequantized_weights

                    if self.verbose:
                        code_change_rate = torch.not_equal(prev_codes, new_codes).any(-1).float().mean().item()
                        maybe_distributed_msg = ""
                        if torch.distributed.is_initialized():
                            maybe_distributed_msg = f" (rank {torch.distributed.get_rank()})"
                        maybe_limit_msg = ""
                        if self.max_code_change_per_step is not None:
                            maybe_limit_msg = f"(limit {self.max_code_change_per_step})"
                        maybe_individual_msg = ""
                        if quantized_weight.num_codebooks > 1:
                            subcode_change = torch.not_equal(prev_codes, new_codes).float().mean().item()
                            maybe_individual_msg = f" | overall change {subcode_change:.8f}"
                        maybe_delta_msg = ""
                        if self.delta_decay != 1:
                            _dequantized_weight = quantized_weight()
                            delta_norm = (reference_weight - _dequantized_weight).norm().item()
                            relative_error = delta_norm / max(_dequantized_weight.norm().item(), 1e-9)
                            maybe_delta_msg = (
                                f"\t||quantized_weight - optimized_weight|| / ||quantized_weight||"
                                f" = {relative_error}\n"
                            )
                        print(
                            end=f"Updated codes for {name}{maybe_distributed_msg}:\n\tFraction of weights with at "
                            f"least one code change: {code_change_rate:.8f} "
                            f"{maybe_limit_msg}{maybe_individual_msg}\n{maybe_delta_msg}\n"
                        )
        assert len(remaining_quantized_weights) == 0

    @torch.no_grad()
    def _update_dequantized_weights(self):
        """Assign dequantized weight buffers to latest quantized weights after codebook/scale/code updates"""
        own_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        async_ops = list()
        for name in self.ordered_quantized_weight_names:
            quantized_weight = self.quantized_weights_by_name[name]
            dequantized_weight_buffer = self.dequantized_weights_by_name[name]
            dequantized_weight_buffer.fill_(float("nan"))  # this is to ensure that the update reaches the buffer

            if not self.sharded:
                dequantized_weight_buffer[...] = quantized_weight()

            else:
                if isinstance(quantized_weight, QuantizedWeight):
                    new_dequantized_weight = quantized_weight().to(dequantized_weight_buffer.dtype)
                    shard_sizes: Sequence[int] = self.sharded_param_sizes_by_rank[name]
                    assert sum(shard_sizes) == new_dequantized_weight.numel()
                    new_dequantized_weight_parts = new_dequantized_weight.flatten().split_with_sizes(shard_sizes)
                    for i in range(world_size):
                        if i != own_rank:
                            async_ops.append(torch.distributed.isend(new_dequantized_weight_parts[i], dst=i))
                        else:
                            dequantized_weight_buffer.copy_(new_dequantized_weight_parts[i])

                else:
                    assert isinstance(quantized_weight, YourQuantizedWeightIsInAnotherRank)
                    source_rank = self.quantized_weights_by_name[name].rank
                    async_ops.append(torch.distributed.irecv(dequantized_weight_buffer, src=source_rank))
        for handle in async_ops:
            handle.wait()

    def zero_grad(self, set_to_none: bool = True, *args, **kwargs) -> None:
        super().zero_grad(set_to_none=set_to_none, *args, **kwargs)
        for param in self.dequantized_weights_by_name.values():
            # dequantized weights are not in param_groups, but they still accumulate grads; reset them manually
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()

    def iterate_local_quantized_weights(self) -> Iterator[Tuple[str, QuantizedWeight]]:
        """Iterate over (name, QuantizedWeight) pairs for all quantized weights trained by this optimizer and rank"""
        for name, quantized_weight in self.quantized_weights_by_name.items():
            if isinstance(quantized_weight, QuantizedWeight):  # skip YourQuantizedWeightIsInAnotherRank if sharded
                yield name, quantized_weight

    def state_dict(self) -> StateDict:
        state_dict = super().state_dict()
        assert "quantized_weight_state_dicts" not in state_dict
        state_dict["quantized_weight_state_dicts"] = {
            name: quantized_weight.state_dict() for name, quantized_weight in self.iterate_local_quantized_weights()
        }
        state_dict["straight_through_buffers"] = dict(self.straight_through_buffer_by_name)  # may be empty
        # note: the de-quantized params are not saved here; instead, they are saved with model.state_dict
        return state_dict

    def load_state_dict(self, state_dict: StateDict) -> None:
        quantized_weight_state_dicts: Dict[str, StateDict] = dict(state_dict.pop("quantized_weight_state_dicts"))
        for name, quantized_weight in self.iterate_local_quantized_weights():
            quantized_weight.load_state_dict(quantized_weight_state_dicts.pop(name))
        assert len(quantized_weight_state_dicts) == 0, f"unused keys: {quantized_weight_state_dicts.keys()}"

        straight_through_buffers = state_dict.pop("straight_through_buffers")
        assert all(name in straight_through_buffers for name in self.straight_through_buffer_by_name)
        for name, loaded_values in straight_through_buffers.items():
            self.straight_through_buffer_by_name[name][...] = loaded_values
        super().load_state_dict(state_dict)


def _get_sharded_param_sizes_by_rank(named_dequantized_params: Dict[str, torch.Tensor]) -> Dict[str, Sequence[int]]:
    """For each parameter name, return a tuple of sizes (numbers of elements) this parameter across all FSDP ranks"""
    assert torch.distributed.is_initialized()
    own_dequantized_param_shard_size = {name: param.numel() for name, param in named_dequantized_params.items()}
    world_size = torch.distributed.get_world_size()
    gathered_list = [{} for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered_list, own_dequantized_param_shard_size)
    assert all(name in sizes_dict for sizes_dict in gathered_list for name in own_dequantized_param_shard_size)
    dequantized_param_sizes_by_rank = dict()
    for name in named_dequantized_params.keys():
        dequantized_param_sizes_by_rank[name] = [gathered_list[rank][name] for rank in range(world_size)]
    return dequantized_param_sizes_by_rank


def _aggregate_tensors_by_name(
    sharded_tensors_by_name: Dict[str, torch.Tensor],
    shard_sizes_by_name: Dict[str, Sequence[int]],
    target_rank_by_name: Dict[str, int],
    name_order: Optional[Sequence[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    :param sharded_tensors_by_name: a dictionary from string to flat (1d) tensors available on the current shard
    :note: the keys should be the same across ranks and go in the same order; if not, use ordered_names
    :param shard_sizes_by_name: a dictionary from name to a list of sizes (numel) for this key across ranks
    :param target_rank_by_name: a dictionary from name to a rank that this name should be aggregated to
    :param name_order: if specified, this defines the order in which devices go over named shards
    """
    assert torch.distributed.is_initialized()
    own_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    aggregated_tensors_by_name = dict()
    async_ops = list()

    for name in sorted(sharded_tensors_by_name.keys()) if name_order is None else name_order:
        shard = sharded_tensors_by_name[name]
        assert shard.ndim == 1
        destination_rank = target_rank_by_name[name]
        shard_sizes: Sequence[int] = shard_sizes_by_name[name]
        if destination_rank == own_rank:
            total_numel = sum(shard_sizes)
            combined_buffer = torch.full((total_numel,), fill_value=torch.nan, dtype=shard.dtype, device=shard.device)
            gather_buffers = list(combined_buffer.split_with_sizes(shard_sizes))
            assert all(
                part.untyped_storage().data_ptr() == combined_buffer.untyped_storage().data_ptr()
                for part in gather_buffers
            )
            for i in range(world_size):
                if shard_sizes[i] == 0:
                    continue  # optimization: this handles FSDP where some param/grad shards are empty
                elif i != own_rank:
                    async_ops.append(torch.distributed.irecv(gather_buffers[i], src=i))
                else:
                    gather_buffers[i].copy_(shard)
            aggregated_tensors_by_name[name] = combined_buffer
        else:
            if shard_sizes[own_rank] == 0:
                continue
            async_ops.append(torch.distributed.isend(shard, destination_rank))

    for handle in async_ops:
        handle.wait()
    return aggregated_tensors_by_name
```

### `src/pv_utils.py`

```python
import contextlib
import dataclasses
import hashlib
import json
import time
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from typing import Dict, List, Optional, Tuple

import torch
import transformers
from torch import nn as nn

from src.aq import QuantizedLinear, QuantizedWeight


def infer_module_classes(model: nn.Module, class_name: str) -> Tuple[type[nn.Module], ...]:
    """find transformer block classes that should be wrapped with inner FullyShardedDataParallel (auto_wrap_policy)"""
    found_module_types = []
    for module in model.modules():
        if module.__class__.__name__ == class_name:
            found_module_types.append(type(module))
    if not found_module_types:
        raise ValueError(f"Could not find {class_name} among submodules of {model}")
    found_module_types = tuple(found_module_types)
    assert any(isinstance(module, found_module_types) for module in model.modules())
    return found_module_types


def create_dequantized_model(
    model: transformers.PreTrainedModel, *, reuse_non_quantized: bool, dequantized_dtype: Optional[torch.dtype] = None
) -> transformers.PreTrainedModel:
    """
    Create a version of the model where all QuanizedWeight and derivative layers are de-quantized and cast to dtype.
    :param model: model to be dequantized (out-of-place)
    :param reuse_non_quantized: if True, any non-quantized parameters and buffers are reused for de-quantized model;
        otherwise (default) they are copied and linked in the returned dictionary
    :returns: a model (converted out-of-place) and a mapping (dict) from de-quantized to master parameters
    """
    memo = dict()  # for deepcopy with replacement
    master_parameters = dict()
    all_quantized_weight_parameters = set()

    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            assert module not in master_parameters and id(module) not in memo, f"{name} is converted more than once"
            quantized_weight = module.quantized_weight

            dequantized_module = nn.Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                dtype=dequantized_dtype if dequantized_dtype is not None else quantized_weight.get_codebooks().dtype,
                device=next(quantized_weight.parameters()).device,
            )
            with torch.no_grad():
                dequantized_module.weight[...] = quantized_weight()
                dequantized_module.weight.requires_grad = any(p.requires_grad for p in quantized_weight.parameters())

                if module.bias is not None and not reuse_non_quantized:
                    dequantized_module.bias[...] = module.bias
                    dequantized_module.bias.requires_grad = dequantized_module.bias.requires_grad
                elif module.bias is not None and reuse_non_quantized:
                    dequantized_module.bias = module.bias

            memo[id(module)] = dequantized_module
            master_parameters[f"{name}.weight"] = quantized_weight
            if dequantized_module.bias is not module.bias:
                master_parameters[f"{name}.bias"] = module.bias
            all_quantized_weight_parameters |= set(quantized_weight.parameters())
            assert all(
                param in {dequantized_module.weight, dequantized_module.bias}
                for param in dequantized_module.parameters()
            )

    for name, param_or_buffer in chain(model.named_parameters(), model.named_buffers()):
        if name in master_parameters or param_or_buffer in all_quantized_weight_parameters:
            continue  # parameter already accounted for in the previous loop
        assert name not in master_parameters, name
        assert id(param_or_buffer) not in memo, name
        if reuse_non_quantized:
            new_param_or_buffer = param_or_buffer
        elif isinstance(param_or_buffer, nn.Parameter):
            new_param_or_buffer = nn.Parameter(param_or_buffer.data.clone(), param_or_buffer.requires_grad)
        else:
            new_param_or_buffer = param_or_buffer.detach().clone().requires_grad_(param_or_buffer.requires_grad)
        if new_param_or_buffer is not param_or_buffer:
            master_parameters[name] = new_param_or_buffer
        memo[id(param_or_buffer)] = new_param_or_buffer

    dequantized_model = deepcopy(model, memo=memo)

    for name, module in dequantized_model.named_modules():
        assert not isinstance(module, QuantizedWeight), (
            f"Dequantized model should not have quantized weights, " f"but found {name} that is {module}"
        )
    if reuse_non_quantized:
        assert all(isinstance(master, QuantizedWeight) for master in master_parameters.values())
    verify_dequantized_model(dequantized_model, master_parameters)
    return dequantized_model, master_parameters


def verify_dequantized_model(dequantized_model: nn.Module, master_parameters: dict):
    """Test that the dequantized model parameters still match the dequantized_to_master dictionary"""
    unmatched_master_parameters = set(master_parameters.keys())
    for name, param_or_buffer in chain(dequantized_model.named_parameters(), dequantized_model.named_buffers()):
        if name not in master_parameters:
            continue  # non-quantized weight
        master_param_or_buffer = master_parameters[name]
        assert param_or_buffer.shape == master_param_or_buffer.shape
        unmatched_master_parameters.remove(name)
    assert len(unmatched_master_parameters) == 0, f"Found unmatched tensors: {unmatched_master_parameters}"


def get_original_named_parameters_from_fsdp_module(dequantized_model) -> Dict[str, nn.Parameter]:
    return {name.replace("_fsdp_wrapped_module.", ""): param for name, param in dequantized_model.named_parameters()}


@contextlib.contextmanager
def print_runtime_stats(operation_name: str, enabled: bool = True, device: Optional[torch.device] = None):
    if not enabled:
        yield
        return

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if device is None:
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if torch.device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    yield
    if torch.device.type == "cuda":
        torch.cuda.synchronize(device)
    maybe_distributed_msg = f"rank {rank} " if torch.distributed.is_initialized() else ""
    print(end=f"{maybe_distributed_msg}{operation_name} took {time.perf_counter() - start_time}\n")


def split_quantized_weights_between_ranks(quantized_weights: Dict[str, QuantizedWeight], verify_checksums: bool):
    """
    Split all quantized weights between ranks in a distributed setup; uses greedy knapsack heuristic.
    Note that unlike FSDP, this heuristic will always assign the entire quantized weight to one rank.

    :param quantized_weights: a dictionary [parameter_name] -> QuantizedWeight
    :returns: a dictionary similar to quantized weights or pointers to different ranks.
        If your rank stores this quantized weight for [name], then returned_dict[name] is quantized_weights[name]
        Otherwise, returned_dict[name] = YourQuantizedWeightIsInAnotherRank(rank=where_it_is_stored)
    :param verify_checksums: if True, synchronize with other ranks and verify that parameters are split consistently.
        If False, do not synchronize, but instead print a hash of checksum for each rank to be verified by the user.
    """
    assert torch.distributed.is_initialized()
    own_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    all_quantized_weights: Dict[QuantizedWeight, List[str]] = defaultdict(list)
    for name, quantized_weight in quantized_weights.items():
        all_quantized_weights[quantized_weight].append(name)

    # order quantized weights in a rank-agnostic way: order by (param size desc, linked param name asc)
    def _compute_size(qw: QuantizedWeight) -> float:
        return qw.out_features * qw.in_features * qw.estimate_nbits_per_parameter()

    ordered_quantized_weights = sorted(
        all_quantized_weights, key=lambda qw: (-_compute_size(qw), min(all_quantized_weights[qw]))
    )
    assert len(ordered_quantized_weights) > 0, "internal error: could not find any linked QuantizedWeight in state"

    # split between ranks
    quantized_weight_to_rank = dict()
    total_size_by_rank = [0 for _ in range(world_size)]
    for quantized_weight in ordered_quantized_weights:
        least_busy_rank = min(range(world_size), key=lambda rank: total_size_by_rank[rank])
        total_size_by_rank[least_busy_rank] += _compute_size(quantized_weight)
        quantized_weight_to_rank[quantized_weight] = least_busy_rank

    checksum = tuple(
        (min(all_quantized_weights[qw]), quantized_weight_to_rank[qw], _compute_size(qw))
        for qw in ordered_quantized_weights
    )
    if verify_checksums:
        checksums = [() for _ in range(world_size)]
        torch.distributed.all_gather_object(checksums, checksum)
        assert checksums[own_rank] == checksum, (checksums, own_rank, checksum)
        assert all(other_checksum == checksum for other_checksum in checksums), checksums
    else:
        hashing = hashlib.sha256()
        hashing.update(json.dumps(checksum).encode())
        print(end=f"Splitting quantized weights, rank {own_rank} checksum hash: {hashing.hexdigest()}\n")

    sharded_quantized_weights = dict()
    for name, quantized_weight in list(quantized_weights.items()):
        target_rank = quantized_weight_to_rank[quantized_weight]
        if target_rank == own_rank:
            sharded_quantized_weights[name] = quantized_weight
        else:
            sharded_quantized_weights[name] = YourQuantizedWeightIsInAnotherRank(target_rank)
    return sharded_quantized_weights


@dataclasses.dataclass(init=True, frozen=True)
class YourQuantizedWeightIsInAnotherRank:
    """This replaces quantized weights that are not held on this rank"""

    rank: int
```

### `src/utils.py`

```python
"""Common utility functions for additive quantization"""
from __future__ import annotations

import contextlib
import functools
import os
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence, Union

import torch
import torch.distributed
from torch import nn
from torch.nn import functional as F

ellipsis = type(...)


def get_mean_nbits_by_codebook(codes: torch.IntTensor, huffman_group_size: int = 2):
    """
    Calculates average code length in codebooks.
    :param codes: codebook codes
    :param huffman_group_size: huffman compresssion dimension count
    """
    import huffman

    _, codebook_size, num_codebooks = codes.shape
    flat_codes_by_codebook = codes.permute(2, 0, 1).flatten(1, 2)
    code_counts = torch.zeros(
        num_codebooks, codebook_size, device=flat_codes_by_codebook.device, dtype=flat_codes_by_codebook.dtype
    ).scatter_add(
        -1, flat_codes_by_codebook, torch.ones_like(flat_codes_by_codebook)
    )  # shape: [current beam_size, num_codebooks, codebook_size], initial beam_size = 1
    code_probs = code_counts / code_counts.sum(dim=-1, keepdim=True).float()
    code_probs = code_probs.cpu().numpy()
    assert num_codebooks % huffman_group_size == 0

    mean_code_lengths = []
    for group_index in range(num_codebooks // huffman_group_size):
        group_code_probs = {(): 1}

        for codebook_index in range(group_index * huffman_group_size, (group_index + 1) * huffman_group_size):
            new_group_code_probs = {}
            for group, group_prob in group_code_probs.items():
                for code, code_prob in tuple(enumerate(code_probs[codebook_index])):
                    new_group_code_probs[group + (code,)] = group_prob * code_prob
            group_code_probs = new_group_code_probs

        huffman_codebook_i = huffman.codebook(list(group_code_probs.items()))
        codebook_mean_code_length_i = sum(
            len(huffman_codebook_i[code]) * prob for code, prob in group_code_probs.items()
        )
        mean_code_lengths.append(codebook_mean_code_length_i)
    return mean_code_lengths


@functools.lru_cache()
def maybe_script(fn: callable) -> callable:
    """Apply torch.jit.script to function unless one is using TPU. TPU does not support torch.jit.script."""
    using_tpu = bool(os.environ.get("TPU_NAME"))
    # this is a reserved variable that must be set to TPU address (e.g. grpc://11.22.33.44:1337) for TPU to function
    should_script = int(os.environ.get("AQ_USE_JIT", not using_tpu))
    return torch.jit.script(fn) if should_script else fn


@maybe_script
def _dequantize_weight(
    codes: torch.Tensor, codebooks: torch.Tensor, scales: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Decode float weights from quantization codes. Differentiable.
    :param codes: tensor of integer quantization codes, shape [*dims, num_out_groups, num_in_groups, num_codebooks]
    :param codebooks: tensor of vectors for each quantization code, [num_codebooks, codebook_size, out_group_size, in_group_size]
    :param scales: weight will be multiplied by this factor, must be broadcastble with [*dims, out_groups, num_in_groups, out_group_size, in_group_size]
    :return: reconstructed weight tensor of shape [*dims, num_in_groups*group_size]
    """
    num_out_groups, num_in_groups, num_codebooks = codes.shape[-3:]
    num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape
    out_features = num_out_groups * out_group_size
    in_features = num_in_groups * in_group_size
    codebook_offsets = torch.arange(
        0, num_codebooks * codebook_size, codebook_size, device=codes.device
    )  # shape: [num_codebooks]
    reconstructed_weight_flat = F.embedding_bag(
        codes.flatten(0, -2) + codebook_offsets, codebooks.flatten(0, 1).flatten(-2, -1), mode="sum"
    )  # [prod(dims) * num_out_groups * num_in_groups, out_group_size * in_group_size]

    reconstructed_weight_groupwise = reconstructed_weight_flat.view(
        list(codes.shape[:-3]) + [num_out_groups, num_in_groups, out_group_size, in_group_size]
    )
    if scales is not None:
        reconstructed_weight_groupwise = reconstructed_weight_groupwise.mul(scales)
    return reconstructed_weight_groupwise.swapaxes(-3, -2).reshape(list(codes.shape[:-3]) + [out_features, in_features])


@contextlib.contextmanager
def using_tf32(enabled: bool):
    was_cudnn = torch.backends.cudnn.allow_tf32
    was_matmul = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = enabled
    torch.backends.cuda.matmul.allow_tf32 = enabled
    yield
    torch.backends.cudnn.allow_tf32 = was_cudnn
    torch.backends.cuda.matmul.allow_tf32 = was_matmul


def iterate_minibatches(
    *tensors: torch.Tensor,
    batch_size: int,
    allow_incomplete: bool = True,
    device: Optional[torch.device] = None,
    callback: Callable[[Sequence[torch.Tensor]], Sequence[torch.Tensor]] = lambda x: x,
) -> Iterator[Sequence[torch.Tensor]]:
    """
    Samples data points *forever*, in random order, with less overhead than DataLoader;
    Adapted from https://github.com/stanis-morozov/unq/blob/master/lib/utils.py
    probably implemented over9000 times in transformers, torch, etc
    :param tensors: one or more tensors with the same 0-th dimension
    :param batch_size: sample this many points with each yield
    :param allow_incomplete: if True and if dataset size is not divisible by batch size, the last batch
        may have less than :batch_size: samples to cover the entire dataset. If False, the last batch is dropped
    :param callback: optional function to be called on each batch of tensors before it is yielded to the user
    :returns: generates a tuple of minibatches from each tensor, same length as input *tensors
        If a batch contains only one tensor, this function will yield a tensor (and not a tuple/list with one tensor)
    """
    num_samples = len(tensors[0])
    assert all(len(x) == num_samples for x in tensors)
    indices = torch.randperm(num_samples, device=tensors[0].device)
    while True:
        prev_batch = None
        for batch_start in range(0, len(indices), batch_size):
            if not allow_incomplete and batch_start + batch_size > len(indices):
                break
            batch_ix = indices[batch_start : batch_start + batch_size]
            batch = callback(tuple(tensor[batch_ix].to(device, non_blocking=True) for tensor in tensors))
            if prev_batch is not None:
                yield prev_batch
            prev_batch = batch if isinstance(batch, (list, tuple)) and len(tensors) > 1 else batch[0]
            del batch
        yield prev_batch


def maybe_get_0th_element(x: Union[Any, Sequence[Any]]) -> Any:
    """
    Return first element if input is Sequence, otherwise return input
    """
    if isinstance(x, Sequence):
        return x[0]
    return x


def _extract_into_tensor(tensor_list: List[torch.Tensor], indices: Iterable[int], device=None, dtype=None):
    extracted_items = [maybe_get_0th_element(tensor_list[i]) for i in indices]
    return torch.cat(extracted_items, dim=0).to(device=device, dtype=dtype)


class IntCodes(nn.Module):
    """
    A storage for integer codes that makes them compatible with FullyShardedDataParallel,
    see https://github.com/pytorch/pytorch/issues/123528 for details
    """

    def __init__(self, codes: torch.tensor, storage_dtype: torch.dtype = torch.float64):
        super().__init__()
        assert torch.finfo(storage_dtype).bits % torch.iinfo(codes.dtype).bits == 0
        self.dtype, self.shape, self.numel = codes.dtype, codes.shape, codes.numel()
        size_ratio = torch.finfo(storage_dtype).bits // torch.iinfo(codes.dtype).bits
        codes = F.pad(codes.flatten().clone(), pad=[0, -codes.numel() % size_ratio])
        assert len(codes.untyped_storage()) == codes.nbytes  # no offset / stride / tail
        self.storage_dtype = storage_dtype
        self.data = nn.Parameter(
            torch.as_tensor(codes.untyped_storage(), device=codes.device, dtype=storage_dtype), requires_grad=False
        )

    def forward(self):
        assert self.data.is_contiguous() and self.data.dtype == self.storage_dtype
        byte_offset = self.data.storage_offset() * self.data.nbytes // self.data.numel()
        return torch.as_tensor(
            self.data.untyped_storage()[byte_offset : byte_offset + self.data.nbytes],
            device=self.data.device,
            dtype=self.dtype,
        )[: self.numel].view(*self.shape)


@contextlib.contextmanager
def one_rank_at_a_time(local: bool = False, group_size: int = 1):
    """
    In distributed setting, let only group_size processes enter at a time
    :param local: if True, the limit is enforced within each host, i.e. distributed hosts can act concurrently
    :param group_size: if more than one is specified,
    """
    distributed = torch.distributed.is_initialized()
    rank = int(os.environ.get("LOCAL_RANK" if local else "RANK", 0)) if distributed else 0
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE" if local else "WORLD_SIZE", 0)) if distributed else 1
    if distributed:
        torch.distributed.barrier()
    for current_group_index in range(world_size // group_size):
        if current_group_index == rank // group_size:
            yield
        if distributed:
            torch.distributed.barrier()


@contextlib.contextmanager
def master_rank_first(local: bool, master_rank: int = 0):
    distributed = torch.distributed.is_initialized()
    rank = int(os.environ.get("LOCAL_RANK" if local else "RANK", 0)) if distributed else 0
    if distributed and rank != master_rank:
        torch.distributed.barrier()
    yield
    if distributed and rank == master_rank:
        torch.distributed.barrier()


def is_signed(dtype: torch.dtype) -> bool:
    """Return True iff an integer dtype is signed"""
    try:
        return dtype.is_signed
    except RuntimeError:  # see https://github.com/pytorch/pytorch/issues/125124
        if dtype.is_floating_point:
            return torch.finfo(dtype).min < 0
        else:
            return torch.iinfo(dtype).min < 0 and torch.iinfo(dtype).max > 0
```


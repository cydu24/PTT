# Think Thrice Before You Act: Progressive Thought Refinement in Large Language Models

**Recent advancements in large language models (LLMs) have demonstrated that progressively refining responses, rather than providing a single answer, leads to more accurate and thoughtful outcomes. However, current methods for progressive refinement often rely heavily on supervision signals or task-specific fine-tuning, which restricts their ability to generalize and depends on costly human-annotated labels.**

**To address these challenges, we introduce PTR (Progressive Thought Refinement), a framework designed to enable LLMs to refine their responses progressively without the need for additional supervision signals or task-specific fine-tuning. PTR operates through two main phases:**

1. [ ] **Thought Data Construction Phase**: We utilize queries from open-domain datasets and generate thoughts and answers using a collaborative weak-strong model selection process, eliminating the need for manual labeling. Additionally, filtering strategies are applied to ensure logical consistency in the generated thoughts.
2. [ ] **Weighted Thought-Mask Fine-Tuning Phase**: We redefine the fine-tuning objective to enhance the model's progressive refinement capabilities. The model is trained to iteratively improve its responses, starting from initial thoughts and advancing to more refined answers. We incorporate a weighted loss function to promote consistent improvement, maintain logical coherence, and build confidence.

**Experimental results show that our method significantly enhances LLM performance across ten diverse tasks, all without task-specific fine-tuning or additional supervision signals.**

<img src='utils/main_fig.png' alt=''>

**The source code and datasets for our project are available at: **[https://anonymous.4open.science/r/PTR/](https://anonymous.4open.science/r/PTR/).

## Requirements

**To run this project, you will need the following packages:**

```
 numpy==1.26.3
 tokenizers==0.15.0
 torch==2.1.2
 tqdm==4.66.1
 transformers==4.39.3
 vllm==0.3.2
```

## PTR Preparation

Preparation Script

**Below is a script that demonstrates how to download models and prepare data for PTR:**

```
 declare -a model_paths=(
 
 )
 
 declare -a model_names=(
 
 )
 
 for i in "${!model_paths[@]}"; do
     echo "Downloading ${model_names[$i]}..."
     python PTR_preparation/thought_generation_vllm.py \
     --model_path "${model_paths[$i]}" \
     --model_name "${model_names[$i]}" \
     --max_new_tokens 2048 \
     --prompt_paths  \
     --temperature 0.95 \
     --top_p 0.9 \
     --top_k 30 \
     --use_vllm \
     --data_len 
 done
 
 echo "All models have been downloaded successfully."
```

## PTR Fine-Tuning

### Encoder

**You can run the encoder using the following script:**

```
 cd /path/to/PTR_finetune
 python encoder/main.py \
     --tokenizer_path \
     --data_loader_path  \
     --corpus_name  \
     --encode_type qa \
     --output_path  \
     --max_length 4096 \
     --merge_data \
     --save_dtype int32 \
```

#### Explanation of Parameters:

* `--tokenizer_path`: Absolute path to the tokenizer
* `--data_loader_path`: Absolute path to your custom data_loader.py
* `--corpus_name`: Name of the encoded corpus, used for output folder naming
* `--encode_type`: Either "qa" or "pretrain"
* `--output_path`: Absolute path to the output
* `--max_length`: Maximum length of each piece of data
* `--merge_data`: Whether to merge QA data
* `--save_dtype`: Data type of the saved file (either `int32` or `int16`)
* `--tokens_per_file`: Specifies the number of valid tokens per file (default: 5e8)

### Trainer

**You can train the model using:**

```
 cd /path/to/PTR_finetune
 
 deepspeed --include localhost:0 --master_port 12345 trainer/train.py \
     --model_path /path/to/your/model \
     --data_path /path/to/your/data \
     --save_name <your_model_name> \
     --output_path /path/to/your/output/path \
     --deepspeed_config_path /path/to/your/deepspeed_config.json \
     --max_epochs 3 \
     --save_epochs 1 \
     --save_steps 1000 \
```

#### Explanation of Training Parameters:

* `--model_path`: Path to the base model
* `--max_epochs`: Maximum number of epochs
* `--max_steps`: Maximum number of training steps
* `--load_ckpt_path`: Path to a checkpoint, if any
* `--data_path`: Root directory of the data path
* `--deepspeed_config_path`: Absolute path to `ds_config.json`
* `--output_path`: Path for saving outputs
* `--save_name`: Folder name for saving checkpoints
* `--save_steps`: Save checkpoints every N steps
* `--save_epochs`: Save checkpoints every N epochs
* `--save_optimizer`: Include optimizer state in saved checkpoints
* `--use_lora`: Use LoRA (Low-Rank Adaptation)
* `--lora_config_path`: Absolute path to `lora_config.json`

## Running Evaluations

**This lightweight evaluation framework supports several commonly used benchmarks. It is highly extensible and can accommodate new tasks. The framework employs the **`transformers` and `vllm` libraries for inference.

### Getting Started

**Install required packages:**

```
 pip install -r requirements.txt
```

**Configure the **`run.sh` script:

* **Model configuration: **`model_path`, `model_type`, `format_type`
* **Task configuration: **`tasks`, `config_path`, `data_path`
* **Output configuration: **`output_path`, `save_name`, etc.
* **Inference configuration: **`rounds`, `seed`, `temperature`, `top_p`, `top_k`, `max_new_tokens`

### Example: Evaluating Two Models on MMLU and GSM8K

```
 export CUDA_VISIBLE_DEVICES=3
 export TOKENIZERS_PARALLELISM=false
 
 declare -A models=(
    ["model_name1"]="model_path1"
 ["model_name2"]="model_path2"
 )
 
 for model_name in "${!models[@]}"; do
     model_path=${models[$model_name]}
     python main.py \
         --model_path "$model_path" \
         --model_type vllm \
         --format_type default \
         --tasks mmlu gsm8k \
         --save_name "$model_name" \
         --save_infer_texts \
         --save_infer_results \
         --config_path "config.json" \
         --output_path output/debug \
         --max_new_tokens 180 \
         --temperature 0 \
         --top_p 0.2 \
         --top_k 20 \
 done
```

### Configuring Evaluation Tasks

**The default **`config.json` file provides task configurations. An example configuration looks like:

```
 {
     "gsm8k": {
         "num_fewshots": 8,
         "limit": 0
     },
     "mmlu": {
         "num_fewshots": 5,
         "limit": null
     }
 }
```

**If a configuration entry is missing, default settings for that task will be used.**

## Supported Evaluation Tasks

### ARC

* **Source**:[opencompass](https://github.com/open-compass/opencompass)
* **Metric**: Matches the first option in the answer

### CommonsenseQA

* **Source**:[opencompass](https://github.com/open-compass/opencompass)
* **Metric**: Matches the first option in the answer

### DROP

* **Source**: [opencompass](https://github.com/open-compass/opencompass)
* **Metric**: Uses regex to match the answer after 'answer is'

### GPQA

* **Source**:Huggingface[https://github.com/open-compass/opencompass](https://github.com/open-compass/opencompass))
* **Metric**: Matches the first option in the answer

### GSM8K

* **Source**: Huggingface, few-shot prompts from [cot-hub](https://github.com/FranxYao/chain-of-thought-hub)
* **Metrics**: Exact Match (accuracy), Flexible Match

### MATH

* **Source**: Huggingface
* **Metrics**: Exact Match (accuracy), Flexible Match

### HellaSwag

* **Source**: Huggingface
* **Metric**: Matches the first option

### HumanEval

* **Source**: Huggingface
* **Metric**: Evaluates code correctness using the HumanEval library

### MMLU

* **Source**: [opencompass](https://github.com/open-compass/opencompass)
* **Metric**: Matches the first option

### WinoGrande

* **Source**: [opencompass ](https://github.com/open-compass/opencompass)[cot-hub](https://github.com/FranxYao/chain-of-thought-hub)
* **Metric**: Matches the first option

### XSum

* **Source**:[EdinburghNLP/XSum(github.com)](https://github.com/EdinburghNLP/XSum)
* **Metric**: Calculates similarity with BAAI/bge-m3 model

**This guide provides a comprehensive view of using and evaluating the PTR framework with various tasks. Feel free to reach out for any further assistance or inquiries!**

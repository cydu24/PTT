from vllm import LLM, SamplingParams
import torch

vllm_initialized = False

def vllm_initialize(args):
    global vllm_initialized
    global sampling_params
    global model
    model = LLM(
        model=args.model_path, 
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    sampling_params = SamplingParams(
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        stop=['<|im_end|>','<|endoftext|>','<|eot_id|>', '<|/s|>'],
        temperature=args.temperature,
    )
    vllm_initialized = True

def vllm_generate(prompts, args, generation_name=""):
    global vllm_initialized
    global model
    global sampling_params
    if not vllm_initialized:
        vllm_initialize(args)
    generated_texts = []
    prompts_list = []
    prompts_per_step = 1000
    for i in range(0, len(prompts), prompts_per_step):
        prompts_list.append(prompts[i:i+prompts_per_step])

    for i, inputs in enumerate(prompts_list):
        print(f"generating {generation_name} [{i * prompts_per_step + 1}, {len(prompts)}] in {len(prompts)} prompts")
        outputs = model.generate(inputs, sampling_params)
        for i in range(len(outputs)):
            output = outputs[i]
            generated_text = output.outputs[0].text
            generated_texts.append(generated_text.strip())
    return generated_texts


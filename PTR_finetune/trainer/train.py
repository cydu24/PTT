import argparse, os, json, random, datetime
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import deepspeed
from peft import get_peft_model, PeftModel, LoraConfig

from dataset import TorchMultiFileBinaryDataset
from draw_loss import draw_loss


def print0(*args, **kwargs):
    if torch.distributed.get_rank() == 0:
        print(*args, **kwargs)


def setup_distributed_environment(local_rank):
    if local_rank != -1:  
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=local_rank,
            world_size=torch.cuda.device_count(),
        )
    else:  
        device = torch.device("cuda")
    deepspeed.init_distributed()
    print0("\n" + "=" * 20 + "\nDistributed environment is initialized.\n" + "=" * 20)
    return device


def initialize_model(device, lora_config, args):
    print0("\n" + "=" * 20 + "\nLoading model...\n" + "=" * 20 + "\n")
    
    if args.use_lora:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        if args.load_ckpt_path:
            load_ckpt_path = os.path.join(args.load_ckpt_path, args.ckpt_path, f"step_{args.load_ckpt_step}")
            model = PeftModel.from_pretrained(model, load_ckpt_path, is_trainable=True)
        else:
            model = get_peft_model(model, lora_config)
    else:
        model_path = args.model_path if not args.load_ckpt_path else os.path.join(args.load_ckpt_path, args.ckpt_path, f"step_{args.load_ckpt_step}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        ).to(device)

        if args.add_tokens:
            num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': args.add_tokens})
            model.resize_token_embeddings(len(tokenizer))  
            embedding_layer = model.get_input_embeddings()  
            with torch.no_grad():
                new_token_indices = range(len(tokenizer) - num_added_tokens, len(tokenizer))
                for token_index in new_token_indices:
                    embedding_layer.weight[token_index].uniform_(-0.1, 0.1)  
    
    return model, tokenizer


def prepare_dataloader(deepspeed_config, device, args):
    print0("\n" + "=" * 20 + "\nLoading dataset...\n" + "=" * 20 + "\n")
    train_dataset = TorchMultiFileBinaryDataset(args.data_path, device)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if args.local_rank != -1 else None
    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=deepspeed_config["train_micro_batch_size_per_gpu"],
        num_workers=0,
        # drop_last=True,
    )
    print0("\n" + "=" * 20 + "\nDataset is loaded.\n" + "=" * 20 + "\n")
    return train_dataloader
class ProgressiveRefinementLoss(nn.Module):
    def __init__(self, lambda_1=0.4, lambda_2=0.3, lambda_3=0.3):
        super(ProgressiveRefinementLoss, self).__init__()
        self.lambda_1 = lambda_1 
        self.lambda_2 = lambda_2 
        self.lambda_3 = lambda_3 
        self.ce_loss = nn.CrossEntropyLoss()  

    def forward(self, final_pred, final_target, thought_sequence, prev_thought_sequence, confidence_scores):
        final_loss = self.lambda_1 * self.ce_loss(final_pred, final_target)
        consistency_loss = self.lambda_2 * torch.mean((thought_sequence - prev_thought_sequence) ** 2)
        confidence_loss = self.lambda_3 * torch.mean(1 - confidence_scores)
        total_loss = final_loss + consistency_loss + confidence_loss
        return total_loss

def train_model(model, tokenizer, train_dataloader, ds_config, args):
    engine, optimizer, _, _ = deepspeed.initialize(
        config=ds_config,
        model=model,
        model_parameters=model.parameters(),
    )

    step = 0
    losses = []
    begin_epoch = 1  
    begin_epoch_step = 0  
    end_epoch = args.max_epochs if args.max_epochs else (args.max_steps - 1) // len(train_dataloader) + 1
    loss_fn = ProgressiveRefinementLoss(lambda_1=1.0, lambda_2=0.5, lambda_3=0.2)
    
    if args.load_ckpt_path:
        print0("\n" + "=" * 20 + "\nLoading ckpt...\n" + "=" * 20 + "\n")
        ckpt_path = os.path.join(args.load_ckpt_path, args.ckpt_path)
        load_ckpt_step = f"step_{args.load_ckpt_step}"
        load_path = os.path.join(ckpt_path, load_ckpt_step)

        try:
            engine.load_checkpoint(ckpt_path, load_ckpt_step)
        except:
            pass

        loss_fn = os.path.join(load_path, "loss_list.json")
        with open(loss_fn, "r") as f:
            losses = json.load(f)

        step = args.load_ckpt_step
        begin_epoch = step // len(train_dataloader) + 1
        begin_epoch_step = step % len(train_dataloader)

    if dist.get_rank() == 0:
        if args.max_steps:
            total_train_steps = args.max_steps - step
        else:
            total_train_steps = (args.max_epochs - begin_epoch + 1) * len(train_dataloader) - begin_epoch_step
        pbar = tqdm(total=total_train_steps, ncols=95)
        
        if args.load_ckpt_path:
            skip_pbar = tqdm(total=begin_epoch_step, desc="Loading checkpoint", ncols=90)

    for epoch in range(begin_epoch, end_epoch + 1):
        begin_step = 0 if epoch > begin_epoch else begin_epoch_step

        if args.local_rank != -1 and isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)

        for batch_id, batch in enumerate(train_dataloader):
            if epoch == begin_epoch and batch_id < begin_step:
                if dist.get_rank() == 0:
                    skip_pbar.update(1)
                continue
            if epoch == begin_epoch and batch_id == begin_step and dist.get_rank() == 0 and args.load_ckpt_path:
                skip_pbar.close()
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            prev_thought_sequence = batch["thought"].to(device) 
            outputs = engine(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                use_cache=False,
            )
            final_pred = outputs.logits 
            confidence_scores = torch.softmax(final_pred, dim=-1).max(dim=-1).values    
            
            
            loss = loss_fn(
                final_pred=final_pred,  
                final_target=labels,  
                thought_sequence=final_pred,  
                prev_thought_sequence=prev_thought_sequence, 
                confidence_scores=confidence_scores 
            )        
            engine.backward(loss)
            engine.step()
            step += 1
            losses.append(loss.item())
      
            if dist.get_rank() == 0:
                pbar.update()
                pbar.set_description(f"epoch:{epoch},batch:{batch_id + 1}/{len(train_dataloader)},loss:{np.mean(losses[-200:]):.4f}")

            if args.save_steps and step % args.save_steps == 0:
                save_checkpoint(engine, tokenizer, step, losses, args)
            
            if args.max_steps and step >= args.max_steps:
                break
        
        if args.save_epochs and epoch % args.save_epochs == 0:
            save_checkpoint(engine, tokenizer, step, losses, args)
        
        if args.max_steps and step >= args.max_steps:
            break

    if args.save_steps and args.max_steps and args.max_steps % args.save_steps != 0:
        save_checkpoint(engine, tokenizer, step, losses, args)
    if not args.save_steps and args.save_epochs and args.max_epochs % args.save_epochs != 0:
        save_checkpoint(engine, tokenizer, step, losses, args)

    if dist.get_rank() == 0:
        pbar.close()


def save_checkpoint(engine, tokenizer, step, losses, args):
    ckpt_path = os.path.join(args.save_path, args.ckpt_path)
    save_ckpt_step = f"step_{step}"
    save_path = os.path.join(ckpt_path, save_ckpt_step)
    os.makedirs(save_path, exist_ok=True)

    # 使用 engine.save_checkpoint 
    if args.save_optimizer:
        engine.save_checkpoint(ckpt_path, tag=save_ckpt_step)

    engine.save_16bit_model(save_path)

    with open(os.path.join(save_path, 'config.json'), 'w') as f:  # 保存config
        print(json.dumps(engine.module.config.to_dict(), indent=4), file=f)
    
    loss_file_name = os.path.join(save_path, "loss_list.json")
    with open(loss_file_name, "w") as f:
        json.dump(losses, f)
    draw_loss(save_path)

    if dist.get_rank() == 0 or args.local_rank == -1:
        tokenizer.save_pretrained(save_path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def initialize(args):
    set_seed(args.seed)

    # load config
    with open(args.deepspeed_config_path, "r") as f:
        deepspeed_config = json.load(f)
    lora_config = None

    if args.load_ckpt_path:
        with open(args.lora_config_path, "r") as f:
            lora_config = LoraConfig(**json.load(f))
        args.save_path = args.load_ckpt_path
        ckpt_path = os.path.join(args.load_ckpt_path, args.ckpt_path)
        file_list = os.listdir(ckpt_path)
        file_list = [x for x in file_list if os.path.isdir(os.path.join(ckpt_path, x))]
        file_list.sort(key=lambda x:int(x.split("_")[-1]))
        args.load_ckpt_step = int(file_list[-1].split("_")[-1])
        with open(os.path.join(args.load_ckpt_path, f"train_config_0.json"), "r") as f:
            initial_config = json.load(f)
            initial_num_gpus = initial_config["args"]["num_gpus"]
            assert torch.cuda.device_count() == initial_num_gpus, "num_gpus can't change when loading ckpt!"
    else:
        t = datetime.datetime.now()
        if args.no_timestamp:
            args.save_path = os.path.join(args.output_path, args.save_name)
        else:
            args.save_path = os.path.join(args.output_path, f"{t.year}-{t.month:02d}-{t.day:02d}_{t.hour:02d}-{t.minute:02d}_{args.save_name}")
        args.load_ckpt_step = 0
        
    os.makedirs(args.save_path, exist_ok=True)
    config_fn = os.path.join(args.save_path, f"train_config_{args.load_ckpt_step}.json")
    with open(config_fn, "w") as f:
        config_show = {
            "args": {
                "num_gpus": torch.cuda.device_count(),
                **vars(args),
            },
            "deepspeed_config": deepspeed_config,
        }
        if args.use_lora:
            config_show.update({"lora_config": lora_config})

        json.dump(config_show, f, indent=4)
    return deepspeed_config, lora_config


def parse_args():
    parser = argparse.ArgumentParser()
    # train params
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)    
    parser.add_argument("--seed", type=int, default=19260817)
    parser.add_argument("--load_ckpt_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, help="the root folder of your data")
    parser.add_argument("--deepspeed_config_path", type=str, default="deepspeed_config.json")
    # save params
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--save_epochs", type=int, default=None)
    parser.add_argument("--ckpt_path", type=str, default="ckpt")
    parser.add_argument("--save_name", type=str, required=True)
    parser.add_argument("--save_optimizer", action="store_true")
    parser.add_argument("--no_timestamp", action="store_true")
    # lora params
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_config_path", type=str, default="lora_config.json")
    # finetune params
    parser.add_argument("--add_tokens", nargs='+', default=None)
    # distribute params
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    assert bool(args.max_steps) != bool(args.max_epochs), "Specify exactly one of --max_steps and --max_epochs"
    assert args.save_steps or args.save_epochs, "Specify at least one of --save_steps and --save_epochs"
    if not args.use_lora:
        assert bool(args.model_path) or bool(args.load_ckpt_path), "Specify --model_path or --load_ckpt_path to define the base model."
    else:
        assert args.lora_config_path, "Specify --lora_config_path when --use_lora is set"
        assert args.add_tokens is None, "Do not specify --add_tokens when --use_lora is set."
        assert args.model_path, "Specify --model_path when --use_lora is set."
    if args.save_steps is not None and args.save_steps <= 0:
        raise ValueError("--save_steps must be greater than 0")
    if args.save_epochs is not None and args.save_epochs <= 0:
        raise ValueError("--save_epochs must be greater than 0")
    return args


def main():
    args = parse_args()
    deepspeed_config, lora_config = initialize(args)
    device = setup_distributed_environment(args.local_rank)
    model, tokenizer = initialize_model(device, lora_config, args)
    train_dataloader = prepare_dataloader(deepspeed_config, device, args)
    train_model(model, tokenizer, train_dataloader, deepspeed_config, args)


if __name__ == "__main__":
    main()
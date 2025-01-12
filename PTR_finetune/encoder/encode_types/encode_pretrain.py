import numpy as np


def encode_pretrain(tokenizer, document):
    input_ids = tokenizer(document)["input_ids"]
    if tokenizer.bos_token_id and input_ids[0] != tokenizer.bos_token_id:
        input_ids = [tokenizer.bos_token_id] + input_ids
    if input_ids[-1] != tokenizer.eos_token_id:
        input_ids = input_ids + [tokenizer.eos_token_id]
    return input_ids, len(input_ids)


def data_func_pretrain(data, tokenizer, args):
    save_dtype = np.int16 if args.save_dtype=="int16" else np.int32
    tokens = []
    for item in data:
        tokens.extend(item)
    max_length = args.max_length
    ret = []
    for i in range(0, len(tokens) - max_length, max_length):
        input_ids = tokens[i:i + max_length]
        input_ids = np.array(input_ids, dtype=save_dtype)
        ret.append(input_ids)

    pad_len = max_length - len(tokens) % max_length
    input_ids = tokens[-(len(tokens) % max_length):] + [tokenizer.eos_token_id] * pad_len
    labels = tokens[-(len(tokens) % max_length):] + [-100] * pad_len
    input_ids = np.array(input_ids, dtype=save_dtype)
    labels = np.array(labels, dtype=save_dtype)
    ret.append((input_ids, labels))
    return ret
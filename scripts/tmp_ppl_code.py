# Written by Yukang Chen
# Some code based on https://github.com/epfml/landmark-attention
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os,sys
import math
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm
import transformers
# from peft import PeftModel
# from llama_attn_replace import replace_llama_attn
# sys.path.append("/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/transformers/examples/pytorch/language-modeling")
# from llama_sft_forward_thisversion import replace_llama_forward, replace_llama_forward_forfastinfer


sys.path.append("/apdcephfs/share_733425/vinnylywang/jianhuipang/gogollm/codes")
from llama_sft_forward_thisversion_new import replace_llama_forward_forinference_withasan as myinfer


anchortoid={
    ".":29889,
    ",":29892,
    ";":29936,
    "?":29973,
    "!":29991,
    "<AC>":32001
}


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size during inference')
    parser.add_argument('--base_model', type=str, default="/data1/pretrained-models/llama-7b-hf")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--seq_len', type=int, default=2048, help='context length during evaluation')
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--peft_model', type=str, default=None, help='')
    parser.add_argument('--flash_attn', type=bool, default=True, help='')
    parser.add_argument('--data_path', type=str, default="./test.bin", help='')
    parser.add_argument('--torch_load_data', type=bool, default=False, help='')
    parser.add_argument('--fast', type=bool, default=False, help='output file')
    parser.add_argument('--use_cache', type=bool, default=False, help='output file')
    parser.add_argument('--anchor', type=str, default=None, help='anchors . , ? !')
    parser.add_argument('--device', type=str, default="cuda:0", help='device: cuda:0')
    args = parser.parse_args()
    return args

def get_as_batch(data, seq_length, batch_size, device='cpu', sliding_window=256):
    all_ix = list(range(0, len(data) - seq_length, sliding_window))
    all_ix.pop()

    for idx in range(0, len(all_ix), batch_size):
        ix = all_ix[idx:idx+batch_size]
        assert all([idx + seq_length + 1 <= len(data) for idx in ix])
        x = torch.stack([torch.from_numpy((data[i:i+seq_length]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_length]).astype(np.int64)) for i in ix])
        if device != 'cpu':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        yield x, y

def iceildiv(x, y):
    return (x + y - 1) // y

def evaluate(model, data, batch_size, device, seq_length, kvlen=2000, use_cache=True):
    stats = {}

    model.eval()

    loss_list_val, acc_list = [], []
    loss_step_list_val = []

    with torch.no_grad():
        print(f"Using seq length {seq_length}")
        torch.set_printoptions(sci_mode=False)

        val_loss = 0.
        acc = 0.
        cnt = 0

        idx = 0
        
        
        prefix = torch.stack([torch.from_numpy((data[:kvlen]).astype(np.int64))])
        inputs = torch.stack([torch.from_numpy((data[kvlen:]).astype(np.int64))])




            
            for part_idx, i in enumerate(range(0, x.shape[1], seq_length)):
                # if not getattr(model.model, "acindex", False):
                model.model.acindex = []
     
                if seq_length > context_size:
                    use_cache = True
                    overlength=seq_length-context_size
                    ctcount = int(seq_length/context_size)
                    restlen = seq_length - ctcount*context_size

                    past_key_values=None
                    for k in range(ctcount):
                        part_len = x[:, i + context_size*k :i + context_size * (k+1)].shape[1]
                        input_ids=x[:, i + context_size*k :i + context_size * (k+1)]
                        
                        if part_len == 0:
                            val_loss = 0.0 + val_loss
                            acc = 0.0 + acc
                            cnt += 0.0
                            while len(loss_step_list_val) <= part_idx:
                                loss_step_list_val.append([])
                            loss_step_list_val[part_idx].append(0.0)
                        
                        else:
                        
                            outputs = model(
                                input_ids=x[:, i + context_size*k :i + context_size * (k+1)],
                                past_key_values = past_key_values,
                                labels=x[:, i + context_size*k :i + context_size * (k+1)].contiguous(),
                                use_cache=use_cache)
                            past_key_values = outputs.past_key_values

                            val_loss = outputs.loss * part_len + val_loss
                            acc = ((outputs.logits.argmax(-1) == y[:, i + context_size*k :i + context_size * (k+1)]).float().sum()) + acc
                            cnt += part_len
                            while len(loss_step_list_val) <= part_idx:
                                loss_step_list_val.append([])
                            loss_step_list_val[part_idx].append(outputs.loss.item())
                            # print('a',k, part_len, past_key_values[0][0].shape[2])


                    if restlen > 0:
                        part_len = x[:, i + context_size*ctcount:i + context_size * ctcount+restlen].shape[1]

                        if part_len == 0:
                            val_loss = 0.0 + val_loss
                            acc = 0.0 + acc
                            cnt += 0.0
                            while len(loss_step_list_val) <= part_idx:
                                loss_step_list_val.append([])
                            loss_step_list_val[part_idx].append(0.0)

                        else:
                            outputs = model(
                                input_ids=x[:, i + context_size*ctcount:i + context_size * ctcount+restlen],
                                past_key_values = past_key_values,
                                labels=x[:, i + context_size*ctcount:i + context_size * ctcount+restlen].contiguous(),
                                use_cache=use_cache)
                            past_key_values = outputs.past_key_values

                            val_loss = outputs.loss * part_len + val_loss
                            acc = ((outputs.logits.argmax(-1) == y[:, i + context_size*ctcount:i + context_size * ctcount+restlen]).float().sum()) + acc
                            cnt += part_len
                            # print('b', part_len, past_key_values[0][0].shape[2])
                       

                            while len(loss_step_list_val) <= part_idx:
                                loss_step_list_val.append([])
                            loss_step_list_val[part_idx].append(outputs.loss.item())

                else:
                    
                    part_len = x[:, i:i + seq_length].shape[1]

                    outputs = model(
                        input_ids=x[:, i:i + seq_length],
                        labels=x[:, i:i+seq_length].contiguous(),
                        use_cache=use_cache)

                    val_loss = outputs.loss * part_len + val_loss
                    acc = ((outputs.logits.argmax(-1) == y[:, i:i+seq_length]).float().sum()) + acc
                    cnt += part_len

                    while len(loss_step_list_val) <= part_idx:
                        loss_step_list_val.append([])
                    loss_step_list_val[part_idx].append(outputs.loss.item())

                model.model.acindex=[]
                
            val_loss /= cnt
            acc /= cnt
            
            loss_list_val.append(val_loss.item())
            acc_list.append(acc.item())

    stats['val_acc'] = torch.as_tensor(acc_list).mean().item()
    stats['val_loss'] = torch.as_tensor(loss_list_val).mean().item()
    stats['val_perplexity'] = 2.71828 ** stats['val_loss']
    stats['val_perplexity_per_chunk'] = torch.exp(torch.as_tensor(loss_step_list_val).mean(dim=1))

    return stats

def main(args):

    # device = "cuda:1"
    device = args.device
    seed = 2
    torch.cuda.set_device(device)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if args.anchor is not None:
        achorid=[]
        anchor_syms=args.anchor.strip().split('|')
        for ans in anchor_syms:
            achorid.append(anchortoid[ans])
            myinfer(achorid)
        print("ac !")
    else:
        print("no ac !")

    if args.torch_load_data:
        data = torch.load(args.data_path)
        data["val"] = np.array(data["val"])
    else:
        data = {'val': np.memmap(args.data_path, dtype=np.uint16, mode='r')}

    data["val"] = data["val"][:3000]

    print(f"Num validation tokens: {len(data['val'])}")
    print("data path", args.data_path)
    print("base model", args.base_model)
    print("peft model", args.peft_model)

    # if args.flash_attn:
    #     replace_llama_attn(use_flash_attn=True, use_full=True)
    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
    )

    context_size = args.context_size if args.context_size > 0 else args.seq_len
    orig_ctx_len = getattr(config, "max_position_embeddings", None) # this value should be 4096 for LLaMA2 models
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if args.anchor is None:
        model.resize_token_embeddings(32001)
    # model.resize_token_embeddings(32001)

    # if args.peft_model:
    #     trainable_params = os.path.join(args.peft_model, "trainable_params.bin")
    #     if os.path.isfile(trainable_params):
    #         model.load_state_dict(torch.load(trainable_params, map_location=model.device), strict=False)
    #     else:
    #         raise ValueError("Trainable input embedding and normalization are required.")
    #     model = PeftModel.from_pretrained(
    #         model,
    #         args.peft_model,
    #         device_map="auto",
    #         torch_dtype=torch.float16,
    #     )
    use_cache=args.use_cache
    stats = evaluate(model, data, args.batch_size, device, args.seq_len, kvlen=2000, use_cache=True)

    print(stats)


if __name__ == "__main__":
    args = parse_config()
    main(args)
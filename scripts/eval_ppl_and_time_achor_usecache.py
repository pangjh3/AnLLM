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
import time
import numpy as np
# from peft import PeftModel
# from llama_attn_replace import replace_llama_attn
# sys.path.append("/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/transformers/examples/pytorch/language-modeling")
# from llama_sft_forward_thisversion import replace_llama_forward, replace_llama_forward_forfastinfer


# sys.path.append("/apdcephfs/share_733425/vinnylywang/jianhuipang/gogollm/codes")
sys.path.append("/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/gogollm/codes")
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
    parser.add_argument('--seq_len', type=int, default=3072, help='context length during evaluation')
    parser.add_argument('--context_size', type=int, default=4096, help='context size during fine-tuning')
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

def insert_anchors(tensor, n):
    if n <= 0:
        return tensor

    result = []
    for i, value in enumerate(tensor):
        if i > 0 and (i+1) % n == 0:
            result.append(32001)
        result.append(value.item())

    return np.array(result)

def evaluate(model, data, batch_size, device, seq_length=3072, kvlen=2048, every_len_anchor = 512, sliding_window=256, use_cache=False):
    
    stats = {}

    model.eval()

    loss_list_val, acc_list = [], []
    loss_step_list_val = []

    timecut = 0.0
    start=time.time()
    with torch.no_grad():
        print(f"Using seq length {seq_length}")
        torch.set_printoptions(sci_mode=False)
        for idx, (x, y) in tqdm(
            enumerate(
                get_as_batch(
                    data['val'], 
                    seq_length, 
                    batch_size, 
                    device=device,
                    sliding_window=sliding_window
                )
            ),
            total=iceildiv(
                iceildiv(len(data['val']), sliding_window),
                batch_size
            )
        ):
            if idx >1:
                break
            val_loss = 0.
            acc = 0.
            cnt = 0
            for part_idx, i in enumerate(range(0, x.shape[1], seq_length)):
                # if not getattr(model.model, "acindex", False):
                model.model.acindex = []
                model.model.offset = 0
                ss = time.time()
                if every_len_anchor != 0:
                    xprefix = torch.stack([torch.from_numpy(insert_anchors(x[0,i:i+kvlen], every_len_anchor).astype(np.int64))])
                    yprefix = torch.stack([torch.from_numpy(insert_anchors(y[0,i:i+kvlen], every_len_anchor).astype(np.int64))])
                    if device != 'cpu':
                        xprefix, yprefix = xprefix.pin_memory().to(device, non_blocking=True), yprefix.pin_memory().to(device, non_blocking=True)
                else:
                    xprefix = x[:,i:i+kvlen]
                    yprefix = y[:, i:i+kvlen]
                ee = time.time()
                timecut += ee - ss
                # part_len = x[:, i:i + seq_length].shape[1]
                # prefix_len = x[:,i:i+kvlen].shape[1]
                prefix_len = xprefix.shape[1]

                input_len = x[:,i+kvlen:].shape[1]

                outputs = model(
                    input_ids=xprefix,
                    past_key_values=None,
                    labels=xprefix,
                    use_cache=True)
                
                val_loss = outputs.loss * prefix_len + val_loss
                acc = ((outputs.logits.argmax(-1) == yprefix).float().sum()) + acc
                cnt += prefix_len
                past_key_values = outputs.past_key_values
                plen=past_key_values[0][0].shape
            
                while len(loss_step_list_val) <= part_idx:
                    loss_step_list_val.append([])
                loss_step_list_val[part_idx].append(outputs.loss.item())


                if kvlen < seq_length:
                    input_len = x[:,i+kvlen:i+seq_length].shape[1]
                    outputs = model(
                        input_ids=x[:,i+kvlen:i+seq_length],
                        past_key_values = past_key_values,
                        labels=x[:,i+kvlen:i+seq_length],
                        use_cache=use_cache)
                    past_key_values = outputs.past_key_values

                    val_loss = outputs.loss * input_len + val_loss
                    acc = ((outputs.logits.argmax(-1) == y[:,i+kvlen:i+seq_length]).float().sum()) + acc
                    cnt += input_len
                    print(prefix_len,input_len,plen)

                    while len(loss_step_list_val) <= part_idx:
                        loss_step_list_val.append([])
                    loss_step_list_val[part_idx].append(outputs.loss.item())

                '''tokens by tokens genernating'''
                # for i in range(seq_length-kvlen):
                    
                #     inputi_len = x[:,i+kvlen:i+kvlen+1].shape[1]
                #     outputs = model(
                #         input_ids=x[:,i+kvlen:i+kvlen+1],
                #         past_key_values = past_key_values,
                #         labels=x[:,i+kvlen:i+kvlen+1],
                #         use_cache=True)
                #     past_key_values = outputs.past_key_values

                #     val_loss = outputs.loss * inputi_len + val_loss
                #     acc = ((outputs.logits.argmax(-1) == y[:,i+kvlen:i+kvlen+1]).float().sum()) + acc
                #     cnt += inputi_len
                #     # print(inputi_len,val_loss,acc,cnt)

                #     while len(loss_step_list_val) <= part_idx:
                #         loss_step_list_val.append([])
                #     loss_step_list_val[part_idx].append(outputs.loss.item())

                model.model.acindex=[]
                model.model.offset = 0

                
            val_loss /= cnt
            acc /= cnt
            
            loss_list_val.append(val_loss.item())
            acc_list.append(acc.item())
    end=time.time()

    stats['val_acc'] = torch.as_tensor(acc_list).mean().item()
    stats['val_loss'] = torch.as_tensor(loss_list_val).mean().item()
    stats['val_perplexity'] = 2.71828 ** stats['val_loss']
    stats['val_perplexity_per_chunk'] = torch.exp(torch.as_tensor(loss_step_list_val).mean(dim=1))
    stats["alltime"] = end-start-timecut
    
    
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
        print("load_torchdata")
    else:
        data = {'val': np.memmap(args.data_path, dtype=np.uint16, mode='r')}
    
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
    print(use_cache)
    start=time.time()
    stats = evaluate(model, data, args.batch_size, device, args.seq_len, sliding_window=256, use_cache=use_cache)
    end=time.time()
    print(stats)


if __name__ == "__main__":
    args = parse_config()
    main(args)
##############################
# Function: inference
# Author: Wenxiang Jiao
# Last modified: 2023/04/06
##############################

import argparse
from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
import torch
import random
import json
import numpy as np
import sys
import time

sys.path.append("/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/gogollm/codes")
# from llama_sft_forward_thisversion_new import replace_llama_forward_forinference_withasan as myinfer
from llama_sft_forward_thisversion_new import replace_llama_forward_forinference_withasan_addfullatt as myinfer


import nltk
nltkdir="$rootpath/jianhuipang/nltk_data/"
sent_tokenizer = nltk.data.load(nltkdir+'tokenizers/punkt/english.pickle')
DEFAULT_PJH_TOKEN="<AC>"
def add_ac(text):
    adds = ' ' + DEFAULT_PJH_TOKEN + ' '
    textsents = sent_tokenizer.tokenize(text)
    text = adds.join(textsents)
    return text

# Instruction language, default: 'en'
lang_instruction = {
    'de': {'de': "Deutsch", 'en': "Englisch", 'ja': "Japanisch", 'zh': "Chinesisch"},
    'en': {'de': "German", 'en': "English", 'ja': "Japanese", 'zh': "Chinese",'he':'Hebrew','uk':'Ukrainian'},
    'ja': {'de': "ドイツ語", 'en': "英語", 'ja': "日本語", 'zh': "中国語"},
    'zh': {'de': "德语", 'en': "英语", 'ja': "日语", 'zh': "中文"},
}

# Special tokens in llama
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_PJH_TOKEN = "<AC>"

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_input_ac": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n <AC>"
        "### Instruction:\n{instruction}\n\n <AC> ### Input:\n{input}\n\n <AC> ### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


# Read task instruction, fill in languages
def read_instruct(path, src, tgt, lang_ins="en"):
    source, target = lang_instruction[lang_ins][src], lang_instruction[lang_ins][tgt]
    ins_list = []
    with open(path, 'r', encoding='utf-8') as f:
        for l in f:
            line = l.strip().replace("[SRC]", source).replace("[TGT]", target)
            ins_list.append(line)
    return ins_list


# Read input data for inference
def read_input(path):
    with open(path, 'r', encoding='utf-8') as f:
        input_data = f.readlines()
    return input_data


# Assembly instruction and input data, handle hints
def create_prompt(instruct, input_data, template="prompt_no_input"):
    if "###" in instruct:
        instruct, input_suffix = instruct.split("###")
        hint = "\n\n### Hint: {}".format(input_suffix)
    else:
        instruct =  instruct
        hint = ""
    if template == "prompt_input_ac":
        list_data_dict = [{"instruction": instruct, "input": add_ac(p.strip()) + hint} for p in input_data]
        prompt_input = PROMPT_DICT[template]
        sources = [ prompt_input.format_map(example) for example in list_data_dict ]  
    elif template == "prompt_input":
        list_data_dict = [{"instruction": instruct, "input": p.strip() + hint} for p in input_data]
        prompt_input = PROMPT_DICT[template]
        sources = [ prompt_input.format_map(example) for example in list_data_dict ]
    else:
        list_data_dict = [{"instruction": "\n\n".join([instruct, p.strip() + hint]).strip(), "input": ""} for p in input_data]
        prompt_input = PROMPT_DICT[template]
        sources = [ prompt_input.format_map(example) for example in list_data_dict ]
    return sources


# # Post-process the output, extract translations
# def pre_process(text):
#     text = text.split("### Input:")
#     t1 = text[0]
#     t2 = "### Input:"
#     t3 = text[1]
#     return t1, t2, t3

# Post-process the output, extract translations
def post_process(text):
    text = text.split("### Response:")[1].strip()
    text = text.replace("\n", " ")
    if " <AC> " in text:
        text = text.replace(" <AC> ", "")
    if "<AC>" in text:
        text = text.replace("<AC>", "")

    # Cut for contrastive instruction
    if "</p>" in text:
        text = text.split("</p>")[0].split("<p>")[-1]
    return text

def split_text(text):
    text = text.split("### Response:")[1].strip()
    text = text.replace("\n", " ")
    if " <AC> " in text:
        text = text.replace(" <AC> ", "")
    if "<AC>" in text:
        text = text.replace("<AC>", "")

    # Cut for contrastive instruction
    if "</p>" in text:
        text = text.split("</p>")[0].split("<p>")[-1]
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--anchor', '-ac', type=str, default=None, help='instruction file')
    parser.add_argument('--model-name-or-path', type=str, required=True, help='model name in the hub or local path')
    parser.add_argument('--inst-file', '-ins', type=str, default=None, help='instruction file')
    parser.add_argument('--input-file','-i', type=str, required=True, help='input file')
    parser.add_argument('--output-file','-o', type=str, required=True, help='output file')
    parser.add_argument('--lang-pair', '-lp', type=str, default='zh-en', help='language pair: zh-en, en-de')
    parser.add_argument('--search-algorithm', '-sa', type=str, default='beam', help='search algorithms: sample, beam')
    parser.add_argument('--batch', '-b', type=int, default=2, help='batch size')
    parser.add_argument('--template', '-tp', type=int, default=1, help='0: prompt_no_input, 1: prompt_input')
    parser.add_argument('--temperature', '-t', type=float, default=0.1, help='temperature: 0.7 for text generation')
    parser.add_argument('--topp', '-tpp', type=float, default=0.95, help='temperature: 0.7 for text generation')
    parser.add_argument('--topk', '-tpk', type=int, default=40, help='temperature: 0.7 for text generation')
    parser.add_argument('--length', '-l', type=int, default=1024, help='length of the output text')
    parser.add_argument('--block', '-bl', type=int, default=4096, help='length of the window')
    parser.add_argument('--num-beams', '-nb', type=int, default=1, help='length of the window')
    args = parser.parse_args()
    model_name_or_path = args.model_name_or_path
    inst_file = args.inst_file
    input_file = args.input_file
    output_file = args.output_file
    lang_pair = args.lang_pair
    search = args.search_algorithm
    batch = args.batch
    temperature = args.temperature
    temp = args.template
    length = args.length
    block_size = args.block
    num_beams=args.num_beams
    topp=args.topp
    topk=args.topk
    template = "prompt_input" if temp > 0 else "prompt_no_input"

    # copy from zefeng
    with open(model_name_or_path+'/config.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    if int(data['max_position_embeddings'])< int(block_size):
        data['max_position_embeddings'] = block_size
    # with open(model_name_or_path+'/config.json', 'w', encoding='utf-8') as file:
    #     json.dump(data, file, ensure_ascii=False, indent=4)

    # Load checkpoints
    print(f'Loading Mater Model weights from path: {model_name_or_path}')
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
    # print(model.hf_device_map)
    # bloom uses only fast tokenize
    to_use_fast = False
    if "bloom" in model_name_or_path:
        to_use_fast = True
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=to_use_fast)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # from tokenizers import AddedToken
    # tokenizer.add_tokens(AddedToken("\n", normalized=False))

    # replace_llama_forward(tokenizer.convert_tokens_to_ids(DEFAULT_PJH_TOKEN), tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
    anchorid=None
    if args.anchor is not None:
        print("replace forward function...")
        myinfer([tokenizer.convert_tokens_to_ids(args.anchor)])
        anchorid=tokenizer.convert_tokens_to_ids(args.anchor)

    if args.anchor == "<AC>":
        template = "prompt_input_ac"

    gen_config = GenerationConfig(temperature=temperature,
                                  top_p=topp,
                                  top_k=topk,
                                  do_sample=True,
                                  num_beams=num_beams,
                                  max_new_tokens=length,
                                  eos_token_id=tokenizer.eos_token_id,
                                  pad_token=tokenizer.pad_token_id,
                                  )

    if search == "beam":
        gen_config = GenerationConfig(temperature=temperature,
                                      top_p=topp,
                                      top_k=topk,
                                      num_beams=num_beams,
                                      max_new_tokens=length,
                                      eos_token_id=tokenizer.eos_token_id,
                                      pad_token=tokenizer.pad_token_id,
                                      )

    # Prepare input data
    srcl, tgtl = lang_pair.split('-')
    if inst_file is not None:
        instructs = read_instruct(inst_file, srcl, tgtl)
        instruct = instructs[0] if len(instructs) > 0 else ""
    else: # In case instruction file is missing, then use input as instruction
        instruct = ""
        template = "prompt_no_input"
        

    input_data = read_input(input_file)
    prompt = create_prompt(instruct, input_data, template)
    avgkvl = 0.0
    avgallkvl=0.0
    avganchorkvl=0.0
    count = 0.0
    # Generate
    torch.manual_seed(0)
    with open(output_file, 'w', encoding='utf-8') as fo,open(output_file+".hyp", 'w', encoding='utf-8') as fo2:
        for i in range(0, len(prompt), batch):
            count += 1
            model.model.acindex=[]
            model.model.offset=0
            p = prompt[i:i+batch]
            tokenized = tokenizer(p, padding=True, return_tensors="pt")
            input_ids = tokenized.input_ids.cuda()
            attn_mask = tokenized.attention_mask.cuda()
            input_ids = input_ids[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else input_ids
            attn_mask = attn_mask[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else attn_mask
            keyinfoindex = torch.nonzero(input_ids[0]==13).squeeze().tolist()
            model.model.keyindexrange = [keyinfoindex[-3], keyinfoindex[-2]]
            with torch.no_grad():
                start = time.time()
                generated_ids = model.generate(inputs=input_ids,attention_mask=attn_mask, generation_config=gen_config)
                end = time.time()
                tt = end-start
            decoded_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            if anchorid is not None:
                indextruefalse = generated_ids[0]==anchorid
                anchorkvl = torch.sum(indextruefalse)
                # acindex = torch.nonzero(indextruefalse).squeeze().tolist()
                acindex = model.model.acindex[:]
                lastindex=acindex[-1]
                # kvl=len(acindex)+generated_ids[0,lastindex:].shape[-1]
                kvl=model.model.maxkvl
                # allkv = len(torch.nonzero(generated_ids[0]!=anchorid).squeeze().tolist())
                allkv = generated_ids.shape[-1]
                print("anchorkv:{}, kv cache: {}, allkv: {}".format(anchorkvl, kvl, allkv), file=fo, flush=True)
            else:
                kvl = generated_ids.shape[-1]
                anchorkvl=0
                allkv=kvl
                print("kv cache: {}".format(kvl), file=fo, flush=True)
            avgkvl+=kvl
            avganchorkvl+=anchorkvl
            avgallkvl+=allkv
            for dec in decoded_tokens:
                print(dec, file=fo, flush=True)
                print(post_process(dec), file=fo2, flush=True)
            print("use time: {}".format(tt), file=fo, flush=True)   
            model.model.acindex=[]
            model.model.offset=0
            model.model.maxkvl=0
        print("avganchorkvl:{}, avgkvl: {},avgallkv:{}, count: {}".format(avganchorkvl, avgkvl, avgallkvl, count), file=fo, flush=True)


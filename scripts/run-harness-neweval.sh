# export CUDA_VISIBLE_DEVICES=0

# export http_proxy="http://star-proxy.oa.com:3128" 
# export https_proxy="http://star-proxy.oa.com:3128"
# export HF_HOME=/jianhuipang_qy3/old_hf_cache
# export TRANSFORMERS_CACHE=/jianhuipang_qy3/old_hf_cache

# codepath=/jianhuipang_qy3/gogollm/newmodel_harness.py

# results_path=./test/harness/addresults_n_split

# mkdir -p $results_path

# prefix=allm-7b-ac-5shot-10split
# model_path=/jianhuipang_qy3/opensourcellms/anllm-ac-ct/allm-ac-7b
# model_name=boolq.${prefix}
# python $codepath \
#     --model hf-causal-experimental \
#     --model_args pretrained=${model_path} \
#     --tasks boolq \
#     --anchors "<AC>" \
#     --num_fewshot 5 \
#     --batch_size 1 \
#     --no_cache \
#     --output_path $results_path/${model_name}.json \
#     --device cuda:0 \
#     | tee -a $results_path/${model_name}.log


# export CUDA_VISIBLE_DEVICES=7  

# export http_proxy="http://star-proxy.oa.com:3128" 
# export https_proxy="http://star-proxy.oa.com:3128"
# export HF_HOME=/jianhuipang_qy3/old_hf_cache
# export TRANSFORMERS_CACHE=/jianhuipang_qy3/old_hf_cache

# codepath=/jianhuipang_qy3/gogollm/newmodel_harness.py

# results_path=./test/harness/addresults_sent_split

# pythonp=/jianhuipang_qy3/anaconda3/envs/pyforoldharnessenv/bin/python

# mkdir -p $results_path

# prefix=allm-7b-ac-0shot
# model_path=/jianhuipang_qy3/opensourcellms/anllm-ac-ct/allm-ac-7b
# model_name=qasper.${prefix}
# $pythonp $codepath \
#     --model hf-causal-experimental \
#     --model_args pretrained=${model_path} \
#     --tasks qasper \
#     --anchors "<AC>" \
#     --num_fewshot 0 \
#     --batch_size 1 \
#     --no_cache \
#     --output_path $results_path/${model_name}.json \
#     --device cuda:0 \
#     | tee -a $results_path/${model_name}.log


# export CUDA_VISIBLE_DEVICES=7  

# export http_proxy="http://star-proxy.oa.com:3128" 
# export https_proxy="http://star-proxy.oa.com:3128"
# export HF_HOME=/jianhuipang_qy3/old_hf_cache
# export TRANSFORMERS_CACHE=/jianhuipang_qy3/old_hf_cache

# codepath=/jianhuipang_qy3/gogollm/newmodel_harness.py

# results_path=./test/harness/addresults_sent_split

# pythonp=/jianhuipang_qy3/anaconda3/envs/pyforoldharnessenv/bin/python

# mkdir -p $results_path

# prefix=llama2-7b-0shot
# model_path=/jianhuipang_qy3/opensourcellms/Llama-2-7b-hf
# model_name=qasper.${prefix}
# $pythonp $codepath \
#     --model hf-causal-experimental \
#     --model_args pretrained=${model_path} \
#     --tasks qasper \
#     --num_fewshot 0 \
#     --batch_size 1 \
#     --no_cache \
#     --output_path $results_path/${model_name}.json \
#     --device cuda:0 \
#     | tee -a $results_path/${model_name}.log



# export CUDA_VISIBLE_DEVICES=7  

# export http_proxy="http://star-proxy.oa.com:3128" 
# export https_proxy="http://star-proxy.oa.com:3128"
# export HF_HOME=/jianhuipang_qy3/old_hf_cache
# export TRANSFORMERS_CACHE=/jianhuipang_qy3/old_hf_cache

# codepath=/jianhuipang_qy3/gogollm/newmodel_harness.py

# results_path=./test/harness/addresults_sent_split

# pythonp=/jianhuipang_qy3/anaconda3/envs/pyforoldharnessenv/bin/python

# mkdir -p $results_path

# prefix=anllm-7b-juhao-0shot
# model_path=/jianhuipang_qy3/opensourcellms/anllm-ep-ct/allm-juhao-7b
# model_name=qasper.${prefix}
# $pythonp $codepath \
#     --model hf-causal-experimental \
#     --model_args pretrained=${model_path} \
#     --tasks qasper \
#     --anchors "." \
#     --num_fewshot 0 \
#     --batch_size 1 \
#     --no_cache \
#     --output_path $results_path/${model_name}.json \
#     --device cuda:0 \
#     | tee -a $results_path/${model_name}.log




export CUDA_VISIBLE_DEVICES=7  

export http_proxy="http://star-proxy.oa.com:3128" 
export https_proxy="http://star-proxy.oa.com:3128"
export HF_HOME=/jianhuipang_qy3/old_hf_cache
export TRANSFORMERS_CACHE=/jianhuipang_qy3/old_hf_cache

codepath=/jianhuipang_qy3/gogollm/newmodel_harness.py

results_path=./test/harness/addresults_sent_split

pythonp=/jianhuipang_qy3/anaconda3/envs/pyforoldharnessenv/bin/python

mkdir -p $results_path

prefix=allm-7b-ac-0shot-puncac-2
model_path=/jianhuipang_qy3/opensourcellms/anllm-ac-ct/allm-ac-7b
model_name=qasper.${prefix}
$pythonp $codepath \
    --model hf-causal-experimental \
    --model_args pretrained=${model_path} \
    --tasks qasper \
    --anchors "<AC>" \
    --num_fewshot 0 \
    --batch_size 1 \
    --no_cache \
    --output_path $results_path/${model_name}.json \
    --device cuda:0 \
    | tee -a $results_path/${model_name}.log

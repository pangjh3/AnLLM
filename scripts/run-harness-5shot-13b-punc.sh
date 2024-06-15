
export http_proxy="http://star-proxy.oa.com:3128" 
export https_proxy="http://star-proxy.oa.com:3128"

# export HF_HOME=/jianhuipang/hf_cache
# export TRANSFORMERS_CACHE=/jianhuipang/hf_cache
export CUDA_VISIBLE_DEVICES=6
export HF_HOME=/jianhuipang_qy3/old_hf_cache
export TRANSFORMERS_CACHE=/jianhuipang_qy3/old_hf_cache

codepath=/jianhuipang/gogollm/newmodels/newmodel_harness.py

results_path=./test/harness/allresults_13b_punc

mkdir -p $results_path

prefix=allm-13b-punc-5shot-anchor-2
model_path=/jianhuipang/gogollm/newmodels/checkpoints_ct/punc/allm-juhao-7b
model_path=gogollm/checkpoints2/llama2_13b_sfton_RedPajama-Data-1T-Sample-32gpus-8accgrad_douhao
model_name=all.${prefix}
python $codepath \
    --model hf-causal-experimental \
    --model_args pretrained=${model_path} \
    --tasks hellaswag,sciq,winogrande,arc_easy,arc_challenge,boolq,piqa,openbookqa \
    --num_fewshot 5 \
    --batch_size 1 \
    --anchors "." \
    --no_cache \
    --output_path $results_path/${model_name}.json \
    --device cuda:0 \
    | tee -a $results_path/${model_name}.log

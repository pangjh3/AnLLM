export CUDA_VISIBLE_DEVICES=7

export http_proxy="http://star-proxy.oa.com:3128" 
export https_proxy="http://star-proxy.oa.com:3128"
export HF_HOME=/jianhuipang_qy3/old_hf_cache
export TRANSFORMERS_CACHE=/jianhuipang_qy3/old_hf_cache

codepath=/jianhuipang_qy3/gogollm/newmodel_harness.py

results_path=./test/harness/addresults_sent_split

pythonp=/jianhuipang_qy3/anaconda3/envs/pyforoldharnessenv/bin/python

mkdir -p $results_path


prefix=allm-7b-ac-5shot-sent
model_path=/jianhuipang_qy3/opensourcellms/anllm-ac-ct/allm-ac-7b
model_name=all.${prefix}
$pythonp $codepath \
    --model hf-causal-experimental \
    --model_args pretrained=${model_path} \
    --tasks hellaswag,sciq,winogrande,arc_easy,arc_challenge,boolq,piqa,openbookqa \
    --anchors "<AC>" \
    --num_fewshot 5 \
    --batch_size 1 \
    --no_cache \
    --output_path $results_path/${model_name}.json \
    --device cuda:0 \
    | tee -a $results_path/${model_name}.log

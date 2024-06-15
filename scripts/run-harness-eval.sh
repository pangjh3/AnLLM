
export http_proxy="http://star-proxy.oa.com:3128" 
export https_proxy="http://star-proxy.oa.com:3128"
export HF_HOME=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/old_hf_cache
export TRANSFORMERS_CACHE=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/old_hf_cache

codepath=/apdcephfs/share_733425/vinnylywang/jianhuipang/gogollm/newmodels/newmodel_harness.py

results_path=./test/harness/newestresults

mkdir -p $results_path

prefix=allm-7b-ac-5shot-noanchor
model_path=/apdcephfs/share_733425/vinnylywang/jianhuipang/gogollm/newmodels/checkpoints_ct/ac/allm-ac-7b
model_name=boolq2.${prefix}
python3 $codepath \
    --model hf-causal-experimental \
    --model_args pretrained=${model_path} \
    --tasks boolq \
    --num_fewshot 5 \
    --batch_size 1 \
    --no_cache \
    --output_path $results_path/${model_name}.json \
    --device cuda:0 \
    | tee -a $results_path/${model_name}.log



export http_proxy="http://star-proxy.oa.com:3128" 
export https_proxy="http://star-proxy.oa.com:3128"
export HF_HOME=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/old_hf_cache
export TRANSFORMERS_CACHE=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/old_hf_cache

codepath=/apdcephfs/share_733425/vinnylywang/jianhuipang/gogollm/newmodels/newmodel_harness.py

results_path=./test/harness/newestresults

mkdir -p $results_path

export CUDA_VISIBLE_DEVICES=6
prefix=allm-7b-punc-5shot-noanchor
model_path=/apdcephfs/share_733425/vinnylywang/jianhuipang/gogollm/newmodels/checkpoints_ct/punc/allm-juhao-7b
model_name=boolq2.${prefix}
python3 $codepath \
    --model hf-causal-experimental \
    --model_args pretrained=${model_path} \
    --tasks boolq \
    --num_fewshot 5 \
    --batch_size 1 \
    --no_cache \
    --output_path $results_path/${model_name}.json \
    --device cuda:0 \
    | tee -a $results_path/${model_name}.log


# hendrycksTest-* hellaswag,sciq,winogrande,arc_easy,arc_challenge,boolq,piqa,openbookqa,coqa


export http_proxy="http://star-proxy.oa.com:3128" 
export https_proxy="http://star-proxy.oa.com:3128"
export HF_HOME=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/old_hf_cache
export TRANSFORMERS_CACHE=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/old_hf_cache

codepath=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/gogollm/newmodel_harness.py

results_path=./test/harness/addresults_n_split

mkdir -p $results_path

prefix=allm-7b-punc-5shot-anchor
model_path=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/opensourcellms/anllm-ac-ct/allm-ac-7b
model_name=boolq2.${prefix}
python3 $codepath \
    --model hf-causal-experimental \
    --model_args pretrained=${model_path} \
    --tasks boolq \
    --anchors "." \
    --num_fewshot 5 \
    --batch_size 1 \
    --no_cache \
    --output_path $results_path/${model_name}.json \
    --device cuda:0 \
    | tee -a $results_path/${model_name}.log






# export http_proxy="http://star-proxy.oa.com:3128" 
# export https_proxy="http://star-proxy.oa.com:3128"
# export HF_HOME=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/old_hf_cache
# export TRANSFORMERS_CACHE=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/old_hf_cache
# codepath=/apdcephfs/share_733425/vinnylywang/jianhuipang/gogollm/newmodels/newmodel_harness.py

# results_path=./test/harness/newestresults

# mkdir -p $results_path

# task=boolq
# prefix=allm-7b-ac-5shot-oneanchorforeach
# model_path=/apdcephfs/share_733425/vinnylywang/jianhuipang/gogollm/newmodels/checkpoints_ct/ac/allm-ac-7b
# model_name=boolq.${prefix}
# python3 $codepath \
#     --model hf-causal-experimental \
#     --model_args pretrained=${model_path} \
#     --tasks $task \
#     --num_fewshot 5 \
#     --anchors "<AC>" \
#     --batch_size 1 \
#     --no_cache \
#     --output_path $results_path/${model_name}.json \
#     --device cuda:0 \
#     | tee -a $results_path/${model_name}.log
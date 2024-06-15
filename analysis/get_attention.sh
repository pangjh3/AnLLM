
model=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/opensourcellms/anllm-ep-ct/allm-juhao-7b
# model=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/opensourcellms/Llama-2-7b-hf
python ./get_infer_attention.py \
    --model-name-or-path $model \
    -o ./outputs.llama2.apple
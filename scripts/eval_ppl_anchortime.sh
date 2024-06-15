ppl(){


datapath=$1
modelpath=$2
name=$3
device=$4
seql=$5
# seql=$6

export CUDA_VISIBLE_DEVICES=$device

resultpath=./testpplandtime/testsampled

mkdir -p $resultpath

# for seql in 256 512 1024 2048 4096;do
# for seql in 8192;do 
# --anchor "<AC>" \

rname=$seql.$name

python3 eval_ppl_and_time_achor_usecache.py \
--seq_len $seql \
--use_cache true \
--batch_size 1 \
--anchor "<AC>" \
--base_model $modelpath \
--data_path $datapath \
--device cuda:0 \
> $resultpath/${rname}.txt


# done

}

datapath=/apdcephfs/share_733425/vinnylywang/jianhuipang/gogollm/test/pgforppl/test_sampled_data.bin
# datapath=/apdcephfs/share_733425/vinnylywang/jianhuipang/gogollm/test/pgforppl/test_sampled_data_forac.bin
# modelpath=/apdcephfs/share_733425/vinnylywang/jianhuipang/gogollm/newmodels/checkpoints_ct/punc/allm-juhao-7b
modelpath=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/apdcephfs/jianhuipang/gogollm/newmodels/checkpoints_ct/ac/allm-ac-7b
# modelpath=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/apdcephfs/jianhuipang/opensourcellms/llama2/Llama-2-7b-hf
# ppl $datapath $modelpath allm-juhao-7b-oversize 1 0
# ppl $datapath $1 $2 $3 $4
ppl $datapath $modelpath allmac7b-ac-512 0 3072
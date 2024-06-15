ppl(){


datapath=$1
modelpath=$2
name=$3
bs=$4 #4 for 13b
device=$5
seql=$6
ctx=$7

export CUDA_VISIBLE_DEVICES=$device

resultpath=./testppl/testsampled_txl

mkdir -p $resultpath

# for seql in 256 512 1024 2048 4096;do
# for seql in 8192;do

rname=$seql.$ctx.$name

python3 eval_ppl_usecache.py \
--seq_len $seql \
--context_size $ctx \
--batch_size $bs \
--base_model $modelpath \
--data_path $datapath \
--device cuda:0 \
> $resultpath/${rname}.txt


# done

}

datapath=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/gogollm/test/pgforppl/test_sampled_data.bin
modelpath=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/opensourcellms/Llama-2-7b-hf
# modelpath=/apdcephfs/share_733425/vinnylywang/jianhuipang/gogollm/newmodels/checkpoints_ct/punc/allm-juhao-7b

for seql in 512 1024 2048 4096;do
    ppl $datapath $modelpath llama2fortxl 8 1  $seql 128
done


for seql in 512 1024 2048 4096;do
    ppl $datapath $modelpath llama2fortxl 8 1  $seql 64
done


for seql in 512 1024 2048 4096;do
    ppl $datapath $modelpath llama2fortxl 8 1  $seql 256
done


# ppl $datapath $modelpath allm-juhao-7b-oversize 1 0
# ppl $datapath $modelpath llama2fortxl 8 1  512 256
# ppl $datapath $modelpath llama2fortxl 8 1  512 256
# ppl $datapath $modelpath llama2fortxl 8 1  1024 512
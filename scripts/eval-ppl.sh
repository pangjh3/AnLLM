ppl(){


datapath=$1
modelpath=$2
name=$3
bs=$4 #4 for 13b
device=$5
# seql=$6

export CUDA_VISIBLE_DEVICES=$device

resultpath=./testppl/testsampled

mkdir -p $resultpath

for seql in 256 512 1024 2048 4096;do
# for seql in 256;do

rname=$seql.$name

python3 eval_ppl.py \
--seq_len $seql \
--context_size 4096 \
--anchor "." \
--fast true \
--batch_size $bs \
--base_model $modelpath \
--data_path $datapath \
--device cuda:0 \
> $resultpath/${rname}.txt


done

}

datapath=/apdcephfs/share_733425/vinnylywang/jianhuipang/gogollm/test/pgforppl/test_sampled_data.bin
# modelpath=/apdcephfs/share_733425/vinnylywang/jianhuipang/gogollm/checkpoints2/llama2_7b_sfton_RedPajama-Data-1T-Sample-32gpus-8accgrad_douhao

# ppl $datapath $modelpath llama2-7b-cptonred-testsample

# datapath=/apdcephfs/share_733425/vinnylywang/jianhuipang/gogollm/test/pgforppl/test.bin
# ppl $datapath $1 $2 $3 $4 $5
ppl $datapath $1 $2 $3 $4
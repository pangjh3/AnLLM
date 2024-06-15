ppl(){


datapath=$1
modelpath=$2
name=$3
bs=$4 #4 for 13b
device=$5
seql=$6
ctx=$7

export CUDA_VISIBLE_DEVICES=$device

resultpath=./testppl/testsampled

mkdir -p $resultpath

# for seql in 256 512 1024 2048 4096;do
# for seql in 8192;do

rname=$seql.$ctx.$name

python3 eval_ppl_usecache.py \
--seq_len $seql \
--context_size $ctx \
--batch_size $bs \
--torch_load_data true \
--base_model $modelpath \
--data_path $datapath \
--device cuda:0 \
> $resultpath/${rname}.txt


# done

}

datapath=/jianhuipang/gogollm/test/pgforppl/test_sampled_data.bin
# modelpath=/jianhuipang/gogollm/newmodels/checkpoints_ct/punc/allm-juhao-7b
datapath=/jianhuipang/gogollm/test/pgforppl/test_sampled_data_forac.bin
# ppl $datapath $modelpath allm-juhao-7b-oversize 1 0
ppl $datapath $1 $2 $3 $4 $5 $6

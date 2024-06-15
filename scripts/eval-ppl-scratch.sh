ppl(){


datapath=$1
modelpath=$2
name=$3
bs=$4 #4 for 13b
device=$5
# seql=$6

export CUDA_VISIBLE_DEVICES=$device

resultpath=./testppl/anllmpt-testwikiart-103-epochppl-ctwithanchor-ct664

mkdir -p $resultpath

# for seql in 4096 2048 1024 512 256;do
for seql in 4096;do
# for seql in 256;do

rname=$seql.$name

python3 eval_ppl_forscratch.py \
--seq_len $seql \
--context_size 4096 \
--fast true \
--anchor "." \
--batch_size $bs \
--base_model $modelpath \
--data_path $datapath \
--device cuda:0 \
| tee -a $resultpath/${rname}.txt


done

}

datapath=/jianhuipang_qy3/datasets/wikitext-103/testforppl.bin
modelpath=/jianhuipang_qy3/models/checkpoints_ct/punc/allm-juhao-txllarge-wiki103
# ppl $datapath $modelpath llama2-7b-cptonred-testsample
modelpath=/jianhuipang_qy3/models/checkpoints_ct/punc/allm-juhao-txllarge-wiki103/checkpoint-500
modelpath=/jianhuipang_qy3/models/checkpoints_ct/punc/allm-juhao-txllarge-wiki103-bigbatch-5epoch/checkpoint-120
modelpath=/jianhuipang_qy3/models/checkpoints_ct/punc/allm-juhao-txllarge-wiki103-bigbatch-5epoch/checkpoint-180
modelpath=/jianhuipang_qy3/models/checkpoints_ct/punc/allm-juhao-txllarge-wiki103-bigbatch-5epoch/checkpoint-240
modelpath=/jianhuipang_qy3/models/checkpoints_ct/punc/allm-juhao-txllarge-wiki103-bigbatch-5epoch/checkpoint-5640
modelpath=/jianhuipang_qy3/models/checkpoints_ct/punc/allm-juhao-txllarge-wiki103-bigbatch-saveeachepoch-2
modelpath=/jianhuipang_qy3/models/checkpoints_ct/punc/allm-juhao-txllarge-wiki103-bigbatch-saveeachepoch-2/checkpoint-664
modelpathprefix=/jianhuipang_qy3/models/checkpoints_ct/punc/allm-juhao-txllarge-wiki103-bigbatch-saveeachepoch-2/checkpoint
modelpathprefix=/jianhuipang_qy3/models/wikiart/punc/allm-juhao-txllarge-fullatt-ctwithanchor-ct664/checkpoint
# datapath=/apdcephfs/share_733425/vinnylywang/jianhuipang/gogollm/test/pgforppl/test.bin
# ppl $datapath $1 $2 $3 $4 $5

for id in 28 57 86 115 144 173 202 231 259 288 317 346 375 404 433 462 490 519 548 577 606 635 664 693 721 750 779 808 837 840;do

ppl $datapath ${modelpathprefix}-${id} pt-allmjuhaotxllargewiki103 1 6

done

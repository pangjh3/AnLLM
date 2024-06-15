

for seql in 256 512 1024 2048 4096;do
modelpath=jianhuipang/gogollm/newmodels/checkpoints_ct/punc/allm-juhao-7b
gpu=3
bash eval-ppl-usecache.sh $modelpath allm-juhao-7b-oversize 1 $gpu $seql 4096
done

modelpath=jianhuipang/gogollm/newmodels/checkpoints_ct/punc/allm-juhao-7b
gpu=0
seql=8192
bash eval-ppl-usecache.sh $modelpath allm-juhao-7b-oversize-testcache 1 $gpu $seql 8192


modelpath=jianhuipang/opensourcellms/llama2/Llama-2-7b-hf
gpu=1
seql=8192
bash eval-ppl-usecache-normal.sh $modelpath Llama-2-7b-hf-oversize 1 $gpu $seql 8192




for seql in 256 512 1024 2048 4096;do
modelpath=jianhuipang/gogollm/newmodels/checkpoints_ct/punc/allm-juhao-7b
gpu=3
bash eval-ppl-usecache.sh $modelpath allm-juhao-7b-oversize 1 $gpu $seql 4096
done



### ac

for seql in 4096;do
modelpath=jianhuipang/gogollm/newmodels/checkpoints_ct/ac/allm-ac-7b
gpu=6
bash eval-ppl-usecache-normal.sh $modelpath allm-ac-7b-orgdatanoaddanchor 1 $gpu $seql 4096
done

for seql in 256 512 1024 2048 4096;do
modelpath=jianhuipang/gogollm/newmodels/checkpoints_ct/ac/allm-ac-7b
gpu=5
bash eval-ppl-usecache-ac.sh $modelpath allm-ac-7b 1 $gpu $seql 4096
done

for seql in 8192;do
modelpath=jianhuipang/gogollm/newmodels/checkpoints_ct/ac/allm-ac-7b
gpu=1
bash eval-ppl-usecache-ac.sh $modelpath allm-ac-7b-2 1 $gpu $seql 8192
done

for seql in 256 512 1024 2048 4096;do
modelpath=jianhuipang/gogollm/newmodels/checkpoints_ct/ac/allm-ac-7b
gpu=3
bash eval-ppl-usecache-ac-normal.sh $modelpath allm-ac-7b-newdatanoanchor 1 $gpu $seql 4096
done

gpu=0
for seql in 256 512 1024 2048 4096;do
modelpath=jianhuipang/gogollm/newmodels/checkpoints_ct/ac/allm-ac-7b
bash eval-ppl-usecache-ac-normal.sh $modelpath allm-ac-7b-newdatanoanchor-4 1 $gpu $seql 4096
done



modelpath=jianhuipang/gogollm/newmodels/checkpoints_ct/ac/allm-ac-7b
bash eval-ppl-usecache-ac-normal.sh $modelpath allm-ac-7b-newdatanoanchor-4 1 3 256 4096

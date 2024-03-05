

allmac=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/gogollm/checkpoints_ctthensft/fortranslation/ac2/allm-addac-alpacanewstest17to20-ac-7b
# allmjuhao=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/gogollm/checkpoints_ctthensft/fortranslation/punc/allm-addjuhao-alpacanewstest17to20-juhao-7b

# for t in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9;do
# 0.7 0.1 40
# 0.1 0.95 40

t=$1
topp=$2
topk=$3
gpu=$4
src=de
tgt=en
# done
modelpath=$allmac
modelname=allm-ac
datapath=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23
runfile=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/gogollm/applications/translation/inference_addfullatt-ac.py
srcfile=$datapath/test.${src}2${tgt}.${src}
tgtfile=$datapath/test.${src}2${tgt}.${tgt}
# srcfile=$datapath/test.de2en.${src}cd
# tgtfile=$datapath/test.de2en.${tgt}
# srcfile=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23/mergewmt2023test/test.500to1000.de2en.de
# tgtfile=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23/mergewmt2023test/test.500to1000.de2en.en
outdir=./translation/output-addfullatt/$modelname
insfilr=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/apdcephfs/jianhuipang/LLMs4MT/test/instruct_inf.txt

mkdir -p $outdir

export CUDA_VISIBLE_DEVICES=$gpu

outfile=$outdir/test.${src}2${tgt}.en.$t.$topp.$topk.out
hypfile=$outfile.hyp

python3 $runfile --model-name-or-path $modelpath \
    -lp ${src}-${tgt} \
    -t $t \
    -tpp $topp \
    -tpk $topk \
    -sa 'sample' \
    -ac '<AC>' \
    -b 1 \
    -ins $insfilr \
    -i $srcfile \
    -o $outfile

wc -l $hypfile | tee -a $outdir/${src}2${tgt}-results.txt
wc -l $tgtfile | tee -a $outdir/${src}2${tgt}-results.txt
cat $hypfile | sacrebleu -w 2 $tgtfile | tee -a $outdir/${src}2${tgt}-results.txt
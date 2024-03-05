allmjuhao=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/apdcephfs/jianhuipang/gogollm/newmodels/checkpoints_ctthensft/punc/allm-alpaca-juhao-7b
allmjuhao=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/gogollm/checkpoints_ctthensft/fortranslation/punc/allm-addjuhao-alpacanewstest17to20-juhao-7b
export CUDA_VISIBLE_DEVICES=6
t=0.1
topp=0.95
topk=40
src=de
tgt=en
# done
modelpath=$allmjuhao
modelname=allm-juhao-fortrans-final
datapath=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23
runfile=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/gogollm/applications/translation/inference_allmac.py
srcfile=$datapath/test.${src}2${tgt}.${src}
tgtfile=$datapath/test.${src}2${tgt}.${tgt}
# srcfile=$datapath/test.de2en.${src}
# tgtfile=$datapath/test.de2en.${tgt}
# srcfile=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23/mergewmt2023test/test.500to1000.de2en.de
# tgtfile=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23/mergewmt2023test/test.500to1000.de2en.en
outdir=./translation/output-fortranslationckpt/$modelname
insfilr=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/apdcephfs/jianhuipang/LLMs4MT/test/instruct_inf.txt

mkdir -p $outdir

outfile=$outdir/test.${src}2${tgt}.en.$t.$topp.$topk.out
hypfile=$outfile.hyp

python3 $runfile --model-name-or-path $modelpath \
    -lp ${src}-${tgt} \
    -t $t \
    -tpp $topp \
    -tpk $topk \
    -sa 'sample' \
    -ac '.' \
    -b 1 \
    -ins $insfilr \
    -i $srcfile \
    -o $outfile

echo "t-$t.topp-$topp.topk-$topk" | tee -a $outdir/${src}2${tgt}-results.txt
wc -l $hypfile | tee -a $outdir/${src}2${tgt}-results.txt
wc -l $tgtfile | tee -a $outdir/${src}2${tgt}-results.txt
cat $hypfile | sacrebleu -w 2 $tgtfile | tee -a $outdir/${src}2${tgt}-results.txt
echo ""


# allmjuhao=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/apdcephfs/jianhuipang/gogollm/newmodels/checkpoints_ctthensft/punc/allm-alpaca-juhao-7b

# for t in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9;do

# src=de
# tgt=en
# # done
# modelpath=$allmjuhao
# modelname=allm-juhao
# datapath=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23
# runfile=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/gogollm/applications/translation/inference_allmac.py
# srcfile=$datapath/test.${src}2${tgt}.${src}
# tgtfile=$datapath/test.${src}2${tgt}.${tgt}
# # srcfile=$datapath/test.de2en.${src}
# # tgtfile=$datapath/test.de2en.${tgt}
# # srcfile=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23/mergewmt2023test/test.500to1000.de2en.de
# # tgtfile=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23/mergewmt2023test/test.500to1000.de2en.en
# outdir=./translation/output/$modelname
# insfilr=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/apdcephfs/jianhuipang/LLMs4MT/test/instruct_inf.txt

# mkdir -p $outdir

# outfile=$outdir/test.${src}2${tgt}.en.${t}.out
# hypfile=$outfile.hyp

# python3 $runfile --model-name-or-path $modelpath \
#     -lp ${src}-${tgt} \
#     -t $t \
#     -sa 'sample' \
#     -ac '.' \
#     -b 1 \
#     -ins $insfilr \
#     -i $srcfile \
#     -o $outfile

# wc -l $hypfile | tee -a $outdir/${src}2${tgt}-results.txt
# wc -l $tgtfile | tee -a $outdir/${src}2${tgt}-results.txt
# cat $hypfile | sacrebleu -w 2 $tgtfile | tee -a $outdir/${src}2${tgt}-results.txt

# done


# for t in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9;do

# src=zh
# tgt=en
# # done
# modelpath=$allmjuhao
# modelname=allm-juhao
# datapath=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23
# runfile=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/gogollm/applications/translation/inference_allmac.py
# srcfile=$datapath/test.${src}2${tgt}.${src}
# tgtfile=$datapath/test.${src}2${tgt}.${tgt}
# # srcfile=$datapath/test.de2en.${src}
# # tgtfile=$datapath/test.de2en.${tgt}
# # srcfile=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23/mergewmt2023test/test.500to1000.de2en.de
# # tgtfile=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23/mergewmt2023test/test.500to1000.de2en.en
# outdir=./translation/output/$modelname
# insfilr=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/apdcephfs/jianhuipang/LLMs4MT/test/instruct_inf.txt

# mkdir -p $outdir

# outfile=$outdir/test.${src}2${tgt}.en.${t}.out
# hypfile=$outfile.hyp

# python3 $runfile --model-name-or-path $modelpath \
#     -lp ${src}-${tgt} \
#     -t $t \
#     -sa 'sample' \
#     -ac '.' \
#     -b 1 \
#     -ins $insfilr \
#     -i $srcfile \
#     -o $outfile

# wc -l $hypfile | tee -a $outdir/${src}2${tgt}-results.txt
# wc -l $tgtfile | tee -a $outdir/${src}2${tgt}-results.txt
# cat $hypfile | sacrebleu -w 2 $tgtfile | tee -a $outdir/${src}2${tgt}-results.txt

# done
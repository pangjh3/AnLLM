


allmac=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/gogollm/checkpoints_ctthensft/ac/allm-alpaca-ac-7b
# allmac=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/gogollm/checkpoints_ctthensft/fortranslation/ac/allm-addac-alpacanewstest17to20-ac-7b
allmac=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/gogollm/checkpoints_ctthensft/fortranslation/ac2/allm-addac-alpacanewstest17to20-ac-7b

# for t in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9;do
t=0.1
topp=0.95
topk=40
src=de
tgt=en
# done
modelpath=$allmac
modelname=allm-ac-acinput-final
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

export CUDA_VISIBLE_DEVICES=7

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

echo "t-$t.topp-$topp.topk-$topk" | tee -a $outdir/${src}2${tgt}-results.txt
wc -l $hypfile | tee -a $outdir/${src}2${tgt}-results.txt
wc -l $tgtfile | tee -a $outdir/${src}2${tgt}-results.txt
cat $hypfile | sacrebleu -w 2 $tgtfile | tee -a $outdir/${src}2${tgt}-results.txt
echo ""
# done



# export CUDA_VISIBLE_DEVICES=7

# allmpuncfortrans=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/gogollm/checkpoints_ctthensft/fortranslation/punc/allm-addjuhao-alpacanewstest17to20-juhao-7b
# src=de
# tgt=en
# # done
# modelpath=$allmpuncfortrans
# modelname=allm-juhao-fortrans
# datapath=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23
# runfile=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/gogollm/applications/translation/inference_allmac.py
# # srcfile=$datapath/test.${src}2${tgt}.${src}
# # tgtfile=$datapath/test.${src}2${tgt}.${tgt}
# srcfile=$datapath/test.de2en.${src}
# tgtfile=$datapath/test.de2en.${tgt}
# # srcfile=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23/mergewmt2023test/test.500to1000.de2en.de
# # tgtfile=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23/mergewmt2023test/test.500to1000.de2en.en
# outdir=./translation/output-fortranslationckpt/$modelname
# insfilr=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/apdcephfs/jianhuipang/LLMs4MT/test/instruct_inf.txt

# mkdir -p $outdir

# outfile=$outdir/test.${src}2${tgt}.en.out
# hypfile=$outfile.hyp

# python3 $runfile --model-name-or-path $modelpath \
#     -lp ${src}-${tgt} \
#     -t 0.1 \
#     -sa 'sample' \
#     -ac '.' \
#     -b 1 \
#     -ins $insfilr \
#     -i $srcfile \
#     -o $outfile

# wc -l $hypfile | tee -a $outdir/${src}2${tgt}-results.txt
# wc -l $tgtfile | tee -a $outdir/${src}2${tgt}-results.txt
# cat $hypfile | sacrebleu -w 2 $tgtfile | tee -a $outdir/${src}2${tgt}-results.txt

# src=zh
# tgt=en
# # done
# modelpath=$allmjuhao
# modelname=allm-juhao
# datapath=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23
# runfile=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/gogollm/applications/translation/inference_allmac.py
# # srcfile=$datapath/test.${src}2${tgt}.${src}
# # tgtfile=$datapath/test.${src}2${tgt}.${tgt}
# srcfile=$datapath/test.zh2en.${src}
# tgtfile=$datapath/test.zh2en.${tgt}
# # srcfile=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23/mergewmt2023test/test.500to1000.de2en.de
# # tgtfile=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23/mergewmt2023test/test.500to1000.de2en.en
# outdir=./translation/output/$modelname
# insfilr=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/apdcephfs/jianhuipang/LLMs4MT/test/instruct_inf.txt

# mkdir -p $outdir

# outfile=$outdir/test.${src}2${tgt}.en.out
# hypfile=$outfile.hyp

# python3 $runfile --model-name-or-path $modelpath \
#     -lp ${src}-${tgt} \
#     -t 0.1 \
#     -sa 'sample' \
#     -ac '.' \
#     -b 1 \
#     -ins $insfilr \
#     -i $srcfile \
#     -o $outfile

# wc -l $hypfile | tee -a $outdir/${src}2${tgt}-results.txt
# wc -l $tgtfile | tee -a $outdir/${src}2${tgt}-results.txt
# cat $hypfile | sacrebleu -w 2 $tgtfile | tee -a $outdir/${src}2${tgt}-results.txt






# # ---- 



# allmjuhao=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/apdcephfs/jianhuipang/gogollm/newmodels/checkpoints_ctthensft/punc/allm-alpaca-juhao-7b
# allmac=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/apdcephfs/jianhuipang/gogollm/newmodels/checkpoints_sft/ac/allm-alpaca-ac-7b

# src=de
# tgt=en
# # done
# modelpath=$allmac
# modelname=allm-acsft
# datapath=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23
# runfile=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/gogollm/applications/translation/inference_allmac.py
# # srcfile=$datapath/test.${src}2${tgt}.${src}
# # tgtfile=$datapath/test.${src}2${tgt}.${tgt}
# srcfile=$datapath/test.de2en.${src}
# tgtfile=$datapath/test.de2en.${tgt}
# # srcfile=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23/mergewmt2023test/test.500to1000.de2en.de
# # tgtfile=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23/mergewmt2023test/test.500to1000.de2en.en
# outdir=./translation/output/$modelname
# insfilr=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/apdcephfs/jianhuipang/LLMs4MT/test/instruct_inf.txt

# mkdir -p $outdir

# outfile=$outdir/test.${src}2${tgt}.en.out
# hypfile=$outfile.hyp

# python3 $runfile --model-name-or-path $modelpath \
#     -lp ${src}-${tgt} \
#     -t 0.1 \
#     -sa 'sample' \
#     -ac '<AC>' \
#     -b 1 \
#     -ins $insfilr \
#     -i $srcfile \
#     -o $outfile

# wc -l $hypfile | tee -a $outdir/${src}2${tgt}-results.txt
# wc -l $tgtfile | tee -a $outdir/${src}2${tgt}-results.txt
# cat $hypfile | sacrebleu -w 2 $tgtfile | tee -a $outdir/${src}2${tgt}-results.txt


# allmac=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/apdcephfs/jianhuipang/gogollm/newmodels/checkpoints_sft/ac/allm-alpaca-ac-7b

# src=zh
# tgt=en
# # done
# modelpath=$allmac
# modelname=allm-acsft
# datapath=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23
# runfile=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/gogollm/applications/translation/inference_allmac.py
# # srcfile=$datapath/test.${src}2${tgt}.${src}
# # tgtfile=$datapath/test.${src}2${tgt}.${tgt}
# srcfile=$datapath/test.zh2en.${src}
# tgtfile=$datapath/test.zh2en.${tgt}
# # srcfile=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23/mergewmt2023test/test.500to1000.de2en.de
# # tgtfile=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23/mergewmt2023test/test.500to1000.de2en.en
# outdir=./translation/output/$modelname
# insfilr=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/apdcephfs/jianhuipang/LLMs4MT/test/instruct_inf.txt

# mkdir -p $outdir

# outfile=$outdir/test.${src}2${tgt}.en.out
# hypfile=$outfile.hyp

# python3 $runfile --model-name-or-path $modelpath \
#     -lp ${src}-${tgt} \
#     -t 0.1 \
#     -sa 'sample' \
#     -ac '<AC>' \
#     -b 1 \
#     -ins $insfilr \
#     -i $srcfile \
#     -o $outfile

# wc -l $hypfile | tee -a $outdir/${src}2${tgt}-results.txt
# wc -l $tgtfile | tee -a $outdir/${src}2${tgt}-results.txt
# cat $hypfile | sacrebleu -w 2 $tgtfile | tee -a $outdir/${src}2${tgt}-results.txt


# # -----

# allmsftjuhao=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/apdcephfs/jianhuipang/gogollm/newmodels/checkpoints_sft/punc/allm-alpaca-juhao-7b

# src=de
# tgt=en
# # done
# modelpath=$allmsftjuhao
# modelname=allm-juhaosft
# datapath=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23
# runfile=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/gogollm/applications/translation/inference_allmac.py
# # srcfile=$datapath/test.${src}2${tgt}.${src}
# # tgtfile=$datapath/test.${src}2${tgt}.${tgt}
# srcfile=$datapath/test.de2en.${src}
# tgtfile=$datapath/test.de2en.${tgt}
# # srcfile=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23/mergewmt2023test/test.500to1000.de2en.de
# # tgtfile=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23/mergewmt2023test/test.500to1000.de2en.en
# outdir=./translation/output/$modelname
# insfilr=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/apdcephfs/jianhuipang/LLMs4MT/test/instruct_inf.txt

# mkdir -p $outdir

# outfile=$outdir/test.${src}2${tgt}.en.out
# hypfile=$outfile.hyp

# python3 $runfile --model-name-or-path $modelpath \
#     -lp ${src}-${tgt} \
#     -t 0.1 \
#     -sa 'sample' \
#     -ac '.' \
#     -b 1 \
#     -ins $insfilr \
#     -i $srcfile \
#     -o $outfile

# wc -l $hypfile | tee -a $outdir/${src}2${tgt}-results.txt
# wc -l $tgtfile | tee -a $outdir/${src}2${tgt}-results.txt
# cat $hypfile | sacrebleu -w 2 $tgtfile | tee -a $outdir/${src}2${tgt}-results.txt

# src=zh
# tgt=en
# # done
# modelpath=$allmsftjuhao
# modelname=allm-juhaosft
# datapath=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23
# runfile=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/gogollm/applications/translation/inference_allmac.py
# # srcfile=$datapath/test.${src}2${tgt}.${src}
# # tgtfile=$datapath/test.${src}2${tgt}.${tgt}
# srcfile=$datapath/test.zh2en.${src}
# tgtfile=$datapath/test.zh2en.${tgt}
# # srcfile=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23/mergewmt2023test/test.500to1000.de2en.de
# # tgtfile=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23/mergewmt2023test/test.500to1000.de2en.en
# outdir=./translation/output/$modelname
# insfilr=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/apdcephfs/jianhuipang/LLMs4MT/test/instruct_inf.txt

# mkdir -p $outdir

# outfile=$outdir/test.${src}2${tgt}.en.out
# hypfile=$outfile.hyp

# python3 $runfile --model-name-or-path $modelpath \
#     -lp ${src}-${tgt} \
#     -t 0.1 \
#     -sa 'sample' \
#     -ac '.' \
#     -b 1 \
#     -ins $insfilr \
#     -i $srcfile \
#     -o $outfile

# wc -l $hypfile | tee -a $outdir/${src}2${tgt}-results.txt
# wc -l $tgtfile | tee -a $outdir/${src}2${tgt}-results.txt
# cat $hypfile | sacrebleu -w 2 $tgtfile | tee -a $outdir/${src}2${tgt}-results.txt



# # ----orgmodel

# orgmodel=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/apdcephfs/jianhuipang/LLMs4MT/model/newptmodel-llms4mt-zh2en-32a100/llama2-sfton-0-bitexts-and-alpacagpt4-and-newstests17to20

# src=de
# tgt=en
# # done
# modelpath=$orgmodel
# modelname=llama2-7b
# datapath=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23
# runfile=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/gogollm/applications/translation/inference_allmac.py
# # srcfile=$datapath/test.${src}2${tgt}.${src}
# # tgtfile=$datapath/test.${src}2${tgt}.${tgt}
# srcfile=$datapath/test.de2en.${src}
# tgtfile=$datapath/test.de2en.${tgt}
# # srcfile=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23/mergewmt2023test/test.500to1000.de2en.de
# # tgtfile=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23/mergewmt2023test/test.500to1000.de2en.en
# outdir=./translation/output/$modelname
# insfilr=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/apdcephfs/jianhuipang/LLMs4MT/test/instruct_inf.txt

# mkdir -p $outdir

# outfile=$outdir/test.${src}2${tgt}.en.out
# hypfile=$outfile.hyp

# python3 $runfile --model-name-or-path $modelpath \
#     -lp ${src}-${tgt} \
#     -t 0.1 \
#     -sa 'sample' \
#     -b 1 \
#     -ins $insfilr \
#     -i $srcfile \
#     -o $outfile

# wc -l $hypfile | tee -a $outdir/${src}2${tgt}-results.txt
# wc -l $tgtfile | tee -a $outdir/${src}2${tgt}-results.txt
# cat $hypfile | sacrebleu -w 2 $tgtfile | tee -a $outdir/${src}2${tgt}-results.txt

# src=zh
# tgt=en
# # done
# modelpath=$orgmodel
# modelname=llama2-7b
# datapath=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23
# runfile=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/gogollm/applications/translation/inference_allmac.py
# # srcfile=$datapath/test.${src}2${tgt}.${src}
# # tgtfile=$datapath/test.${src}2${tgt}.${tgt}
# srcfile=$datapath/test.zh2en.${src}
# tgtfile=$datapath/test.zh2en.${tgt}
# # srcfile=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23/mergewmt2023test/test.500to1000.de2en.de
# # tgtfile=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23/mergewmt2023test/test.500to1000.de2en.en
# outdir=./translation/output/$modelname
# insfilr=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/apdcephfs/jianhuipang/LLMs4MT/test/instruct_inf.txt

# mkdir -p $outdir

# outfile=$outdir/test.${src}2${tgt}.en.out
# hypfile=$outfile.hyp

# python3 $runfile --model-name-or-path $modelpath \
#     -lp ${src}-${tgt} \
#     -t 0.1 \
#     -sa 'sample' \
#     -b 1 \
#     -ins $insfilr \
#     -i $srcfile \
#     -o $outfile

# wc -l $hypfile | tee -a $outdir/${src}2${tgt}-results.txt
# wc -l $tgtfile | tee -a $outdir/${src}2${tgt}-results.txt
# cat $hypfile | sacrebleu -w 2 $tgtfile | tee -a $outdir/${src}2${tgt}-results.txt
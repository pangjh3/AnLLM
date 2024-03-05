
# src=de
# tgt=en
# # size=$3
# prenum=64204
# if [ "$src" == "en" ]; then
# l1=$tgt
# l2=$src
# lanp=${l1}2${l2}
# else
# l1=$src
# l2=$tgt
# lanp=${l1}2${l2}
# fi

cometmodeldir=$rootpath/jianhuipang/opensourcellms/comet/wmt22-comet-da/checkpoints/model.ckpt

# for size in 100000;do

# for t in 1.0; do
# # for t in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do


# # $rootpath/jianhuipang/LLMs4MT/test/WMT23/newptmodel_sample/test.de2en.model.10000.0.1.out.hyp
# modelpath=$model_save
# datapath=$rootpath/jianhuipang/LLMs4MT/test/WMT23/
# srcfile=$datapath/test.$lanp.${src}
# tgtfile=$datapath/test.$lanp.${tgt}
# outfile=$datapath/newptmodel_sample/test.${src}2${tgt}.model.${size}.${t}.out
# # outfile=$rootpath/jianhuipang/LLMs4MT/test/WMT23/newptmodel/test.de2en.model.100000.out
# hypfile=$outfile.hyp

# # python3 train/inference_llama2.py --model-name-or-path $modelpath \
# #     -lp ${src}-${tgt} \
# #     -t 0.1 \
# #     -sa 'beam' \
# #     -ins test/instruct_inf.txt \
# #     -i $srcfile \
# #     -o $outfile

# wc -l $hypfile | tee -a $datapath/${src}2${tgt}-results.txt
# wc -l $tgtfile | tee -a $datapath/${src}2${tgt}-results.txt
# cat $hypfile | sacrebleu -w 2 $tgtfile | tee -a $datapath/${src}2${tgt}-results.txt


# cometresultpath=$datapath/comet

# mkdir -p $cometresultpath

# comet-score -s $srcfile -t $hypfile -r $tgtfile --model $cometmodeldir | tee -a $cometresultpath/${t}.${src}2${tgt}-comet.txt

# done

# done

hypfile=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/gogollm/applications/translation/output/llama2-7b/test.de2en.en.out.hyp

hypfile=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/gogollm/applications/translation/output/allm-ac-noacinput/test.de2en.en.out.hyp

hypfile=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/gogollm/applications/translation/output/allm-ac-acinput/test.de2en.en.out

hypfile=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/gogollm/applications/translation/output/allm-juhao/test.de2en.en.out

hypfile=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/gogollm/applications/translation/output/allm-juhao/test.de2en.en.0.9.out.hyp

srcfile=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/apdcephfs/jianhuipang/LLMs4MT/test/WMT23/test.de2en.de
reffile=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/apdcephfs/jianhuipang/LLMs4MT/test/WMT23/test.de2en.en


# hypfile=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/gogollm/applications/translation/output/llama2-7b/test.zh2en.en.out
# srcfile=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/apdcephfs/jianhuipang/LLMs4MT/test/WMT23/test.zh2en.zh
# reffile=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/apdcephfs/jianhuipang/LLMs4MT/test/WMT23/test.zh2en.en


comet-score -s $srcfile -t $hypfile -r $reffile --model $cometmodeldir 
# comet-score -s $srcfile -t $hypfile -r $reffile --model $cometmodeldir | tee -a ./llama2-zh2en-comet.txt

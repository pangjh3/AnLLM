

cometmodeldir=$rootpath/jianhuipang/opensourcellms/comet/wmt22-comet-da/checkpoints/model.ckpt




hypfile=$1

srcfile=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/apdcephfs/jianhuipang/LLMs4MT/test/WMT23/test.de2en.de
reffile=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/apdcephfs/jianhuipang/LLMs4MT/test/WMT23/test.de2en.en


comet-score -s $srcfile -t $hypfile -r $reffile --model $cometmodeldir | tee -a ./de2en-comet.txt

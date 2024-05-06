

cometmodeldir=$rootpath/jianhuipang/opensourcellms/comet/wmt22-comet-da/checkpoints/model.ckpt




hypfile=$1

srcfile=$datapath/WMT23/test.de2en.de
reffile=$datapath/WMT23/test.de2en.en


comet-score -s $srcfile -t $hypfile -r $reffile --model $cometmodeldir | tee -a ./de2en-comet.txt

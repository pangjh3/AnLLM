
cometmodeldir=$rootpath/jianhuipang/opensourcellms/comet/wmt22-comet-da/checkpoints/model.ckpt



hypfile=./applications/translation/output/llama2-7b/test.de2en.en.out.hyp

# hypfile=./applications/translation/output/allm-ac-noacinput/test.de2en.en.out.hyp

# hypfile=./applications/translation/output/allm-ac-acinput/test.de2en.en.out

# hypfile=./applications/translation/output/allm-juhao/test.de2en.en.out

# hypfile=./applications/translation/output/allm-juhao/test.de2en.en.0.9.out.hyp

srcfile=./WMT23/test.de2en.de
reffile=./WMT23/test.de2en.en


comet-score -s $srcfile -t $hypfile -r $reffile --model $cometmodeldir 


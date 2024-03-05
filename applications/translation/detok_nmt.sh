
hypfile=$rootpath/jianhuipang/fairseq/wmt23models2/results/long/de2en.10000000.5.long.hyp.out

grep ^H $hypfile \
| sed 's/^H\-//' \
| sort -n -k 1 \
| cut -f 3 \
| sacremoses detokenize \
> ./de2en.hyp


grep ^S $hypfile \
| sed 's/^S\-//' \
| sort -n -k 1 \
| cut -f 2 \
| sacremoses detokenize \
> ./de2en.src


grep ^T $hypfile \
| sed 's/^T\-//' \
| sort -n -k 1 \
| cut -f 2 \
| sacremoses detokenize \
> ./de2en.ref
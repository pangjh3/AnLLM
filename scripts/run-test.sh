

run_test(){

modelpath=$1
modelname=$2
testr=$3

testcode=/jianhuipang/gogollm/inference.py
# testcode=/jianhuipang/LLMs4MT/train/inference_llama2.py
testcode=./inference.py

outpath=./test/newresults/$modelname

mkdir -p $outpath

# inspath=./oldtest/ins_s_2.txt
# inputfile=./oldtest/input_2.txt

inspath=./oldtest/ins_s.txt
inputfile=./oldtest/input.txt


# inspath=../test/ins_for_sumary.txt
# inputfile=../test/input_for_sumary.txt

outfile=$outpath/${testr}.out
hypfile=$outfile.hyp

# python3 $testcode --model-name-or-path $modelpath \
#     -t 0.5 \
#     -tpp 0.95 \
#     -a ".|?" \
#     -fs true \
#     -sa 'sample' \
#     -ins $inspath \
#     -i $inputfile \
#     -o $outfile

python3 $testcode --model-name-or-path $modelpath \
    -t 0.1 \
    -tpp 0.9 \
    -a "." \
    -fs true \
    -sa 'sample' \
    -ins $inspath \
    -i $inputfile \
    -o $outfile

}

modelpath=/jianhuipang/LLMs4MT/acllm/checkpoints/llama2_7b_sfton_ac_gpt4alpaca
# modelpath=/jianhuipang/opensourcellms/llama2/Llama-2-7b-hf
# modelpath=/jianhuipang/gogollm/checkpoints2/llama2_7b_sfton_RedPajama-Data-1T-Sample-32gpus/checkpoint-8000
modelpath=/jianhuipang/gogollm/checkpoints_sft/llama2_7b_sfton_ac8000_gpt4alpaca
# modelpath=/jianhuipang/gogollm/checkpoints_sft/llama2_7b_acfinal_sfton_acgpt4alpaca

# best model
modelpath=/jianhuipang/gogollm/checkpoints_sftdouhaocpt/llama2_7b_douhaofinal_sfton_acgpt4alpaca_douhao


# modelpath=/jianhuipang/gogollm/checkpoints_sftsymbols/llama2_7b_sfton_acgpt4alpaca_symbols
# # modelpath=/jianhuipang/gogollm/checkpoints_sftsymbols/llama2chat_7b_sfton_acgpt4alpaca_symbols

# # modelpath=/jianhuipang/LLMs4MT/model/newptmodel-llms4mt-zh2en-32a100/llama2-sfton-0-bitexts-and-alpacagpt4-and-newstests17to20

# # modelpath=/jianhuipang/gogollm/checkpoints_sftsymbols/llama2_7b_sfton_acgpt4alpaca_threesymbols

# modelpath=/jianhuipang/gogollm/checkpoints_sftsymbols/llama2_13b_sfton_acgpt4alpaca_threesymbols

# modelpath=/jianhuipang/gogollm/checkpoints_sftdouhaocpt/llama2_7b_douhaofinal_sfton_gpt4alpaca_douhao

# modelpath=/jianhuipang/gogollm/checkpoints2/llama2_7b_sfton_RedPajama-Data-1T-Sample-32gpus-8accgrad_douhao
# modelpath=/jianhuipang/opensourcellms/llama2/Llama-2-7b-hf

# modelpath=/jianhuipang/gogollm/checkpoints_sftdouhaocpt/llama2_7b_douhaofinal_sfton_gpt4alpaca_douhao

modelpath=/jianhuipang/gogollm/checkpoints_sftdouhaocpt/llama2_13b_douhaofinal_sfton_gpt4alpaca_douhao_wenhao

modelpath=/jianhuipang/gogollm/checkpoints_sftdouhaocpt/llama2_13b_douhaofinal_sfton_gpt4alpaca_douhao_wenhao_withmt
modelpath=/jianhuipang/gogollm/checkpoints_sftdouhaocpt/llama2_13b_epoch2_1kstep_sftonalpaca

modelpath=/jianhuipang/gogollm/checkpoints_sftdouhaocpt/llama2_13b_douhaofinal_sfton_gpt4alpaca_douhao

modelpath=/jianhuipang/gogollm/checkpoints_sftdouhaocpt/llama2_7b_douhaofinal_sfton_acgpt4alpaca_douhao

modelpath=/jianhuipang/gogollm/checkpoints_sftdouhaocpt/llama2_13b_epoch2_sftonalpaca
# modelpath=/jianhuipang/gogollm/checkpoints_sftdouhaocpt/llama2_13b_douhaofinal_sfton_gpt4alpaca_douhao_wenhao_withmt

modelpath=/jianhuipang/gogollm/checkpoints_sftdouhaocpt/llama2_13b_epoch2_sftonalpaca_3epoch

modelpath=/jianhuipang/gogollm/checkpoints_sftdouhaocpt/llama2chat_7b_sftonalpaca_3epoch
modelpath=/jianhuipang/gogollm/checkpoints_sft/allsymbols/llama2chat_7b_sftonalpaca_3epoch
modelpath=/jianhuipang/gogollm/checkpoints_sftsymbols/llama2chat_7b_sfton_acgpt4alpaca_symbols

modelpath=/jianhuipang/gogollm/checkpoints_sft/allsymbols/llama2_7b_ct500steps_sftonalpaca_3epoch

modelpath=/jianhuipang/gogollm/checkpoints_sft/allsymbols/llama2_7b_ct1000steps_sftonalpaca_3epoch

modelpath=/jianhuipang/gogollm/checkpoints_sft/allsymbols/llama2_7b_ct1000steps_sftonlima_15epoch

modelpath=/jianhuipang/gogollm/checkpoints_sft/notallsymbols/llama2_13b_ct1000steps_sftonlima_15epoch

modelpath=/jianhuipang/gogollm/checkpoints_sft/allsymbols/ctthensft/llama2_7b_gpt4alpaca

modelpath=/jianhuipang/gogollm/checkpoints_sftdouhaocpt/llama2_7b_douhaofinal_sfton_acgpt4alpaca_douhao


modelpath=/jianhuipang/gogollm/newmodels/checkpoints_sft/punc/allm-alpaca-juhao-7b

modelpath=/jianhuipang/gogollm/newmodels/checkpoints_ctthensft/punc/allm-alpaca-juhao-7b

modelpath=/apdcephfs_qy3/share_733425/vinnylywang/jianhuipang_qy3/apdcephfs/jianhuipang/gogollm/newmodels/checkpoints_sft/punc/allm-alpaca-juhao-7b
run_test $modelpath checkpoints_sft-punc-allm-alpaca-juhao-7b translate

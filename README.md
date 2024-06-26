<div align="center">
    <img width="20%" alt="Anllm" src="./others/masks.jpeg">
    <h2>
    Anchor-based Large Language Models <br><br>
     <a href="https://arxiv.org/abs/2402.07616"> <img alt="paper link" src="https://img.shields.io/badge/Paper-arXiv-red"> </a>
     <!-- <a href="https://github.com/wxjiao/InstructMT"> <img alt="data link" src="https://img.shields.io/badge/Data-InstructMT-blue"> </a>  -->
    </h2>
</div>

Large language models (LLMs) predominantly employ decoder-only transformer architectures, necessitating the retention of keys/values information for historical tokens to provide contextual information and avoid redundant computation. However, the substantial size and parameter volume of these LLMs require massive GPU memory. This memory demand increases with the length of the input text, leading to an urgent need for more efficient methods of information storage and processing. This study introduces Anchor-based LLMs (AnLLMs), which utilize an innovative anchor-based self-attention network (AnSAN) and also an anchor-based inference strategy. This approach enables LLMs to compress sequence information into an anchor token, reducing the keys/values cache and enhancing inference efficiency. Experiments on question-answering benchmarks reveal that AnLLMs maintain similar accuracy levels while achieving up to 99\% keys/values cache reduction and up to 3.5 times faster inference. Despite a minor compromise in accuracy, the substantial enhancements of AnLLMs employing the AnSAN technique in resource utilization and computational efficiency underscore their potential for practical LLM applications.

## Anchor-based Attention Masks

<p align="center">
<img src="./others/training.jpg" width="600" />
</p>

## Anchor-based Inference

<p align="center">
<img src="./others/infer.jpg" width="500" />
</p>

## Running Environment

- Python 3.8
- Transformer 4.28.0

## Training

### Prepare data

We use the [RedPajama-Data-1T-sample](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample) datasets from Huggingface Repo.
For AnLLM-AC, add an new token \<AC> in the end of each sequeence. 

### Continue Pre-training

```
bash run-allm-ac.sh
```

### Supervised Fine-Tuning

We convert the data into Alpaca format, and fine-tune our models.

```
bash run run-sft-allm-ac.sh
```

## Open-Source Models

- [AnLLM-AC-7B](https://huggingface.co/pangjh3/AnLLM-AC-7B-PT): is continuely pre-trained on RedPajama-Data-1T-sample while added a new token \<AC> as anchor.
- [AnLLM-AC-13B](https://huggingface.co/pangjh3/AnLLM-AC-13B-PT): is continuely pre-trained on RedPajama-Data-1T-sample while added a new token \<AC> as anchor.
- [AnLLM-AC-translation-7B](https://huggingface.co/pangjh3/AnLLM-AC-7B-translation): is being supervised fine-tuned on Alpaca format parallel data, with a new added token \<AC>.
- [AnLLM-EP-7B](https://huggingface.co/pangjh3/AnLLM-EP-7B-PT): is continuely pre-trained on RedPajama-Data-1T-sample and the endpoint is the anchor token.
- [AnLLM-EP-translation-7B](https://huggingface.co/pangjh3/AnLLM-EP-7B-translation): is being supervised fine-tuned on Alpaca format parallel data.


## Information

If you find our work to be useful and beneficial for your research, we would greatly appreciate it if you could kindly cite our paper in your references. Thank you for your support!

```
@article{pang2024anchor,
  title={Anchor-based Large Language Models},
  author={Pang, Jianhui and Ye, Fanghua and Wong, Derek F and Wang, Longyue},
  journal={arXiv preprint arXiv:2402.07616},
  year={2024}
}
```


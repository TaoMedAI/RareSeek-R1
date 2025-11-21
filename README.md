# RareSeek-R1

**RareSeek-R1** is a domain-specialized large language model for rare-disease diagnostic reasoning, developed through a Progressive Parameter-Efficient Transfer Learning framework. The model is first instruction-tuned on the clinically grounded RareMed-Corpus, a large, multi-source dataset deeply integrated from medical textbooks, guidelines, biomedical literature, and real-world EHR narratives. It is then fine-tuned on RareMed-CoT, a high-fidelity corpus designed to instill explicit, stepwise clinical reasoning aligned with real diagnostic workflows.

## Requirements
### Hardware
- **GPU**: NVIDIA H800 GPUs (tested on 4 GPUs for inference)
### Dependencies
- Linux
- Python = 3.6
- PyTorch = 1.10.2
- Python 3.10
- CUDA = 12.6
- transformers = 4.51.3
- tokenizers = 0.21.1
- vLLM = 0.8.4

## Installation
```
$ pip install -r requirements.txt
```

## Model weights
The pretrained weights of MedMPT can be accessed upon request via [weights](https://huggingface.co/liangdi123/MedMPT) for non-commercial research and academic use.
You will typically receive a response within one week of submitting your request. If you do not hear back in a timely manner, please contact the corresponding author listed in the paper.

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
The pretrained weights of RareSeek-R1 can be accessed upon request via [Huggingface](https://huggingface.co/TaoMedAI/RareSeek-R1) for non-commercial research and academic use.Once granted access, please download the model weights and place them in the **models** folder.
You will typically receive a response within one week of submitting your request. If you do not hear back in a timely manner, please contact the corresponding author listed in the paper.

## Inference

### Installation

1. **Clone the repository**:
   ```bash
   git clone <https://github.com/TaoMedAI/RareSeek-R1>
   cd inference
   ```
  
2. **Create conda environment**:
   ```bash
   conda create -n rareseek python=3.10
   conda activate rareseek
   ```
   
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```


```python
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
model_path = "medicalai/ClinicalGPT-R1-Qwen-7B-EN-preview"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
data = pd.read_json('data/test.zip', lines=True).iloc[1]
prompt = f"{data["context"]}\n\nPlease provide a detailed and comprehensive diagnostic analysis of this medical record, and give the diagnostic results.\n"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer([text], return_tensors="pt").to(model.device)
output_ids = model.generate(**input_ids, max_new_tokens=2048, temperature=0.7, do_sample=True).to(model.device)
generated_text = tokenizer.decode(output_ids[0,len(input_ids[0]):], skip_special_tokens=True)
print("Generated Output:\n", generated_text)
```

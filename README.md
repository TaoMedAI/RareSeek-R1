# RareSeek-R1: Domain-Specialized Large Language Model for Rare-Disease Diagnostic Reasoning

[![Model Weights](https://img.shields.io/badge/Model_Weights-Huggingface-FFD21E.svg)](https://huggingface.co/TaoMedAI/RareSeek-R1)
[![Dataset](https://img.shields.io/badge/Dataset-RareMedData-blue.svg)](https://huggingface.co/datasets/TaoMedAI/RareMedData)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License](https://img.shields.io/badge/License-Non_Commercial-red.svg)](#license)

**RareSeek-R1** is a domain-specialized large language model tailored for rare-disease diagnostic reasoning, developed through a Progressive Parameter-Efficient Transfer Learning framework. 

The model is first instruction-tuned on the clinically grounded **RareMed-Corpus**, a large, multi-source dataset deeply integrated from medical textbooks, guidelines, biomedical literature, and real-world EHR (Electronic Health Record) narratives. It is subsequently fine-tuned on **RareMed-CoT**, a high-fidelity corpus designed to instill explicit, stepwise clinical reasoning that aligns with real-world diagnostic workflows.

![Overview of RareSeek-R1](https://github.com/TaoMedAI/RareSeek-R1/blob/main/RareSeek-R1.png)
*Figure 1: Overall framework and pipeline of RareSeek-R1.*

---

## 📑 Table of Contents
- [Data Availability](#-data-availability)
- [Getting Started](#-getting-started)
  - [Environment Setup](#environment-setup)
  - [Model Weights](#model-weights)
- [Usage Pipeline](#-usage-pipeline)
  - [1. EHR Preprocessing](#1-ehr-preprocessing)
  - [2. Phenotype Extraction](#2-phenotype-extraction-baselines--llms)
  - [3. Model Inference](#3-model-inference)
  - [4. Post-processing & Normalization](#4-post-processing--normalization)
  - [5. Evaluation](#5-evaluation)
- [License & Contact](#-license--contact)

---

## 📊 Data Availability

The open-source data underpinning this study, including training datasets and supplementary knowledge graph components, are publicly available.

- **RareMedData (Training Data & General KG)**: Available at [HuggingFace: TaoMedAI/RareMedData](https://huggingface.co/datasets/TaoMedAI/RareMedData).
- **Knowledge Graph Genetic Variation Data**:
  - **ClinVar**: Download the GRCh38 VCF release (2025.02.09) from the [NCBI FTP server](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/).
  - **HGMD**: Human Gene Mutation Database (2024.02 version) can be accessed via [HGMD](https://www.hgmd.cf.ac.uk/ac/index.php).

---

## 🚀 Getting Started

### Environment Setup

**Hardware Requirements:** 
- **GPU**: NVIDIA H800 GPUs (Tested primarily on 4 GPUs for parallel inference).
- **OS**: Linux

**Dependencies:**
1. Clone the repository:
   ```bash
   git clone [https://github.com/TaoMedAI/RareSeek-R1.git](https://github.com/TaoMedAI/RareSeek-R1.git)
   cd RareSeek-R1

# 📊 ConsTagFusion: A Credit Risk Assessment Framework Integrating Consumption Behavior and Financial Data

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Overview
This repository provides complete code and experimental support for the paper *"ConsTagFusion: Integrating Unstructured Consumption Behavior and Structured Financial Data with Hybrid Modeling for Credit Risk Assessment"*.

The framework innovatively integrates **structured financial data** and **unstructured consumption behavior tags**, achieving more accurate credit risk assessment through hybrid modeling of traditional machine learning and large language models (LLMs). The repository includes core model scripts, sample datasets, and pre-trained model configurations, supporting out-of-the-box experimental reproduction.


## 🔍 Key Features of the Project
- **Multimodal Fusion**: Integrates traditional financial features (age, income, etc.) with consumption behavior tags (e.g., credit card repayment, luxury goods consumption).
- **Model Diversity**: Supports mainstream models such as BERT, Llama-2-7B, and Qwen2.5-7B, compatible with traditional machine learning (XGBoost, etc.).
- **Technical Innovations**: Includes chi-square test feature screening, LoRA lightweight fine-tuning, three-stage prompt engineering, and SHAP interpretability analysis.
- **Chinese Optimization**: Optimizes models and prompt strategies for Chinese consumption tags and financial scenarios.


## 🏦 Dataset Description
> ⚠️ **Data Sensitivity Statement**: This data is sourced from a Chinese commercial bank and has been anonymized in accordance with confidentiality agreements. Due to the proprietary nature of commercial data, only 100 sample rows are provided for academic reproduction. For the complete dataset, please apply via email.

**Dataset Temporal Scope**: The data used in this study is from 2018, capturing consumer financial behaviors and credit outcomes within that timeframe.


| File Path | Number of Rows | Content Description |
|------------------------------|----------------|------------------------------------------|
| `Data/Dataset_Samples_Tradition.xlsx` | 100 | Traditional structured financial features (age, income, credit limit, default status, etc.) |
| `Data/Dataset_Samples_ConsumptionTags.xlsx` | 100 | Structured financial data + unstructured consumption behavior tags (including frequency information) |
| `Data/City_Levels.txt` | - | Mapping table of Chinese city tiers (1-6, reflecting economic development levels) |

📧 **For complete dataset application, please contact us**: [zl15750236895@gmail.com](mailto:zl15750236895@gmail.com)


## 🧮 Runtime Environment Requirements
### Hardware Configuration
- **Recommended**: NVIDIA GPU (≥16GB VRAM) + CUDA 11.8+ (accelerates LLM training and inference)
- **Compatible**: CPU mode (lower performance, suitable for small-scale experiments)

### Software Dependencies
- Python 3.9 or higher
- Dependent libraries: Listed in `requirements.txt` (install via `pip install -r requirements.txt`)
- **Special Note**: Need to install PyTorch matching the CUDA version (refer to the [PyTorch Official Guide](https://pytorch.org/get-started/locally/))

### API Requirements
- Running `ConsumptionTag_Risk_Assessor.py` requires an OpenAI or SiliconFlow API key (for consumption tag risk scoring)


## 🚀 Quick Start Guide
### 1. Clone the Repository and Install Dependencies
```bash
git clone https://github.com/YourUser/ConsTagFusion.git
cd ConsTagFusion
pip install -r requirements.txt
```

### 2. Download Pre-trained Models (Optional)
```bash
python utils/Download_Models.py  # Automatically downloads to the Models/ directory, including BERT, Llama-2-7B, Qwen2.5-7B
```

### 3. Configure Environment Parameters
All paths and API keys in the scripts are placeholder with empty strings (`""`) and need to be manually supplemented:
- Model storage path `MODEL_BASE_DIR`
- Data file path (e.g., `file_path`)
- Third-party API key `API_KEY` (if using the LLM scoring function)


## 📁 Directory Structure
```
ConsTagFusion/
├── Data/                           # Sample datasets
│   ├── Dataset_Samples_Tradition.xlsx
│   ├── Dataset_Samples_ConsumptionTags.xlsx
│   └── City_Levels.txt
├── Models/                         # Pre-trained models (supports custom replacement)
│   ├── bert-base-chinese/
│   ├── Llama-2-7b-hf/
│   └── Qwen2.5-7B/
├── src/                            # Core scripts
│   ├── ML_Tradition.py             # Traditional machine learning baseline models (XGBoost, etc.)
│   ├── ML_TraditionBehavior.py     # Machine learning model integrating consumption tags (chi-square weighting)
│   ├── LLM_TraditionBehavior.py    # LLM fine-tuning script (supports local deployment of BERT/Llama/Qwen)
│   ├── LLM_TraditionBehavior_PromptOptimization.py  # Prompt-engineered optimized LLM
│   └── ConsumptionTag_Risk_Assessor.py  # LLM-based consumption tag risk scoring tool
├── utils/
│   └── Download_Models.py          # Model download tool
├── requirements.txt                # Dependency list
└── README.md                       # Project documentation
```


## 📜 Core Script Function Description
| Script Name | Main Function | Key Features |
|----------------------------------------|------------------------------------------|------------------------------------------|
| `ML_Tradition.py` | Risk prediction based on traditional financial features | Includes 8 baseline models (logistic regression, random forest, etc.), supports SHAP interpretation |
| `ML_TraditionBehavior.py` | Integrates financial features and consumption tags | Uses chi-square test to screen high-value tags, implements weighted fusion to improve prediction accuracy |
| `LLM_TraditionBehavior.py` | LLM model fine-tuning and risk prediction | Supports 3 model architectures, integrates LoRA lightweight fine-tuning and 4-bit quantization to reduce memory usage |
| `LLM_TraditionBehavior_PromptOptimization.py` | Prompt-engineered optimized LLM inference | Three-stage reasoning framework (univariate analysis → interaction effect → comprehensive prediction) |
| `ConsumptionTag_Risk_Assessor.py` | Independent risk scoring of consumption tags | Asynchronous processing + caching mechanism, efficiently calculates tag risk levels and reduces API calls |


## 🧪 Experimental Reproduction Process
### 1. Traditional Machine Learning Experiments
```bash
# Train baseline models using only traditional financial features
python src/ML_Tradition.py

# Machine learning experiment integrating consumption tags
python src/ML_TraditionBehavior.py
```

### 2. LLM Model Experiments
```bash
# Basic LLM fine-tuning and evaluation
python src/LLM_TraditionBehavior.py

# Prompt engineering optimization experiment
python src/LLM_TraditionBehavior_PromptOptimization.py
```

### 3. Consumption Tag Risk Analysis
```bash
python src/ConsumptionTag_Risk_Assessor.py  # Independently evaluate the risk contribution of consumption tags
```


## 🛡️ Notes for Reproduction
- **Path Configuration**: All file paths have been anonymized and need to be manually set according to local storage structure.
- **Sample Limitations**: The 100-row sample is only for process verification; the complete dataset yields higher performance.
- **Resource Requirements**: LLM fine-tuning recommends using a GPU with ≥24GB VRAM; otherwise, adjust the batch size.
- **Compliance**: Data has been stripped of sensitive identifiers, complying with financial data privacy protection requirements, and is for academic research only.


## 📄 License
- **Code**: Licensed under the MIT License, allowing non-commercial and commercial use, with original author attribution required.
- **Data**: For academic research only; commercial use is prohibited. Usage must comply with the original bank's confidentiality agreement.

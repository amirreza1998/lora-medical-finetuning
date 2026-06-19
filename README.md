
# Parameter-Efficient Fine-Tuning for Medical Multi-Intent Detection

This repository contains the code, datasets, and experimental pipelines for the fine-tuning of multilingual Large Language Models (LLMs) specifically adapted for the healthcare domain. The project focuses on multi-intent disease query resolution in low-resource environments using Parameter-Efficient Fine-Tuning (PEFT) techniques, primarily Low-Rank Adaptation (LoRA).

## 📖 Project Overview

General-purpose LLMs often struggle with complex, domain-specific medical queries where patients express multiple intents (e.g., describing symptoms, asking for medication side effects, and requesting an appointment in a single sentence). 

This project addresses this challenge by:
* **Generating a Synthetic Dataset:** Utilizing an ensemble of LLMs (Claude 3.5 Sonnet, GPT-4, Gemini) to compile a robust medical knowledge base and synthesize realistic, multi-intent patient-physician dialogues.
* **Parameter-Efficient Fine-Tuning:** Applying 4-bit Quantized LoRA (QLoRA) to the `CohereLabs/aya-23-8B` base model, allowing for efficient training on limited hardware (e.g., 2x T4 GPUs) while maintaining high precision.
* **Memory Optimization:** Implementing advanced memory management techniques including gradient checkpointing, expandable segments, and garbage collection to prevent CUDA out-of-memory errors during training.

## 📁 Repository Structure

```text
16-LLM-RAG-FINETUNE-MEDICAL/
│
├── 1-DATABASE/                      
│   ├── disease_knowledge_base/        # Extracted disease facts, symptoms, and treatments
│   ├── intent_instruction_datasets/   # SFT datasets focusing on multi-intent conversational queries
│   └── README.md                      # Detailed documentation on dataset generation methodology
│
├── scripts/
│   ├── generate-diverse-data.ipynb    # Pipeline for generating complex, multi-symptom scenarios and varying demographics
│   ├── regenerate_diff_style.py       # Script to alter dialogue styles (e.g., formal, friendly, emergency)
│   ├── lora-fine-tune2.ipynb          # Primary fine-tuning notebook with LoRA configuration and memory optimization
│   └── improved_lora_fine_tune.py     # Enhanced training script featuring a MetricsCallback for real-time loss monitoring
│
└── requirements.txt                   # Project dependencies and environment requirements

```

## 🛠️ Key Components

### 1. Data Generation (`scripts/generate-diverse-data.ipynb`)

The synthetic data generation pipeline creates diverse conversational data by mapping extracted medical facts to specific user personas (e.g., "concerned young patient", "elderly patient with multiple questions") and conversation types (e.g., "emergency situation", "treatment discussion"). Quality control functions ensure the generated dialogues maintain high medical accuracy and natural conversational flow.

### 2. LoRA Fine-Tuning (`scripts/improved_lora_fine_tune.py`)

To train the `aya-23-8B` model on 2x T4 GPUs (approx. 30GB VRAM total), the fine-tuning script employs aggressive memory optimizations:

* **Target Modules:** Targets the attention mechanism (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
* **Quantization:** Uses `bitsandbytes` for 4-bit Normal Float (NF4) quantization with double quantization enabled.
* **Training Arguments:** Utilizes a batch size of 1 with gradient accumulation steps set to 32 (or 64 dynamically during OOM fallback) and cosine learning rate decay.

## 🚀 Quick Start & Installation

### 1. Environment Setup

Clone the repository and install the required dependencies using the `requirements.txt` file at the root of the project:

```bash
# Clone the repository
git clone <repository-url>
cd 16-LLM-RAG-FINETUNE-MEDICAL

# Create and activate a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### 2. Data Pipeline Execution

Run the data generation notebook to construct the instruction datasets from the raw knowledge base source files:

```bash
jupyter notebook scripts/generate-diverse-data.ipynb

```

*Note: Ensure you have configured your environment variables with the necessary API keys (OpenAI, Anthropic, or Gemini) before running.*

### 3. Fine-Tuning Execution

Execute the optimized training script to initiate the QLoRA training process. Make sure your Hugging Face authentication token is configured if accessing gated models:

```bash
python scripts/improved_lora_fine_tune.py

```

The final model weights will be saved into the `./final_medical_lora` directory. You can utilize the built-in inference execution blocks to run test queries against the model.

## 📝 Authors

**Amirreza Kazemloo** 
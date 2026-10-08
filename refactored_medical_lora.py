# pyrefly: ignore [missing-import]
import torch
import json
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel,
    PeftConfig
)
import gc
import os
from typing import Dict, List, Tuple
import re
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Improved configuration
class Config:
    load_dotenv()
    # No HF Token needed for Qwen 2.5!
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" # Or "Qwen/Qwen2.5-14B-Instruct" if you use A100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_LENGTH = 1024 # Increased slightly to accommodate rich medical context
    OUTPUT_DIR = "./medical_lora_output"
    FINAL_MODEL_DIR = "./final_medical_lora"

def load_base_model():
    """Enhanced model loading with better error handling for Qwen/Llama architectures"""
    print(f"Loading base model {Config.MODEL_NAME}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME, 
            trust_remote_code=True,
            padding_side="right" # Right padding is recommended for causal LM
        )
        
        # Better padding configuration
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Optimized quantization config for Colab
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, 
            bnb_4bit_quant_storage=torch.uint8 
        )

        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto", # Automatically maps to A100/L4 GPU
            trust_remote_code=True,
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True 
        )

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        return model, tokenizer
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def prepare_training_data_enhanced(data_path: str, tokenizer, validation_split: float = 0.15):
    """Data preparation utilizing tokenizer.apply_chat_template for dynamic prompt construction"""
    
    print("Preparing training data...")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    utterances = data.get("Utterance", [])
    formatted_data = []

    for item in utterances:
        query = item.get("query", "")
        diseases = item.get("diseases", [])
        
        if not query or not diseases:
            continue
            
        diseases_text = "، ".join(diseases)
        
        # Create conversational format compatible with apply_chat_template
        messages = [
            {"role": "system", "content": "شما یک دستیار پزشکی متخصص هستید که بر اساس علائم ارائه شده، بیماری‌های محتمل را تشخیص می‌دهید. پاسخ خود را به صورت دقیق و علمی ارائه دهید."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": f"با توجه به علائم ذکر شده، بیماری‌های محتمل عبارتند از: {diseases_text}\n\nبرای تشخیص دقیق‌تر، توصیه می‌شود با پزشک متخصص مشورت کنید و آزمایش‌های لازم را انجام دهید."}
        ]
        
        # This will format the text exactly how the specific model (e.g. Qwen2.5) expects it!
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        formatted_data.append({"text": formatted_text})

    df = pd.DataFrame(formatted_data)
    dataset = Dataset.from_pandas(df)

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=Config.MAX_LENGTH,
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=dataset.column_names
    )

    split_dataset = tokenized_dataset.train_test_split(
        test_size=validation_split,
        seed=42,
        shuffle=True
    )

    return split_dataset

def create_optimized_lora_config():
    """Optimized LoRA config. The target_modules fit most modern models like Qwen/Llama."""
    return LoraConfig(
        r=16,  # Increased rank for better capacity
        lora_alpha=32,  
        lora_dropout=0.05, 
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )

def train_model_enhanced(model, tokenizer, tokenized_dataset, lora_config):
    """Enhanced training with better monitoring and optimization"""
    print("Clearing GPU memory...")
    torch.cuda.empty_cache()
    gc.collect()
    
    print("Preparing LoRA model...")
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=2, # Optimized for A100/L4
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=2e-4, 
        lr_scheduler_type="cosine",
        warmup_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        fp16=False,
        bf16=True, # bfloat16 is better and fully supported on A100/L4
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=25,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        weight_decay=0.05,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training...")
    try:
        training_result = trainer.train()
        print(f"Training completed. Final loss: {training_result.training_loss:.4f}")
        
        peft_model.save_pretrained(Config.FINAL_MODEL_DIR, safe_serialization=True)
        tokenizer.save_pretrained(Config.FINAL_MODEL_DIR)
            
        return peft_model
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise

def interactive_test(model_path: str):
    """Interactive testing interface"""
    print("Loading model for interactive testing...")
    
    config = PeftConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        load_in_4bit=True,
        device_map="auto"
    )
    
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    print("\n🩺 Medical assistant is ready! (Type 'exit' to quit)")
    
    while True:
        user_input = input("\nDescribe your symptoms (or type 'exit' to quit): ").strip()
        if user_input.lower() in ['exit', 'خروج', 'quit']: break
        if not user_input: continue
        
        messages = [
            {"role": "system", "content": "شما یک دستیار پزشکی متخصص هستید که بر اساس علائم ارائه شده، بیماری‌های محتمل را تشخیص می‌دهید."},
            {"role": "user", "content": user_input}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
                top_p=0.85,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id
            )

        # Get only the generated part
        generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"\n🩺 Medical Assistant response:\n{response}")

def main():
    print("🚀 Starting medical model training")
    
    # Change this path to match your Colab structure (e.g. upload to colab root folder)
    # dataset_path = "/content/multi_intent_disease_queries(method3).json"
    dataset_path = "./database/multi_intent_disease_queries(method3).json" 
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found at {dataset_path}. Please upload the file and correct the path.")
        return

    model, tokenizer = load_base_model()
    tokenized_dataset = prepare_training_data_enhanced(dataset_path, tokenizer)
    lora_config = create_optimized_lora_config()
    
    trained_model = train_model_enhanced(model, tokenizer, tokenized_dataset, lora_config)
    print("✅ Training completed!")

    print("\n🎯 Starting interactive testing...")
    interactive_test(Config.FINAL_MODEL_DIR)

if __name__ == "__main__":
    main()

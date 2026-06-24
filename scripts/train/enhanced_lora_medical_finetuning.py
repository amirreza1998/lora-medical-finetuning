import torch
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training, 
    LoraConfig, 
    get_peft_model, 
    PeftModel, 
    PeftConfig
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import gc
import os
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")

class MedicalLoRATrainer:
    def __init__(self, model_name: str = "CohereLabs/aya-23-8B", hf_token: str = None):
        self.model_name = model_name
        self.hf_token = hf_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.training_history = []
        
        # Prevent memory fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    def load_base_model(self):
        """Load base model with quantization"""
        print(f"بارگذاری مدل {self.model_name}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True, 
            token=self.hf_token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=self.hf_token,
        )
        
        self.model = prepare_model_for_kbit_training(self.model)
        return self.model, self.tokenizer
    
    def prepare_disease_prediction_data(self, data_path: str):
        """Prepare data for disease prediction task"""
        print("آماده‌سازی داده‌های تشخیص بیماری...")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        utterances = data.get("Utterance", [])
        formatted_data = []
        
        for item in utterances:
            query = item.get("query", "")
            diseases = item.get("diseases", [])
            diseases_text = "، ".join(diseases)
            
            response = f"با توجه به علائم شما، بیماری‌های محتمل می‌توانند {diseases_text} باشد"
            
            formatted_text = f"""### سیستم:
شما یک دستیار پزشکی متخصص هستید. وظیفه شما تحلیل علائم کاربر و شناسایی بیماری‌های احتمالی است.

### کاربر:
{query}

### پاسخ:
{response}"""
            
            formatted_data.append({
                "text": formatted_text,
                "query": query,
                "diseases": diseases,
                "diseases_text": diseases_text
            })
        
        dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.15, seed=42)
        
        return tokenized_dataset
    
    def create_lora_config(self):
        """Create LoRA configuration"""
        return LoraConfig(
            r=4,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj"]
        )
    
    def train_model(self, tokenized_dataset, epochs: int = 3, output_dir: str = "./medical_lora_output"):
        """Enhanced training with better monitoring"""
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        # Create PEFT model
        lora_config = self.create_lora_config()
        self.model.gradient_checkpointing_enable()
        peft_model = get_peft_model(self.model, lora_config)
        peft_model.print_trainable_parameters()
        
        # Enhanced training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=32,
            gradient_checkpointing=True,
            save_strategy="epoch",
            logging_steps=10,
            learning_rate=2e-4,
            fp16=True,
            load_best_model_at_end=False,
            eval_strategy="steps",
            eval_steps=100,
            save_total_limit=2,
            prediction_loss_only=True,
            optim="adamw_torch",
            max_grad_norm=1.0,
            warmup_steps=50,
            lr_scheduler_type="cosine",
            report_to="none",
            logging_first_step=True,
            include_inputs_for_metrics=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )
        
        # Enhanced trainer with custom callbacks
        from transformers import TrainerCallback
        
        class EnhancedCallback(TrainerCallback):
            def __init__(self, trainer_instance):
                self.trainer_instance = trainer_instance
                
            def on_log(self, args, state, control, model=None, logs=None, **kwargs):
                if logs:
                    self.trainer_instance.training_history.append({
                        'step': state.global_step,
                        'epoch': state.epoch,
                        **logs
                    })
            
            def on_step_end(self, args, state, control, model=None, **kwargs):
                if state.global_step % 50 == 0:
                    torch.cuda.empty_cache()
        
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset.get("test", None),
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[EnhancedCallback(self)],
        )
        
        print("شروع آموزش...")
        try:
            training_result = trainer.train()
            print(f"آموزش کامل شد. Final loss: {training_result.training_loss:.4f}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("خطای کمبود حافظه! کاهش batch size...")
                torch.cuda.empty_cache()
                training_args.per_device_train_batch_size = 1
                training_args.gradient_accumulation_steps = 64
                trainer = Trainer(
                    model=peft_model,
                    args=training_args,
                    train_dataset=tokenized_dataset["train"],
                    eval_dataset=tokenized_dataset.get("test", None),
                    tokenizer=self.tokenizer,
                    data_collator=data_collator,
                )
                training_result = trainer.train()
            else:
                raise e
        
        # Save model
        print("ذخیره مدل...")
        peft_model.save_pretrained("./final_medical_lora", safe_serialization=True)
        self.tokenizer.save_pretrained("./final_medical_lora")
        
        # Save training history
        with open(f"{output_dir}/training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        return peft_model, trainer.state.log_history
    
    def plot_training_metrics(self, save_path: str = "./training_plots"):
        """Create comprehensive training visualizations"""
        if not self.training_history:
            print("هیچ تاریخچه آموزشی یافت نشد!")
            return
        
        os.makedirs(save_path, exist_ok=True)
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(self.training_history)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training Metrics Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Training Loss
        if 'train_loss' in df.columns:
            axes[0, 0].plot(df['step'], df['train_loss'], 'b-', linewidth=2, label='Training Loss')
            axes[0, 0].set_title('Training Loss Over Steps')
            axes[0, 0].set_xlabel('Steps')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
        
        # 2. Evaluation Loss
        if 'eval_loss' in df.columns:
            eval_df = df.dropna(subset=['eval_loss'])
            if not eval_df.empty:
                axes[0, 1].plot(eval_df['step'], eval_df['eval_loss'], 'r-', linewidth=2, label='Validation Loss')
                axes[0, 1].set_title('Validation Loss Over Steps')
                axes[0, 1].set_xlabel('Steps')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].legend()
        
        # 3. Learning Rate
        if 'learning_rate' in df.columns:
            axes[1, 0].plot(df['step'], df['learning_rate'], 'g-', linewidth=2, label='Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Steps')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
        
        # 4. Epoch Progress
        if 'epoch' in df.columns:
            axes[1, 1].plot(df['step'], df['epoch'], 'purple', linewidth=2, label='Epoch')
            axes[1, 1].set_title('Epoch Progress')
            axes[1, 1].set_xlabel('Steps')
            axes[1, 1].set_ylabel('Epoch')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/training_metrics.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create loss comparison plot
        plt.figure(figsize=(12, 6))
        if 'train_loss' in df.columns:
            plt.plot(df['step'], df['train_loss'], 'b-', linewidth=2, label='Training Loss', alpha=0.8)
        
        if 'eval_loss' in df.columns:
            eval_df = df.dropna(subset=['eval_loss'])
            if not eval_df.empty:
                plt.plot(eval_df['step'], eval_df['eval_loss'], 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
        
        plt.title('Training vs Validation Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_path}/loss_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def comprehensive_model_evaluation(self, model_path: str, test_data_path: str):
        """Comprehensive model evaluation with metrics"""
        print("شروع ارزیابی جامع مدل...")
        
        # Load the fine-tuned model
        config = PeftConfig.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            load_in_8bit=True,
            device_map="auto",
            token=self.hf_token
        )
        
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = PeftModel.from_pretrained(base_model, model_path)
        
        # Load test data
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        test_utterances = test_data.get("Utterance", [])
        
        # Evaluation metrics
        results = {
            'predictions': [],
            'ground_truth': [],
            'exact_matches': [],
            'partial_matches': [],
            'response_quality_scores': []
        }
        
        print("ارزیابی پاسخ‌های مدل...")
        for i, item in enumerate(tqdm(test_utterances[:50])):  # Test on first 50 samples
            query = item.get("query", "")
            expected_diseases = set(item.get("diseases", []))
            
            # Generate prediction
            prediction = self._generate_prediction(model, tokenizer, query)
            
            # Extract predicted diseases
            predicted_diseases = self._extract_diseases_from_response(prediction)
            
            # Calculate metrics
            exact_match = predicted_diseases == expected_diseases
            partial_match = len(predicted_diseases.intersection(expected_diseases)) > 0
            
            # Response quality score (simple heuristic)
            quality_score = self._calculate_response_quality(prediction, query)
            
            results['predictions'].append(list(predicted_diseases))
            results['ground_truth'].append(list(expected_diseases))
            results['exact_matches'].append(exact_match)
            results['partial_matches'].append(partial_match)
            results['response_quality_scores'].append(quality_score)
        
        # Calculate overall metrics
        exact_match_accuracy = np.mean(results['exact_matches'])
        partial_match_accuracy = np.mean(results['partial_matches'])
        avg_quality_score = np.mean(results['response_quality_scores'])
        
        # Disease-level precision, recall, F1
        all_diseases = set()
        for diseases in results['ground_truth'] + results['predictions']:
            all_diseases.update(diseases)
        
        disease_metrics = {}
        for disease in all_diseases:
            y_true = [disease in gt for gt in results['ground_truth']]
            y_pred = [disease in pred for pred in results['predictions']]
            
            if sum(y_true) > 0:  # Only calculate if disease appears in ground truth
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average='binary', zero_division=0
                )
                disease_metrics[disease] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
        
        # Create evaluation report
        report = {
            'overall_metrics': {
                'exact_match_accuracy': exact_match_accuracy,
                'partial_match_accuracy': partial_match_accuracy,
                'average_response_quality': avg_quality_score,
                'total_samples_evaluated': len(results['exact_matches'])
            },
            'disease_level_metrics': disease_metrics,
            'detailed_results': results
        }
        
        # Save evaluation results
        with open("./evaluation_report.json", "w", encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("📊 خلاصه نتایج ارزیابی:")
        print("="*50)
        print(f"دقت تطبیق کامل: {exact_match_accuracy:.2%}")
        print(f"دقت تطبیق جزئی: {partial_match_accuracy:.2%}")
        print(f"میانگین کیفیت پاسخ: {avg_quality_score:.2f}/5.0")
        print(f"تعداد نمونه‌های ارزیابی شده: {len(results['exact_matches'])}")
        
        # Plot evaluation metrics
        self._plot_evaluation_results(report)
        
        return report
    
    def _generate_prediction(self, model, tokenizer, query: str) -> str:
        """Generate prediction for a single query"""
        prompt = f"""### سیستم:
شما یک دستیار پزشکی متخصص هستید. وظیفه شما تحلیل علائم کاربر و شناسایی بیماری‌های احتمالی است.

### کاربر:
{query}

### پاسخ:
"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split('### پاسخ:')[1].strip() if '### پاسخ:' in response else response
    
    def _extract_diseases_from_response(self, response: str) -> set:
        """Extract disease names from model response"""
        # Simple extraction - you might want to make this more sophisticated
        diseases = set()
        
        # Common disease terms in Persian (extend this list)
        disease_keywords = [
            'افسردگی', 'کم‌خونی', 'هیپوتیروئیدیسم', 'سرطان', 'تب', 'پنومونی',
            'بیماری قلبی', 'فشار خون', 'دیابت', 'آسم', 'کبد', 'کلیه'
        ]
        
        response_lower = response.lower()
        for keyword in disease_keywords:
            if keyword in response_lower:
                diseases.add(keyword)
        
        return diseases
    
    def _calculate_response_quality(self, response: str, query: str) -> float:
        """Calculate response quality score (1-5 scale)"""
        score = 1.0
        
        # Length check
        if 50 <= len(response) <= 500:
            score += 1.0
        
        # Relevance check (simple keyword matching)
        if any(word in response.lower() for word in ['بیماری', 'علائم', 'تشخیص']):
            score += 1.0
        
        # Medical terminology
        medical_terms = ['پزشکی', 'درمان', 'تشخیص', 'علائم', 'بیماری']
        if sum(term in response.lower() for term in medical_terms) >= 2:
            score += 1.0
        
        # Response completeness
        if 'احتمال' in response.lower() or 'ممکن' in response.lower():
            score += 1.0
        
        return min(score, 5.0)
    
    def _plot_evaluation_results(self, report: dict):
        """Plot evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # Overall metrics
        metrics = report['overall_metrics']
        metric_names = ['Exact Match', 'Partial Match', 'Response Quality']
        metric_values = [
            metrics['exact_match_accuracy'],
            metrics['partial_match_accuracy'],
            metrics['average_response_quality'] / 5.0  # Normalize to 0-1
        ]
        
        axes[0, 0].bar(metric_names, metric_values, color=['#ff7f0e', '#2ca02c', '#1f77b4'])
        axes[0, 0].set_title('Overall Performance Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_ylim(0, 1)
        
        # Disease-level F1 scores
        disease_metrics = report['disease_level_metrics']
        if disease_metrics:
            diseases = list(disease_metrics.keys())[:10]  # Top 10 diseases
            f1_scores = [disease_metrics[d]['f1'] for d in diseases]
            
            axes[0, 1].barh(diseases, f1_scores, color='skyblue')
            axes[0, 1].set_title('Disease-level F1 Scores')
            axes[0, 1].set_xlabel('F1 Score')
        
        # Prediction distribution
        results = report['detailed_results']
        match_distribution = {
            'Exact Match': sum(results['exact_matches']),
            'Partial Match Only': sum(results['partial_matches']) - sum(results['exact_matches']),
            'No Match': len(results['exact_matches']) - sum(results['partial_matches'])
        }
        
        axes[1, 0].pie(match_distribution.values(), labels=match_distribution.keys(), autopct='%1.1f%%')
        axes[1, 0].set_title('Prediction Accuracy Distribution')
        
        # Quality score distribution
        axes[1, 1].hist(results['response_quality_scores'], bins=10, color='lightgreen', alpha=0.7)
        axes[1, 1].set_title('Response Quality Score Distribution')
        axes[1, 1].set_xlabel('Quality Score')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('./evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()

# Example usage
# def main():
# Initialize trainer
trainer = MedicalLoRATrainer(
    model_name="CohereLabs/aya-23-8B",
    hf_token="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"
)

# Load model
model, tokenizer = trainer.load_base_model()

# Prepare data
dataset_path = r"D:\kazemloo\nlu_final_project\nlu_final_project\intend_database\multi_intent_disease_queries(method3).json"
tokenized_dataset = trainer.prepare_disease_prediction_data(dataset_path)

# Train model
trained_model, training_log = trainer.train_model(tokenized_dataset, epochs=3)

# Plot training metrics
trainer.plot_training_metrics()

# Comprehensive evaluation
evaluation_report = trainer.comprehensive_model_evaluation(
    "./final_medical_lora",
    dataset_path
)

print("آموزش و ارزیابی کامل شد!")
import pdb;pdb.set_trace
# if __name__ == "__main__":
#     main()
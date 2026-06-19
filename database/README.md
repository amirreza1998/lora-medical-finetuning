# Medical Multi-Intent Query Dataset

This directory contains the datasets utilized for the parameter-efficient fine-tuning (PEFT) of Large Language Models (LLMs) for multi-intent disease query resolution. 

Our data generation pipeline is fully synthetic, leveraging multiple state-of-the-art LLMs to first construct a comprehensive medical knowledge base, and subsequently synthesize complex, multi-intent patient-physician dialogues.

## Directory Structure

```text
DATABASE/
├── disease_knowledge_base/          # Raw disease facts, symptoms, and treatments
│   ├── gpt25pro_disease.json        # Extractions from Gemini/GPT-Pro variants
│   ├── gpt41_disease.json           # Extractions from GPT-4 variants
│   ├── sonnet4_disease.json         # Extractions from Claude 3.5 Sonnet
│   ├── treatment_and_medicine.json  # Supplemental therapeutic data
│   ├── merged_diseases.json         # Initial deduplicated knowledge graph
│   └── merged_diseases_extended.json# Final consolidated ground-truth knowledge base
│
└── intent_instruction_datasets/     # Synthesized conversational SFT datasets
    ├── conversation_based_intend_database(method1).json
    ├── intend_database(method2).json
    └── multi_intent_disease_queries(method3).json # Primary fine-tuning target
import json
import random
from typing import List, Dict

class MedicalPromptGenerator:
    def __init__(self, json_file_path: str):
        """
        Initialize the prompt generator with disease data
        
        Args:
            json_file_path: Path to the JSON file containing disease data
        """
        self.json_file_path = json_file_path
        self.diseases_data = self.load_diseases_data()
        
        self.user_personas = [
            "بیمار جوان نگران",
            "والدین کودک بیمار", 
            "سالمند با سوالات متعدد",
            "بیمار با سطح تحصیلات بالا",
            "بیمار با دانش محدود پزشکی"
        ]
        
        self.conversation_types = [
            "patient_initial_complaint",     # شکایت اولیه بیمار
            "symptom_inquiry",               # پرسش درباره علائم
            "diagnosis_explanation",         # توضیح تشخیص
            "treatment_discussion",          # بحث درباره درمان
            "follow_up_questions",           # سوالات پیگیری
            "prevention_advice",             # مشاوره پیشگیری
            "emergency_situations"           # شرایط اورژانسی
        ]
        
        # Persian translations for conversation types
        self.conversation_types_persian = {
            "patient_initial_complaint": "شکایت اولیه بیمار",
            "symptom_inquiry": "پرسش درباره علائم",
            "diagnosis_explanation": "توضیح تشخیص",
            "treatment_discussion": "بحث درباره درمان",
            "follow_up_questions": "سوالات پیگیری",
            "prevention_advice": "مشاوره پیشگیری",
            "emergency_situations": "شرایط اورژانسی"
        }
        self.conversation_styles = [
            "formal_medical",      
            "friendly_doctor",     
            "emergency_urgent",   
            "educational",         
            "reassuring"          
        ]
        self.conversation_styles_persian = [
            "formal_medical": "رسمی پزشکی",
            "friendly_doctor": "دوستانه",
            "emergency_urgent": "اورژانسی",
            "educational": "آموزشی",
            "reassuring": "دلگرم‌کننده",
        ]        


    def load_diseases_data(self) -> List[Dict]:
        """
        Load disease data from JSON file
        
        Returns:
            List of disease dictionaries
        """
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                print(f"✅ Successfully loaded {len(data)} diseases from {self.json_file_path}")
                return data['diseases']
        except FileNotFoundError:
            print(f"❌ Error: File '{self.json_file_path}' not found!")
            return []
        except json.JSONDecodeError:
            print(f"❌ Error: Invalid JSON format in '{self.json_file_path}'!")
            return []
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return []

    def generate_prompt(self, disease_data: Dict = None, persona: str = None, conv_type: str = None) -> str:
        """
        Generate a medical conversation prompt
        
        Args:
            disease_data: Specific disease data (if None, random selection)
            persona: Specific user persona (if None, random selection)
            conv_type: Specific conversation type (if None, random selection)
            
        Returns:
            Formatted prompt string
        """
        if not self.diseases_data:
            return "No disease data available. Please check the JSON file."
        
        # Random selections if not specified
        if disease_data is None:
            disease_data = random.choice(self.diseases_data)
        
        if persona is None:
            persona = random.choice(self.user_personas)
            
        if conv_type is None:
            conv_type = random.choice(self.conversation_types)

        if conv_style is None:
            conv_style = random.choice(self.conversation_style)

        # Convert conversation type to Persian
        conv_type_persian = self.conversation_types_persian.get(conv_type, conv_type)
        # Convert conversation style to Persian
        conv_style_persian = self.conversation_styles_persian.get(conv_style, conv_style)

        # Format disease data nicely
        disease_info = json.dumps(disease_data, ensure_ascii=False, indent=2)
        
        # Create the prompt template
        prompt = f"""بر اساس اطلاعات بیماری زیر:
{disease_info}

یک مکالمه طبیعی و واقعی بین پزشک و {persona} تولید کن.
نوع مکالمه: {conv_type_persian}

قوانین مهم:
1. مکالمه باید کاملاً طبیعی باشد
2. پزشک باید حرفه‌ای و دلسوزانه پاسخ دهد
3. اطلاعات پزشکی دقیق باشد
4. زبان ساده و قابل فهم استفاده شود
5. سوالات متنوع و واقعی باشد

فرمت خروجی:
بیمار: [سوال/شکایت]
پزشک: [پاسخ تخصصی]
بیمار: [سوال پیگیری]
پزشک: [پاسخ]
..."""
        
        return prompt

    def generate_multiple_prompts(self, count: int = 5) -> List[Dict]:
        """
        Generate multiple random prompts
        
        Args:
            count: Number of prompts to generate
            
        Returns:
            List of dictionaries containing prompt info
        """
        prompts = []
        
        for i in range(count):
            disease = random.choice(self.diseases_data) if self.diseases_data else None
            persona = random.choice(self.user_personas)
            conv_type = random.choice(self.conversation_types)
            
            prompt = self.generate_prompt(disease, persona, conv_type)
            
            prompts.append({
                "prompt_number": i + 1,
                "disease_name": disease.get("name_persian")+ " " +disease.get("name_english") if disease else "No disease data",
                "persona": persona,
                "conversation_type": conv_type,
                "conversation_type_persian": self.conversation_types_persian.get(conv_type, conv_type),
                "prompt": prompt
            })
        
        return prompts

    def save_prompts_to_file(self, prompts: List[Dict], output_file: str = "generated_prompts.txt"):
        """
        Save generated prompts to a text file
        
        Args:
            prompts: List of prompt dictionaries
            output_file: Output file name
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write("=" * 80 + "\n")
                file.write("GENERATED MEDICAL CONVERSATION PROMPTS\n")
                file.write(f"Generated on: 2025-08-13 08:36:27 UTC\n")
                file.write(f"Total prompts: {len(prompts)}\n")
                file.write("=" * 80 + "\n\n")
                
                for prompt_info in prompts:
                    file.write(f"PROMPT #{prompt_info['prompt_number']}\n")
                    file.write(f"Disease: {prompt_info['disease_name']}\n")
                    file.write(f"Persona: {prompt_info['persona']}\n")
                    file.write(f"Conversation Type: {prompt_info['conversation_type_persian']}\n")
                    file.write("-" * 50 + "\n")
                    file.write(prompt_info['prompt'])
                    file.write("\n\n" + "=" * 80 + "\n\n")
            
            print(f"✅ Prompts saved to '{output_file}'")
        except Exception as e:
            print(f"❌ Error saving prompts: {e}")

    def display_available_data(self):
        """Display information about loaded data"""
        print("\n📊 AVAILABLE DATA:")
        print(f"Diseases loaded: {len(self.diseases_data)}")
        print(f"User personas: {len(self.user_personas)}")
        print(f"Conversation types: {len(self.conversation_types)}")
        
        if self.diseases_data:
            print(f"\nFirst few diseases:")
            for i, disease in enumerate(self.diseases_data[:3]):
                disease_name = disease.get("name", disease.get("disease_name", "Unknown"))
                print(f"  {i+1}. {disease_name}")
            if len(self.diseases_data) > 3:
                print(f"  ... and {len(self.diseases_data) - 3} more")


def main():
    """Main function to demonstrate usage"""
    print("🏥 Medical Conversation Prompt Generator")
    print("=" * 50)
    
    # Initialize the generator
    generator = MedicalPromptGenerator("./diseases-json-file/merged_diseases.json")
    # Display available data
    generator.display_available_data()

    if not generator.diseases_data:
        print("\n❌ Cannot generate prompts without disease data. Please check the JSON file.")
        return
    
    print("\n🎲 Generating random prompts...")
    
    # Generate a single random prompt
    print("\n" + "="*50)
    print("SINGLE RANDOM PROMPT:")
    print("="*50)
    single_prompt = generator.generate_prompt()
    print(single_prompt)
    
    # Generate multiple prompts
    print(f"\n{'='*50}")
    print("GENERATING MULTIPLE PROMPTS:")
    print("="*50)
    
    prompts = generator.generate_multiple_prompts(count=100)
    
    # Display the prompts
    for prompt_info in prompts:
        print(f"\nPROMPT #{prompt_info['prompt_number']}")
        print(f"Disease: {prompt_info['disease_name']}")
        print(f"Persona: {prompt_info['persona']}")
        print(f"Type: {prompt_info['conversation_type_persian']}")
        print("-" * 30)
        print(prompt_info['prompt'])
        print("\n" + "="*50)
    
    # Save prompts to file
    generator.save_prompts_to_file(prompts, "medical_conversation_prompts.txt")
    
    print(f"\n✅ Process completed! Generated {len(prompts)} prompts.")


if __name__ == "__main__":
    main()
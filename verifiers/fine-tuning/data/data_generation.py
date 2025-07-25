import openai
import json
import random
import time
from typing import List, Dict, Any
import os
from datetime import datetime

class ConfiguratorDatasetGenerator:
    def __init__(self, api_key: str):
        """Initialize the dataset generator with OpenAI API key."""
        self.client = openai.OpenAI(api_key=api_key)
        self.scenarios = self._define_scenarios()
        
    def _define_scenarios(self) -> List[Dict[str, Any]]:
        """Define various configurator scenarios with context and examples."""
        return [
            {
                "domain": "Technology",
                "categories": ["laptops", "smartphones", "tablets", "desktop computers", "gaming PCs", "servers", "network equipment"],
                "use_cases": ["gaming", "professional work", "content creation", "programming", "data science", "3D modeling", "video editing", "office work"],
                "sample_attributes": ["performance", "memory", "storage", "display", "portability", "battery life", "price range"]
            },
            {
                "domain": "Beverages",
                "categories": ["coffee", "tea", "smoothies", "cocktails", "energy drinks", "protein shakes", "juices"],
                "use_cases": ["morning energy", "post-workout", "relaxation", "social gathering", "health conscious", "caffeine boost", "meal replacement"],
                "sample_attributes": ["caffeine content", "flavor profile", "nutritional value", "temperature", "sweetness level", "organic options"]
            },
            {
                "domain": "Textiles",
                "categories": ["clothing", "bedding", "curtains", "upholstery", "carpets", "towels", "outdoor fabrics"],
                "use_cases": ["formal wear", "casual comfort", "athletic performance", "home decoration", "weather protection", "durability", "luxury comfort"],
                "sample_attributes": ["material type", "durability", "comfort", "style", "color", "maintenance", "price point"]
            },
            {
                "domain": "Automotive",
                "categories": ["sedans", "SUVs", "trucks", "electric vehicles", "motorcycles", "commercial vehicles"],
                "use_cases": ["daily commuting", "family trips", "off-road adventures", "cargo transport", "fuel efficiency", "luxury travel"],
                "sample_attributes": ["fuel efficiency", "safety features", "cargo space", "performance", "technology features", "price range"]
            },
            {
                "domain": "Fitness Equipment",
                "categories": ["cardio machines", "strength equipment", "yoga accessories", "home gym setups", "wearable tech"],
                "use_cases": ["weight loss", "muscle building", "endurance training", "rehabilitation", "home workouts", "professional training"],
                "sample_attributes": ["space requirements", "intensity levels", "tracking features", "durability", "ease of use", "budget"]
            },
            {
                "domain": "Kitchen Appliances",
                "categories": ["blenders", "coffee makers", "air fryers", "stand mixers", "food processors", "pressure cookers"],
                "use_cases": ["meal prep", "healthy cooking", "baking", "quick meals", "large family cooking", "gourmet cooking"],
                "sample_attributes": ["capacity", "power", "versatility", "ease of cleaning", "counter space", "energy efficiency"]
            }
        ]

    def _generate_system_prompt(self) -> str:
        """Generate the system prompt for the LLM."""
        return """You are a helpful assistant that generates training data for a configurator AI model. 

Your task is to create a realistic answer that follows this EXACT format:
1. Start with reasoning inside <think> tags about the user's needs
2. Make a search() tool call inside <tool> tags with relevant parameters
3. Make a validate() tool call inside <tool> tags with validation parameters
4. Provide the final recommendation inside <answer> tags as a JSON object with exactly 3 fields: Product, Features, Price

The response should be concise, practical, and directly address the user's request. Do not include unnecessary information or explanations outside of the specified tags.

IMPORTANT: The <answer> section must contain ONLY a valid JSON object with these exact fields:
- "Product": A real product name that exists in the current market
- "Features": An array of 3-5 key features that justify the recommendation
- "Price": A realistic price in USD format (e.g., "$1,299")

Example format:
<think>
[Brief reasoning about user needs and requirements]
</think>

<tool>
search(category="[category]", requirements={"[key1]": "[value1]", "[key2]": "[value2]"}, filters={"[filter1]": "[value1]"})
</tool>

<tool>
validate(product_id="[id]", user_requirements={"[req1]": "[value1]", "[req2]": "[value2]"}, compatibility_check=true)
</tool>

<answer>
{
  "Product": "Real Product Name Model XYZ",
  "Features": ["Feature 1 that meets user needs", "Feature 2 for specific requirement", "Feature 3 for performance", "Feature 4 for value"],
  "Price": "$X,XXX"
}
</answer>"""

    def _generate_user_prompt(self, scenario: Dict[str, Any]) -> str:
        """Generate a user prompt for a specific scenario."""
        category = random.choice(scenario["categories"])
        use_case = random.choice(scenario["use_cases"])
        attributes = random.sample(scenario["sample_attributes"], random.randint(2, 4))
        
        # Create varied question templates
        templates = [
            f"I need a {category} for {use_case}, focusing on {' and '.join(attributes)}",
            f"Can you help me find a {category} that's good for {use_case}? I care about {', '.join(attributes)}",
            f"Looking for a {category} suitable for {use_case}. Priority on {' and '.join(attributes)}",
            f"I want a {category} for {use_case} with excellent {' and '.join(attributes)}",
            f"Recommend a {category} for {use_case}. Important features: {', '.join(attributes)}",
            f"Help me choose a {category} for {use_case}. Key requirements: {' and '.join(attributes)}",
        ]
        
        return random.choice(templates)

    def _generate_single_qa_pair(self, scenario: Dict[str, Any]) -> Dict[str, str]:
        """Generate a single question-answer pair."""
        question = self._generate_user_prompt(scenario)
        
        user_message = f"""Generate a configurator response for this user request: "{question}"

The context is a {scenario['domain'].lower()} configurator tool. Follow the exact format with <think>, two <tool> calls (search then validate), and <answer> tags.

CRITICAL: The <answer> section must contain ONLY a valid JSON object with exactly these 3 fields:
- "Product": Use a real product name that exists in the current market and fits the user's needs
- "Features": Array of 3-5 specific features that justify why this product meets the requirements
- "Price": Realistic market price in USD format (e.g., "$1,299")

Keep the response focused and practical - no unnecessary explanations outside the required tags."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self._generate_system_prompt()},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.8,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Validate the response format
            if self._validate_response_format(answer):
                return {"question": question, "answer": answer}
            else:
                print(f"Warning: Response format validation failed for question: {question[:50]}...")
                return None
                
        except Exception as e:
            print(f"Error generating QA pair: {e}")
            return None

    def _validate_response_format(self, response: str) -> bool:
        """Validate that the response follows the required format."""
        required_tags = ["<think>", "</think>", "<tool>", "</tool>", "<answer>", "</answer>"]
        
        # Check if all required tags are present
        for tag in required_tags:
            if tag not in response:
                return False
        
        # Check if there are exactly 2 tool calls
        tool_count = response.count("<tool>")
        if tool_count != 2:
            return False
        
        # Extract and validate JSON in answer section
        try:
            answer_start = response.find("<answer>") + 8
            answer_end = response.find("</answer>")
            if answer_start == 7 or answer_end == -1:  # <answer> not found
                return False
                
            answer_content = response[answer_start:answer_end].strip()
            
            # Parse JSON and validate required fields
            answer_json = json.loads(answer_content)
            required_fields = ["Product", "Features", "Price"]
            
            for field in required_fields:
                if field not in answer_json:
                    return False
            
            # Validate Features is a list
            if not isinstance(answer_json["Features"], list):
                return False
            
            # Validate Features has 3-5 items
            if len(answer_json["Features"]) < 3 or len(answer_json["Features"]) > 5:
                return False
                
            return True
            
        except (json.JSONDecodeError, ValueError, KeyError):
            return False

    def generate_dataset(self, target_count: int = 1000, output_file: str = None) -> List[Dict[str, str]]:
        """Generate the complete dataset."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"configurator_dataset_{timestamp}.json"
        
        dataset = []
        failed_attempts = 0
        max_failed_attempts = 100
        
        print(f"Starting generation of {target_count} QA pairs...")
        
        while len(dataset) < target_count and failed_attempts < max_failed_attempts:
            # Select random scenario
            scenario = random.choice(self.scenarios)
            
            # Generate QA pair
            qa_pair = self._generate_single_qa_pair(scenario)
            
            if qa_pair:
                dataset.append(qa_pair)
                
                # Progress update
                if len(dataset) % 50 == 0:
                    print(f"Generated {len(dataset)}/{target_count} QA pairs")
                    
                # Save intermediate results every 100 pairs
                if len(dataset) % 100 == 0:
                    self._save_dataset(dataset, f"temp_{output_file}")
                    
            else:
                failed_attempts += 1
                
            # Rate limiting to avoid API limits
            time.sleep(0.1)
        
        if failed_attempts >= max_failed_attempts:
            print(f"Warning: Stopped generation due to too many failed attempts. Generated {len(dataset)} pairs.")
        
        # Save final dataset
        self._save_dataset(dataset, output_file)
        print(f"Dataset generation complete! Saved {len(dataset)} QA pairs to {output_file}")
        
        return dataset

    def _save_dataset(self, dataset: List[Dict[str, str]], filename: str):
        """Save dataset to JSON file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving dataset: {e}")

    def analyze_dataset(self, dataset: List[Dict[str, str]]):
        """Analyze the generated dataset for quality metrics."""
        if not dataset:
            print("No dataset to analyze")
            return
            
        print(f"\n=== Dataset Analysis ===")
        print(f"Total QA pairs: {len(dataset)}")
        
        # Analyze question patterns
        question_lengths = [len(qa['question'].split()) for qa in dataset]
        print(f"Average question length: {sum(question_lengths) / len(question_lengths):.1f} words")
        print(f"Question length range: {min(question_lengths)} - {max(question_lengths)} words")
        
        # Analyze answer patterns
        answer_lengths = [len(qa['answer'].split()) for qa in dataset]
        print(f"Average answer length: {sum(answer_lengths) / len(answer_lengths):.1f} words")
        print(f"Answer length range: {min(answer_lengths)} - {max(answer_lengths)} words")
        
        # Check format compliance
        format_compliant = sum(1 for qa in dataset if self._validate_response_format(qa['answer']))
        print(f"Format compliance: {format_compliant}/{len(dataset)} ({format_compliant/len(dataset)*100:.1f}%)")

def main():
    """Main function to run the dataset generator."""
    # Get API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        api_key = input("Please enter your OpenAI API key: ")
    
    # Initialize generator
    generator = ConfiguratorDatasetGenerator(api_key)
    
    # Generate dataset
    try:
        dataset = generator.generate_dataset(target_count=500)
        
        # Analyze the generated dataset
        generator.analyze_dataset(dataset)
        
        print("\nDataset generation completed successfully!")
        print("The generated dataset is ready for fine-tuning your configurator model.")
        
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user.")
    except Exception as e:
        print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()
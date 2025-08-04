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
        """Define enterprise technology scenarios with context and examples."""
        return [
            {
                "domain": "Enterprise Computing",
                "categories": ["workstations", "servers", "storage systems", "networking equipment", "data center hardware", "cloud infrastructure"],
                "use_cases": ["software development", "data analysis", "machine learning", "database management", "virtualization", "high-performance computing", "enterprise applications", "disaster recovery"],
                "sample_attributes": ["processing power", "memory capacity", "storage speed", "reliability", "scalability", "security features", "support level"]
            },
            {
                "domain": "Enterprise Software",
                "categories": ["ERP systems", "CRM platforms", "business intelligence tools", "collaboration software", "security solutions", "cloud services"],
                "use_cases": ["financial management", "customer relationship management", "data analytics", "team collaboration", "cybersecurity", "digital transformation", "compliance", "process automation"],
                "sample_attributes": ["user capacity", "integration capabilities", "customization options", "security compliance", "deployment model", "support tier"]
            },
            {
                "domain": "Enterprise Networking",
                "categories": ["switches", "routers", "firewalls", "wireless access points", "network security", "SD-WAN solutions"],
                "use_cases": ["network infrastructure", "branch connectivity", "data center networking", "remote work support", "network security", "traffic management", "load balancing", "network monitoring"],
                "sample_attributes": ["port density", "throughput capacity", "security features", "management capabilities", "redundancy", "scalability"]
            },
            {
                "domain": "Enterprise Security",
                "categories": ["identity management", "endpoint protection", "network security", "data loss prevention", "SIEM systems", "threat intelligence"],
                "use_cases": ["access control", "threat detection", "compliance management", "incident response", "vulnerability management", "security monitoring", "data protection", "risk assessment"],
                "sample_attributes": ["threat detection rate", "false positive rate", "integration capabilities", "compliance standards", "deployment complexity", "maintenance requirements"]
            }
        ]

    def _generate_system_prompt(self) -> str:
        """Generate the system prompt for the LLM."""
        return """You are a helpful assistant that generates training data for a CPQ (Configure, Price, Quote) AI model. 

Your task is to create a realistic answer that follows this EXACT format:
1. Start with reasoning inside <think> tags about the user's needs
2. Make a search_product tool call inside <tool> tags with the user's query as the argument
3. Provide a summary of what you did inside <answer> tags

The response should be concise, practical, and directly address the user's request. Do not include unnecessary information or explanations outside of the specified tags.

IMPORTANT: The <answer> section should contain a brief summary of your reasoning and tool call, not a JSON object.

Example format:
<think>
[Brief reasoning about user needs and requirements]
</think>

<tool>
{"name": "search_product", "args": {"query": "user query in natural language"}}
</tool>

<answer>
[Brief summary of what you did - reasoning about user needs and making a tool call to search for products]
</answer>"""

    def _generate_user_prompt(self, scenario: Dict[str, Any]) -> str:
        """Generate a user prompt for a specific scenario."""
        category = random.choice(scenario["categories"])
        use_case = random.choice(scenario["use_cases"])
        attributes = random.sample(scenario["sample_attributes"], random.randint(2, 4))
        
        # Create varied question templates for product search
        templates = [
            f"I need a {category} for {use_case}, focusing on {' and '.join(attributes)}",
            f"Can you help me find a {category} that's good for {use_case}? I care about {', '.join(attributes)}",
            f"Looking for a {category} suitable for {use_case}. Priority on {' and '.join(attributes)}",
            f"I want a {category} for {use_case} with excellent {' and '.join(attributes)}",
            f"Recommend a {category} for {use_case}. Important features: {', '.join(attributes)}",
            f"Help me choose a {category} for {use_case}. Key requirements: {' and '.join(attributes)}",
            f"Search for a {category} that works well for {use_case}, especially {' and '.join(attributes)}",
            f"Find me a {category} for {use_case} with good {' and '.join(attributes)}",
            f"We're evaluating {category} solutions for {use_case}. Key criteria: {', '.join(attributes)}",
            f"Our enterprise needs a {category} for {use_case}. Must have: {', '.join(attributes)}",
            f"Looking for enterprise-grade {category} for {use_case}. Requirements: {', '.join(attributes)}",
            f"Need to source {category} for {use_case} deployment. Critical factors: {', '.join(attributes)}",
        ]
        
        return random.choice(templates)

    def _generate_single_qa_pair(self, scenario: Dict[str, Any]) -> Dict[str, str]:
        """Generate a single question-answer pair."""
        question = self._generate_user_prompt(scenario)
        
        user_message = f"""Generate a CPQ response for this user request: "{question}"

The context is a {scenario['domain'].lower()} product search tool. Follow the exact format with <think>, <tool>, and <answer> tags.

CRITICAL: 
- Use the search_product tool with the user's query as the argument
- The <answer> section should contain a brief summary of your reasoning and tool call
- Keep the response focused and practical - no unnecessary explanations outside the required tags
- Consider enterprise-grade requirements and business needs"""

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
        
        # Check if there is exactly 1 tool call
        tool_count = response.count("<tool>")
        if tool_count != 1:
            return False
        
        # Check if the tool call is for search_product
        try:
            tool_start = response.find("<tool>") + 6
            tool_end = response.find("</tool>")
            if tool_start == 5 or tool_end == -1:  # <tool> not found
                return False
                
            tool_content = response[tool_start:tool_end].strip()
            tool_json = json.loads(tool_content)
            
            # Validate it's search_product tool
            if tool_json.get("name") != "search_product":
                return False
            
            # Validate it has args with query
            if "args" not in tool_json or "query" not in tool_json["args"]:
                return False
            
            # Validate query is a non-empty string
            if not isinstance(tool_json["args"]["query"], str) or len(tool_json["args"]["query"].strip()) == 0:
                return False
                
            return True
            
        except (json.JSONDecodeError, ValueError, KeyError):
            return False

    def generate_dataset(self, target_count: int = 10, output_file: str = None) -> List[Dict[str, str]]:
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
        
        # Analyze tool call patterns
        search_product_calls = 0
        for qa in dataset:
            if 'search_product' in qa['answer']:
                search_product_calls += 1
        print(f"search_product tool calls: {search_product_calls}/{len(dataset)} ({search_product_calls/len(dataset)*100:.1f}%)")

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
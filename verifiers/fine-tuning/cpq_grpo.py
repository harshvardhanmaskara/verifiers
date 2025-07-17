import verifiers as vf
import re
import json
from datasets import load_dataset, Dataset

# Load your catalog data
with open('verifiers/fine-tuning/extracted_catalog.json', 'r') as file:
    json_data = json.load(file)

json_string = json.dumps(json_data, indent=2)

# THINKING MODEL SYSTEM PROMPT - Internal reasoning, clean output
SYSTEM_PROMPT = """You are an AI assistant that helps with product and package recommendations. 

  Please respond in the following format:
  <think>
  Your reasoning process here - analyze the requirements, consider the options, and explain your decision-making process.
  </think>

  <answer>
  Your final recommendation in JSON format: {"product": "product_name", "package": "package_name"}
  </answer>

  CATALOG:
  {json_string}
  """

def get_dataset(split="train") -> Dataset:
  """Create dataset with thinking examples"""
  with open('verifiers/fine-tuning/cleaned_combinations.json', 'r') as file:
      data = json.load(file)

  if split == 'train':
      data = data[:100]
  elif split == 'test':
      data = data[100:]

  formatted_data = []
  for item in data:
      # Create the prompt
      prompt = [
          {'role': 'system', 'content': SYSTEM_PROMPT},
          {'role': 'user', 'content': item['input']}
      ]

      formatted_data.append({
          'prompt': prompt,
          'answer': item['output']  # Keep original for reward calculation
      })

  return Dataset.from_list(formatted_data)

# Custom Parser that extends XMLParser for think/answer format
class ThinkAnswerParser(vf.XMLParser):
    def __init__(self):
        super().__init__(['think', 'answer'])

    def get_format_reward_func(self):
        """Return a reward function that evaluates XML format compliance"""
        def format_reward_func(prompts, completions, answers, **kwargs) -> list[float]:
            rewards = []

            for completion in completions:
                response = completion[0]["content"] if isinstance(completion, list) else completion
                score = 0.0

                # Check if both required tags are present
                has_think = '<think>' in response and '</think>' in response
                has_answer = '<answer>' in response and '</answer>' in response

                if has_think and has_answer:
                    # Full score for having both tags
                    score = 1.0

                    # Parse and check content quality
                    parsed = self.parse(response)
                    think_content = parsed.get('think', '').strip()
                    answer_content = parsed.get('answer', '').strip()

                    # Bonus for non-empty content
                    if think_content and answer_content:
                        score += 0.2

                    # Check proper tag ordering (think should come before answer)
                    think_start = response.find('<think>')
                    answer_start = response.find('<answer>')
                    if think_start < answer_start and think_start >= 0:
                        score += 0.1

                elif has_think or has_answer:
                    # Partial score for having one tag
                    score = 0.3
                else:
                    # No format compliance
                    score = 0.0

                rewards.append(score)

            return rewards

        return format_reward_func

    def extract_json_from_answer(self, text: str) -> dict:
        """Extract JSON from the final answer portion"""
        parsed = self.parse(text)
        answer_text = parsed.get('answer', '')

        # Try to find JSON in the answer
        json_pattern = r'\{\s*"product"\s*:\s*"([^"]*)"[^}]*"package"\s*:\s*"([^"]*)"\s*\}'
        match = re.search(json_pattern, answer_text)
        if match:
            return {'product': match.group(1), 'package': match.group(2)}

        # Try to parse any JSON-like structure
        try:
            json_start = answer_text.find('{')
            json_end = answer_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = answer_text[json_start:json_end]
                parsed_json = json.loads(json_str)
                if isinstance(parsed_json, dict) and 'product' in parsed_json and 'package' in parsed_json:
                    return parsed_json
        except:
            pass

        return {'product': '', 'package': ''}

# Custom reward functions for the verifiers rubric system
def thinking_structure_reward_func(prompts, completions, answers, **kwargs) -> list[float]:
    """Reward proper thinking structure and clean output"""
    parser = kwargs.get('parser')
    if not parser:
        parser = ThinkAnswerParser()

    rewards = []

    for completion in completions:
        response = completion[0]["content"] if isinstance(completion, list) else completion
        score = 0.0

        # Parse the response
        parsed = parser.parse(response)
        thinking = parsed.get('think', '')
        answer = parsed.get('answer', '')

        # Reward having proper structure (0.3 points)
        if thinking and answer:
            score += 0.3

            # Check thinking quality
            if len(thinking) > 50:  # Substantial thinking
                score += 0.1
            if any(word in thinking.lower() for word in ["analyze", "requirements", "compare", "consider"]):
                score += 0.1

        # Reward clean answer after thinking (0.4 points)
        if answer and len(answer) < 200:  # Concise answer
            score += 0.2
        if '{"product":' in answer or '"product"' in answer:
            score += 0.2

        # Reward proper JSON structure (0.3 points)
        try:
            if hasattr(parser, 'extract_json_from_answer'):
                extracted = parser.extract_json_from_answer(response)
                if extracted['product'] and extracted['package']:
                    score += 0.3
        except:
            pass

        rewards.append(score)

    return rewards

def answer_conciseness_reward_func(prompts, completions, answers, **kwargs) -> list[float]:
    """Reward concise, professional answers after thinking"""
    parser = kwargs.get('parser')
    if not parser:
        parser = ThinkAnswerParser()

    rewards = []

    for completion in completions:
        response = completion[0]["content"] if isinstance(completion, list) else completion
        score = 0.0

        parsed = parser.parse(response)
        answer = parsed.get('answer', '')

        if not answer:
            rewards.append(0.0)
            continue

        answer_length = len(answer)

        # Reward appropriate length (0.4 points)
        if 30 <= answer_length <= 150:  # Sweet spot for concise answers
            score += 0.4
        elif 20 <= answer_length <= 200:  # Acceptable range
            score += 0.2

        # Reward professional language (0.3 points)
        professional_phrases = ["based on", "recommend", "optimal", "suitable", "analysis"]
        found_professional = sum(1 for phrase in professional_phrases if phrase.lower() in answer.lower())
        score += min(0.3, found_professional * 0.1)

        # Reward clean JSON inclusion (0.3 points)
        if '{"product":' in answer and answer.count('{') == 1:  # Single clean JSON
            score += 0.3

        rewards.append(score)

    return rewards

def anti_repetition_reward_func(prompts, completions, answers, **kwargs) -> list[float]:
    """Prevent repetitive patterns, especially in thinking sections"""
    rewards = []

    for completion in completions:
        response = completion[0]["content"] if isinstance(completion, list) else completion
        score = 1.0

        if len(response) < 30:
            score = 0.0
        else:
            # Check for character repetition
            char_counts = {}
            for char in response:
                char_counts[char] = char_counts.get(char, 0) + 1

            max_char_count = max(char_counts.values())
            if max_char_count > min(15, len(response) * 0.25):  # No char > 25% of response
                score = 0.0

            # Check for word repetition
            words = response.lower().split()
            if len(words) > 5:
                word_counts = {}
                for word in words:
                    if len(word) > 3:  # Only check substantial words
                        word_counts[word] = word_counts.get(word, 0) + 1

                if word_counts:
                    max_word_count = max(word_counts.values())
                    if max_word_count > max(3, len(words) * 0.2):  # No word > 20%
                        score *= 0.1

            # Specific problematic patterns
            if response.count('!') > 8:
                score = 0.0
            if response.count('?') > 5:
                score *= 0.5

        rewards.append(score)

    return rewards

def correctness_reward_func(prompts, completions, answers, **kwargs) -> list[float]:
    """Reward correct final answers regardless of thinking process"""
    parser = kwargs.get('parser')
    if not parser:
        parser = ThinkAnswerParser()

    rewards = []

    for completion, expected in zip(completions, answers):
        response = completion[0]['content'] if isinstance(completion, list) else completion
        score = 0.0

        try:
            # Extract JSON from the final answer using parser
            if hasattr(parser, 'extract_json_from_answer'):
                extracted = parser.extract_json_from_answer(response)
            else:
                # Fallback extraction
                parsed = parser.parse(response)
                answer_text = parsed.get('answer', '')
                extracted = {'product': '', 'package': ''}
                # Basic JSON extraction logic here

            # Parse expected answer
            if isinstance(expected, str):
                expected_dict = json.loads(expected)
            else:
                expected_dict = expected

            # High reward for correct answers (this is what matters most)
            if extracted.get('product') == expected_dict.get('product'):
                score += 1.5  # Higher weight since thinking is just means to an end

            if extracted.get('package') == expected_dict.get('package'):
                score += 1.5

        except Exception as e:
            print(f"Correctness evaluation error: {e}")
            score = 0.0

        rewards.append(score)

    return rewards

parser = ThinkAnswerParser()
    
# Create the rubric with all reward functions
rubric = vf.Rubric(
    funcs = [thinking_structure_reward_func,
    answer_conciseness_reward_func,
    anti_repetition_reward_func,
    correctness_reward_func,
    parser.get_format_reward_func()],  # Now properly implemented format reward
    weights=[1.0, 0.8, 1.0, 3.0, 0.5]  # Adjust weights as needed
)

# Create the environment
dataset = get_dataset()

vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    system_prompt=SYSTEM_PROMPT,
    parser=parser,
    rubric=rubric
)

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
args = vf.grpo_defaults(run_name='reverse_text_warmup')

# Scale up for larger experiments
args.per_device_train_batch_size = 4
args.num_generations = 3
args.gradient_accumulation_steps = 8
args.max_concurrent = 512
args.num_train_epochs = 3

# Memory optimization
args.gradient_checkpointing = True
args.bf16 = True

# Learning schedule
args.learning_rate = 1e-6
args.lr_scheduler_type = "constant_with_warmup"
args.warmup_steps = 10

# Logging and saving
args.logging_steps = 1
args.save_strategy = "steps"
args.save_steps = 10
args.report_to = "wandb"
args.log_completions = True

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=args,
    #peft_config=vf.lora_defaults()
)
trainer.train()
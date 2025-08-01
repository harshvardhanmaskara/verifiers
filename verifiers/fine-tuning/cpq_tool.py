import verifiers as vf
from verifiers.tools import search_product, validate_product
from datasets import load_dataset
from custom_cpq_rubric import CustomCPQRubric

"""
inference:
CUDA_VISIBLE_DEVICES=0 vf-vllm --model harshvardhanmaskara/SmolLM2-135M-CPQ-SFT --enforce-eager

training:
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 --config-file configs/zero3.yaml verifiers/fine-tuning/cpq_tool.py
"""

TOOL_PROMPT = """
  You are a smart product retrieval assistant. Your job is to understand the user query and help them solve it. You must follow the steps below in order to solve the user query:

  Think step-by-step inside <think>...</think> tags in each message, then either call a tool inside <tool>...</tool> tags, or give your final answer inside <answer>...</answer> tags.

  You have access to the following tools to search and retrieve an appropriate product for the user's configuration query:

  search_product Tool: Searches and retrieves the best product fit for the user's configuration query.
  - Args
    "query": The user query in plain english text
  - Returns
    Formatted string with the product and its associated features in the most suitable configuration
  - Example usage:
    <tool>
    {"name": "search_product", "args": {"query": "user query in natural language"}}
    </tool>

  validate_product Tool: Validates the product retrieved by the search tool before returning it as the final answer
  - Args
    "product": The product configuration returned by the search tool
  - Returns
    "Valid" or "Not Valid" as a string
  - Example usage:
    <tool>
    {"name": "validate_product", "args": {"product": {"Product": "name", "Features": ["list of features"], "Price": "price"}}}
    </tool>

  IMPORTANT: Only call ONE tool at a time. After calling a tool, wait for the result before proceeding. Do not simulate tool results or continue the conversation beyond the tool call.

  The <answer>...</answer> tags should contain only your final answer as a clear, concise response.

  Example to start the conversation:

  <think>
  Let me search for available laptop configurations first, then validate the selected configuration.
  </think>

  <tool>
  {"name": "search_product", "args": {"query": "user query in natural language"}}
  </tool>
"""

# Load the CPQ dataset
dataset = load_dataset('json', data_files='verifiers/fine-tuning/data/dataset.json')

dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = dataset["train"]
eval_ds = dataset["test"]

# Create custom rubric for CPQ environment
custom_rubric = CustomCPQRubric(
    parser=vf.XMLParser(fields=["think", ("tool", "answer")]),
    env_parser=vf.XMLParser(fields=["result"]),
    tools=[search_product, validate_product]
)

# Create the CPQ environment with custom rubric
vf_env = vf.ToolEnv(
    format_prompt=False,
    dataset=train_ds,
    system_prompt=TOOL_PROMPT,
    few_shot=[],
    tools=[search_product, validate_product],
    max_steps=3,  # Allow more steps for CPQ workflow
    rubric=custom_rubric  # Use our custom rubric
)
print(vf_env.system_prompt)

# Load the SFT model
model_name = "harshvardhanmaskara/SmolLM2-135M-CPQ-SFT"
model, tokenizer = vf.get_model_and_tokenizer(model_name, use_liger=False)
run_name = "cpq-grpo_" + model_name.split("/")[-1].lower()

# Configure training arguments
training_args = vf.grpo_defaults(run_name=run_name)
training_args.num_iterations = 2
training_args.per_device_train_batch_size = 4
training_args.num_generations = 5
training_args.gradient_accumulation_steps = 2
training_args.max_length = 2048  # Reduced to prevent excessive length
training_args.learning_rate = 2e-5
training_args.num_train_epochs = 3
training_args.weight_decay = 0.01
training_args.max_grad_norm = 1.0
training_args.report_to = "wandb"
training_args.save_strategy = "epoch"
training_args.save_total_limit = 1
training_args.logging_steps = 1
training_args.save_only_model = True
training_args.log_on_each_node = True
training_args.push_to_hub = True
training_args.hub_model_id = "harshvardhanmaskara/SmolLM2-135M-CPQ-GRPO"

# Additional parameters to prevent reward hacking
training_args.frequency_penalty = 0.1  # Penalize repeated tokens
training_args.top_k = 50  # Limit token selection diversity
training_args.temperature = 0.7  # Moderate temperature to reduce randomness

# Create and start training
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train()
import verifiers as vf
from verifiers.tools import search_product
from datasets import load_dataset
from custom_cpq_rubric import CustomCPQRubric

"""
inference:
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 vf-vllm --model harshvardhanmaskara/SmolLM2-135M-CPQ-SFT --enforce-eager

training:
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1 CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 --config-file configs/zero3.yaml verifiers/fine_tuning/cpq_tool.py
"""

TOOL_PROMPT = f"""
  You are a smart assistant whose job is to understand the user query and solve it by making a tool call with the appropriate argument. Follow the steps listed below:

  Think step-by-step inside <think>...</think> tags about the user query and understand what it is asking. Then call a search tool inside <tool>...</tool> tags, and summarize what you did inside <answer>...</answer> tags.

  You have access to the following search tool to search and retrieve an appropriate product for the user's configuration query:

  search_product Tool: Searches and retrieves the best product fit for the user's configuration query.
  - Args
    "query": The user query in plain english text
  - Returns
    Formatted string with the product and its associated features in the most suitable configuration
  - Example usage:
    <tool>
    {{"name": "search_product", "args": {{"query": "user query in natural language"}}}}
    </tool>

  The <answer>...</answer> tags should contain only a summary of what you did.

  Example to start the conversation:

  <think>
  The user is looking for a high performance gaming laptop. Let me use the search tool to find a suitable product.
  </think>

  <tool>
  {{"name": "search_product", "args": {{"query": "High performance gaming laptop"}}}}
  </tool>

  <answer>
  I reasoned about the user's product needs and made a tool call to retrieve the appropriate option.
  </answer>
"""

# Load the CPQ dataset
dataset = load_dataset('json', data_files='verifiers/fine_tuning/data/dataset.json', split='train')

dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = dataset["train"]
eval_ds = dataset["test"]

# Create custom rubric for CPQ environment
custom_rubric = CustomCPQRubric(
    parser=vf.XMLParser(fields=["think", ("tool", "answer")]),
    env_parser=vf.XMLParser(fields=["result"]),
    tools=[search_product]
)

# Create the CPQ environment with custom rubric
vf_env = vf.ToolEnv(
    format_prompt=False,
    dataset=train_ds,
    system_prompt=TOOL_PROMPT,
    few_shot=[],
    tools=[search_product],
    max_steps=3,  # Allow more steps for CPQ workflow
    rubric=custom_rubric  # Use our custom rubric
)
print(vf_env.system_prompt)

# Load the SFT model
model_name = "harshvardhanmaskara/SmolLM2-135M-CPQ-SFT"
model, tokenizer = vf.get_model_and_tokenizer(model_name, use_liger=False)
run_name = "simple-grpo_" + model_name.split("/")[-1].lower()

# Configure training arguments
training_args = vf.grpo_defaults(run_name=run_name)
training_args.num_iterations = 2
training_args.per_device_train_batch_size = 4
training_args.num_generations = 8
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
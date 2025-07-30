import verifiers as vf
from verifiers.tools import cpq_search, cpq_validate
from datasets import load_dataset

"""
inference:
CUDA_VISIBLE_DEVICES=0 vf-vllm --model harshvardhanmaskara/SmolLM2-135M-CPQ-SFT --enforce-eager

training:
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 --config-file configs/zero3.yaml verifiers/fine-tuning/cpq_tool.py
"""

TOOL_PROMPT = """
Think step-by-step inside <think>...</think> tags in each message, then either call a tool inside <tool>...</tool> tags, or give your final answer inside <answer>...</answer> tags.

You have access to the following tools to help solve CPQ (Configure, Price, Quote) problems:

{tool_descriptions}

Tools can be called by writing a JSON command inside <tool> tags with:
- "name": the name of the tool to use
- "args": the arguments for the tool

Example usage:
<tool>
{{"name": "cpq_search", "args": {{"query": "laptop configurations", "search_type": "product", "max_results": 3}}}}
</tool>

<tool>
{{"name": "cpq_validate", "args": {{"validation_type": "configuration", "data": {{"cpu": "Intel i7", "ram": "16GB", "storage": "512GB"}}}}}}
</tool>

After concluding your message with a tool call,
you will then see the tool's output inside <result> tags as a new message. \
You may call tools multiple times if needed. \
Tool state does not persist between calls. \
Always use tools to solve problems whenever possible, rather than using your own knowledge.

The <answer>...</answer> tags should contain only your final answer as a clear, concise response.

Example:
<think>
Let me search for available laptop configurations first, then validate the selected configuration.
</think>
<tool>
{{"name": "cpq_search", "args": {{"query": "laptop configurations", "search_type": "product", "max_results": 3}}}}
</tool>
<result>
• Dell Latitude 5520 - Intel i7, 16GB RAM, 512GB SSD - $1,299
• HP EliteBook 840 - Intel i5, 8GB RAM, 256GB SSD - $899
• Lenovo ThinkPad X1 - Intel i7, 32GB RAM, 1TB SSD - $1,599
</result>
<think>
Now let me validate the Dell Latitude configuration to ensure it meets requirements.
</think>
<tool>
{{"name": "cpq_validate", "args": {{"validation_type": "configuration", "data": {{"cpu": "Intel i7", "ram": "16GB", "storage": "512GB"}}}}}}
</tool>
<result>
✓ Configuration is valid
✓ All components are compatible
✓ Meets minimum system requirements
✓ Within budget constraints
</result>
<answer>
{{
  "Product": "Dell Latitude 5520",
  "Features": ["Intel i7 processor", "16GB RAM", "512GB SSD", "Compact design"],
  "Price": "$1,299"
}}
</answer>
"""

# Load the CPQ dataset
dataset = load_dataset('json', data_files='verifiers/fine-tuning/data/dataset.json', split='train')

dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = dataset["train"]
eval_ds = dataset["test"]

# Create the CPQ environment with tools
vf_env = vf.ToolEnv(
    format_prompt=False,
    dataset=train_ds,
    system_prompt=TOOL_PROMPT,
    few_shot=[],
    tools=[cpq_search, cpq_validate],
    max_steps=5  # Allow more steps for CPQ workflow
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
training_args.num_generations = 8
training_args.gradient_accumulation_steps = 2
training_args.max_length = 4096
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

# Create and start training
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train()
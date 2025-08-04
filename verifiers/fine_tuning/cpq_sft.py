import verifiers as vf
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

"""
accelerate launch --config-file configs/zero3.yaml --num-processes 1 verifiers/fine-tuning/cpq_sft.py
"""

# convenience function for FA2 initialization
model, tokenizer = vf.get_model_and_tokenizer("HuggingFaceTB/SmolLM2-135M-Instruct", use_liger=False)
dataset = load_dataset('json', data_files='verifiers/fine-tuning/data/simple_dataset.json', split='train')

def to_chat(row):
    # Convert question/answer pair to chat format
    messages = [
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["answer"]}
    ]
    return {"messages": messages}

cols = dataset.column_names
dataset = dataset.map(to_chat, remove_columns=cols)

dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = dataset["train"]
eval_ds = dataset["test"]

tok_counts = []
for row in train_ds:
    # count tokens in chat format
    messages = row['messages']
    toks = tokenizer.apply_chat_template(
        messages,
        tokenize=True
    )
    tok_counts.append(len(toks))

# tok count stats
print(f"Dataset size: {len(tok_counts)}")
print(f"Min tokens: {min(tok_counts)}")
print(f"Max tokens: {max(tok_counts)}")
print(f"Mean tokens: {sum(tok_counts) / len(tok_counts)}")
print(f"Median tokens: {sorted(tok_counts)[len(tok_counts) // 2]}")

args = SFTConfig(
    max_length=4096,
    output_dir="sft-warmup-2.0",
    per_device_train_batch_size=3,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    bf16=True,
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    max_grad_norm=1.0,
    report_to="wandb",
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=1,
    save_only_model=True,
    log_on_each_node=True,
    push_to_hub=True,
    hub_model_id="harshvardhanmaskara/SmolLM2-135M-SFT-2.0",
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_ds, # type: ignore
    eval_dataset=eval_ds,  # type: ignore
)
trainer.train()
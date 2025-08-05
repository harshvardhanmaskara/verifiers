from transformers import pipeline

# Create a pipeline for your task (e.g., text generation)
generator = pipeline("text-generation", model="harshvardhanmaskara/SmolLM2-135M-CPQ-GRPO")

# Answer a single prompt
prompt = "I want a high performance AI inference server"
system_prompt = f"""
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
response = generator([{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}], max_new_tokens=1000, return_full_text=True)[0]
print(response["generated_text"][2]['content'])
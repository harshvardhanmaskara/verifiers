from transformers import pipeline

# Create a pipeline for your task (e.g., text generation)
generator = pipeline("text-generation", model="harshvardhanmaskara/SmolLM2-135M-CPQ-SFT")

# Answer a single prompt
prompt = "I want a high performance compute cluster for AI training with high memory."
system_prompt = """Your task is to answer the user's question in this EXACT format:
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
response = generator([{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}], max_new_tokens=1000, return_full_text=True)[0]
print(response["generated_text"][2]['content'])
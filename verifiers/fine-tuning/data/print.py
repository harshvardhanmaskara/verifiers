import json

with open("dataset.json", "r") as f:
    data = json.load(f)  # This loads the whole array
    print(data[0]['answer'])       # Print the first row (a dict with 'question' and 'answer')
import requests
import json

# Load data from a JSON file
with open('input.json', 'r') as file:
    payload = json.load(file)

url = "http://127.0.0.1:8000/find_best_match"

response = requests.post(url, json=payload)
print(response.json())
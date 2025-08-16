import requests

headers = {
    "Authorization": "Bearer genai-intern-2024",
    "Content-Type": "application/json"
}

data = {
    "blog_texts": ["AI is transforming our world in amazing ways."]
}

response = requests.post(
    "http://localhost:8000/api/analyze-blogs",
    headers=headers,
    json=data
)

print(response.json())
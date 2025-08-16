import requests
import json

headers = {
    "Authorization": "Bearer genai-intern",
    "Content-Type": "application/json"
}


data = {
    "blog_texts": [
        "AI is transforming our world in amazing ways.",
        "Climate change poses serious risks to global economies and ecosystems.",
        "Remote work has redefined productivity and employee well-being.",
        "Healthcare innovation is accelerating with AI-driven diagnostics.",
        "Blockchain is reshaping trust and transparency in finance."
    ]
}

url = "http://localhost:8000/api/analyze-blogs"

response = requests.post(url, headers=headers, json=data)


results = response.json()
print(results)


postman_collection = {
    "info": {
        "name": "Blog Analysis API",
        "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
    },
    "item": [
        {
            "name": "Analyze Blogs",
            "request": {
                "method": "POST",
                "header": [
                    {"key": "Authorization", "value": "Bearer genai-intern"},
                    {"key": "Content-Type", "value": "application/json"}
                ],
                "url": {
                    "raw": url,
                    "protocol": "http",
                    "host": ["localhost"],
                    "port": "8000",
                    "path": ["api", "analyze-blogs"]
                },
                "body": {
                    "mode": "raw",
                    "raw": json.dumps(data, indent=2)
                }
            },
            "response": [
                {
                    "name": "Sample Response",
                    "originalRequest": {
                        "method": "POST",
                        "url": {"raw": url}
                    },
                    "status": "OK",
                    "code": 200,
                    "body": json.dumps(results, indent=2)
                }
            ]
        }
    ]
}


with open("postman_collection.json", "w") as f:
    json.dump(postman_collection, f, indent=2)

print("Postman collection saved to postman_collection.json")

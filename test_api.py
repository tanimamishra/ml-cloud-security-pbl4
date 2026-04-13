import requests

url = "http://127.0.0.1:5000/predict"

# Sample input (NSL-KDD format)
data = {
    "duration": 0,
    "protocol_type": "tcp",
    "service": "http",
    "flag": "SF",
    "src_bytes": 181,
    "dst_bytes": 5450
}

response = requests.post(url, json=data)

print(response.json())
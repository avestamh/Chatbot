import requests

url = "http://127.0.0.1:5000/ask"
payload = {"question": "What is the vacation policy?"}

response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Response Text:", response.text)

import requests
import os

# Verify server connectivity
try:
    response = requests.get("http://localhost:8000/")
    print(f"Server GET response: {response.status_code} - {response.text}")
except requests.exceptions.ConnectionError as e:
    print(f"Server GET connection error: {e}")
    exit()

file_path = r"c:\Users\Sadique\Desktop\ai model\uploads\sample_customer_data.csv"
url = "http://localhost:8000/upload"

with open(file_path, "rb") as f:
    files = {"file": (os.path.basename(file_path), f, "text/csv")}
    response = requests.post(url, files=files)

print(response.json())
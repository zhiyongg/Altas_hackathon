import requests

url = "https://api.stayapi.com/v1/meta/search"
headers = {
    "x-api-key": "sk_live_d754d0f7f852166ac9a174e492105d4df25c1fc425791487e0dbe745baa7c11b",
    "Content-Type": "application/json"
}
params = {
    "hotel_name": "Four Seasons Resort Bali",
    "location": "Jimbaran, Indonesia"
}

# 1. Send Request
response = requests.get(url, headers=headers, params=params)

# 2. Inspect raw status and text first
print(f"Status Code: {response.status_code}")
print(f"Raw Response: {repr(response.text)}")

# 3. Handle response safely
if response.status_code == 200 and response.text.strip():
    try:
        data = response.json()
        links = data.get("data", {}).get("links", {})
        for platform, link_url in links.items():
            print(f"  {platform}: {link_url}")
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
else:
    print(f"Request failed with status {response.status_code}. Response: {response.text}")
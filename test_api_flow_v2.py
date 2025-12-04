import requests
import json
import base64

SUPABASE_URL = "https://ahmzlbndtclnbiptpvex.supabase.co"
ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFobXpsYm5kdGNsbmJpcHRwdmV4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1MzM3MzIsImV4cCI6MjA3OTEwOTczMn0.crigFt3xKl88S9YLlfqUTUGlyeE7dPC-3u6XTKOmdmQ"

# 1. Login
print("=== Step 1: Login ===")
login_response = requests.post(
    f"{SUPABASE_URL}/functions/v1/user-auth",
    headers={
        "Content-Type": "application/json",
        "apikey": ANON_KEY,
        "Authorization": f"Bearer {ANON_KEY}"  # Use anon key for public endpoints
    },
    json={
        "action": "login",
        "email": "test-user-20251119@example.com",
        "password": "TestPass123!"
    }
)

print(f"Status: {login_response.status_code}")
login_data = login_response.json()
print(json.dumps(login_data, indent=2))

if login_response.status_code != 200 or 'error' in login_data:
    print("Login failed!")
    exit(1)

token = login_data['data']['token']
user_id = login_data['data']['user']['id']

print(f"\nToken: {token}")
print(f"User ID: {user_id}")

# 2. Upload file
print("\n=== Step 2: Upload File ===")
with open("/workspace/test_data/test-charging-correct-format.csv", "rb") as f:
    file_content = f.read()
    base64_data = base64.b64encode(file_content).decode('utf-8')

upload_response = requests.post(
    f"{SUPABASE_URL}/functions/v1/file-upload",
    headers={
        "Content-Type": "application/json",
        "apikey": ANON_KEY,
        "Authorization": f"Bearer {token}"  # Use user token
    },
    json={
        "fileData": f"data:text/csv;base64,{base64_data}",
        "fileName": "test-charging-correct-format.csv",
        "fileSize": len(file_content),
        "userId": user_id,
        "analysisName": f"API Test Analysis"
    }
)

print(f"Status: {upload_response.status_code}")
upload_data = upload_response.json()
print(json.dumps(upload_data, indent=2))

if upload_response.status_code != 200 or 'error' in upload_data:
    print("Upload failed!")
    exit(1)

analysis_id = upload_data['data']['analysisId']
print(f"\nAnalysis ID: {analysis_id}")

# 3. Start analysis
print("\n=== Step 3: Start Analysis ===")
analysis_response = requests.post(
    f"{SUPABASE_URL}/functions/v1/charging-analysis-v2",
    headers={
        "Content-Type": "application/json",
        "apikey": ANON_KEY,
        "Authorization": f"Bearer {token}"  # Use user token
    },
    json={
        "analysisId": analysis_id,
        "userId": user_id
    }
)

print(f"Status: {analysis_response.status_code}")
try:
    analysis_data = analysis_response.json()
    print(json.dumps(analysis_data, indent=2))
    
    if analysis_response.status_code == 200 and 'error' not in analysis_data:
        print("\n✅ Analysis completed successfully!")
        print(f"Results count: {len(analysis_data.get('data', {}).get('results', []))}")
    else:
        print("\n❌ Analysis failed!")
except Exception as e:
    print(f"Error parsing response: {e}")
    print(f"Response text: {analysis_response.text}")

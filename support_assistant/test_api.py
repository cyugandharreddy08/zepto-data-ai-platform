import json
from fastapi.testclient import TestClient
from support_assistant.main import app

client = TestClient(app)

print("="*70)
print("TESTING FASTAPI /ask ENDPOINT")
print("="*70)

# Test 1: Policy Question (Triggers Retrieval)
q1 = {"query": "What is the delivery fee for orders under INR 149?"}
resp1 = client.post("/ask", json=q1)
print(f"\nRequest 1 (Policy Question): {json.dumps(q1)}")
print(f"Status Code: {resp1.status_code}")
print(f"Raw Response JSON:\n{json.dumps(resp1.json(), indent=2)}")

# Test 2: General Question (No Retrieval)
q2 = {"query": "Can you explain quantum computing?"}
resp2 = client.post("/ask", json=q2)
print(f"\nRequest 2 (General Question): {json.dumps(q2)}")
print(f"Status Code: {resp2.status_code}")
print(f"Raw Response JSON:\n{json.dumps(resp2.json(), indent=2)}")

# Test 3: Additional Policy Question (Returns & Refunds)
q3 = {"query": "How many days do I have to report a return for perishable grocery items?"}
resp3 = client.post("/ask", json=q3)
print(f"\nRequest 3 (Policy Question - Returns): {json.dumps(q3)}")
print(f"Status Code: {resp3.status_code}")
print(f"Raw Response JSON:\n{json.dumps(resp3.json(), indent=2)}")

assert resp1.status_code == 200
assert resp2.status_code == 200
assert resp3.status_code == 200
assert "doc_01" in resp1.json()["sources"]
assert resp2.json()["sources"] == []
assert "doc_02" in resp3.json()["sources"]

print("\nALL FASTAPI ENDPOINT TESTS PASSED WITH 100% SUCCESS!")

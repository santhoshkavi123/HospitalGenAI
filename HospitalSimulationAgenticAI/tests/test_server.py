import os
import sys

# Add project root and demo directory to python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "demo"))

from fastapi.testclient import TestClient
from demo.server import app

client = TestClient(app)

def test_get_patients():
    """Test retrieving patient list from the API."""
    response = client.get("/api/patients")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0

def test_get_patient_details():
    """Test retrieving details for a specific patient."""
    patients_resp = client.get("/api/patients")
    patient_id = patients_resp.json()[0]["id"]
    
    response = client.get(f"/api/patients/{patient_id}")
    assert response.status_code == 200
    detail = response.json()
    assert detail["Patient_ID"] == patient_id

def test_simulate_workflow_success():
    """Test successful execution of multi-agent simulation workflow."""
    patients_resp = client.get("/api/patients")
    patient_id = patients_resp.json()[0]["id"]
    
    response = client.post("/api/simulate", json={"patient_id": patient_id})
    assert response.status_code == 200
    res = response.json()
    assert res["verified"] is True
    assert res["final_status"] == "Completed"
    assert len(res["steps"]) >= 4

def test_simulate_workflow_fail_verification():
    """Test multi-agent simulation when Front Desk verification fails."""
    patients_resp = client.get("/api/patients")
    patient_id = patients_resp.json()[0]["id"]
    
    response = client.post("/api/simulate", json={"patient_id": patient_id, "force_verify_fail": True})
    assert response.status_code == 200
    res = response.json()
    assert res["verified"] is False
    assert res["final_status"] == "Verification Failed"
    assert len(res["steps"]) == 1

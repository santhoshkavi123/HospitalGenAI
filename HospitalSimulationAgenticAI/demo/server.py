import os
import random
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Hospital Multi-Agent Simulation Demo")

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATASET_PATH = os.path.join(PROJECT_DIR, "datasets", "input", "processed", "updated_patient_df.csv")

# Load patient dataset
try:
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        # Drop unnamed columns if any
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    elif os.path.exists(os.path.join(BASE_DIR, "static", "patients.json")):
        df = pd.read_json(os.path.join(BASE_DIR, "static", "patients.json"))
    else:
        # Fallback dummy data if dataset is missing
        df = pd.DataFrame([
            {
                "Patient_ID": "PAT1000000007",
                "First_Name": "Amrita",
                "Last_Name": "Nair",
                "Gender": "Female",
                "Age": 25,
                "Disease": "Influenza",
                "Fever": "Yes",
                "Cough": "Yes",
                "Fatigue": "Yes",
                "Difficulty Breathing": "Yes",
                "Blood Pressure": "Normal",
                "Cholesterol Level": "Normal",
                "Outcome Variable": "Positive"
            }
        ])
except Exception as e:
    print(f"Error loading dataset: {e}")
    df = pd.DataFrame([])

@app.get("/api/patients")
def get_patients():
    """Return lists of patients for selection."""
    if df.empty:
        return []
    
    patients = []
    for idx, row in df.iterrows():
        patients.append({
            "id": row.get("Patient_ID", f"PAT_{idx}"),
            "first_name": row.get("First_Name", "Unknown"),
            "last_name": row.get("Last_Name", "Unknown"),
            "full_name": f"{row.get('First_Name', '')} {row.get('Last_Name', '')}".strip(),
            "age": int(row.get("Age", 30)),
            "gender": row.get("Gender", "Unknown"),
            "disease": row.get("Disease", "Unknown")
        })
    return patients

@app.get("/api/patients/{patient_id}")
def get_patient_details(patient_id: str):
    """Retrieve full record for a specific patient."""
    records = df[df["Patient_ID"] == patient_id]
    if records.empty:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    row = records.iloc[0]
    return row.to_dict()

class SimulationRequest(BaseModel):
    patient_id: str
    force_verify_fail: bool = False

@app.post("/api/simulate")
def run_simulation(req: SimulationRequest):
    """
    Simulate the multi-agent graph execution step-by-step for the given patient.
    """
    records = df[df["Patient_ID"] == req.patient_id]
    if records.empty:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    patient = records.iloc[0].to_dict()
    steps = []
    
    # ------------------ STEP 1: FRONT DESK AGENT ------------------
    # Pre-train predicted values. ViTs have some small variance.
    # We will simulate gender classification and age group predictions.
    predicted_gender = patient["Gender"].lower()
    
    # Map age to group ranges
    age = int(patient["Age"])
    if age <= 2: age_group = "0-2"
    elif age <= 9: age_group = "3-9"
    elif age <= 19: age_group = "10-19"
    elif age <= 29: age_group = "20-29"
    elif age <= 39: age_group = "30-39"
    elif age <= 49: age_group = "40-49"
    elif age <= 59: age_group = "50-59"
    elif age <= 69: age_group = "60-69"
    else: age_group = "more than 70"

    # Simulate verification logic
    if req.force_verify_fail:
        # Simulate mismatched verification
        predicted_gender = "male" if predicted_gender == "female" else "female"
        verified = False
        verif_msg = f"Verification failed. Patient checks as {predicted_gender} but database record indicates {patient['Gender']}."
    else:
        verified = True
        verif_msg = (
            f"Verification successful.\n"
            f"Patient is: {patient['First_Name']} {patient['Last_Name']}\n"
            f"with ID: {patient['Patient_ID']}\n"
            f"who is verified as {predicted_gender} in age group of {age_group}. "
            f"Proceeding to the Physician."
        )

    steps.append({
        "node": "front_desk_agent_node",
        "title": "Front Desk Agent Node",
        "status": "success" if verified else "failed",
        "outputs": {
            "predicted_gender": predicted_gender,
            "predicted_age_group": age_group,
            "patient_verification": verif_msg,
            "verified": verified
        },
        "logs": [
            "Front Desk Agent activated.",
            "Loading Face ID credentials from FaceID/image1.png...",
            "Running rizvandwiki/gender-classification pipeline...",
            f"ViT Gender classification output: {predicted_gender}",
            "Running nateraw/vit-age-classifier pipeline...",
            f"ViT Age classification output: {age_group}",
            "Querying medical database records...",
            f"Database comparison status: {'MATCH FOUND' if verified else 'NO RECORD MATCH'}",
            verif_msg
        ]
    })
    
    if not verified:
        # Exit immediately if verification fails
        return {
            "patient": patient,
            "verified": False,
            "steps": steps,
            "final_status": "Verification Failed"
        }

    # ------------------ STEP 2: PHYSICIAN - CONSULTATION ------------------
    symptom_questions = {
        'Cough': "Are you having any cough?",
        'Fatigue': "Are you having any Fatigue?",
        'Difficulty Breathing': "Are you experiencing any difficulty while breathing?"
    }
    
    conversation_logs = []
    symptom_summary_items = []
    
    for symptom, question in symptom_questions.items():
        has_symptom = patient.get(symptom, "No") == "Yes"
        answer = "Yes" if has_symptom else "No"
        conversation_logs.append(f"Physician: {question}")
        conversation_logs.append(f"Patient: {answer}")
        
        if has_symptom:
            symptom_summary_items.append(f"You are experiencing {symptom.lower()}")
        else:
            symptom_summary_items.append(f"I'm glad you are not experiencing any {symptom.lower()}")

    physician_summary = (
        f"You are {patient['First_Name']} {patient['Last_Name']}, a {patient['Age']} years old {patient['Gender']} "
        f"with Patient ID: {patient['Patient_ID']}.\n"
        f"I gathered that:\n" + " ; ".join(symptom_summary_items) + "."
    )
    
    steps.append({
        "node": "physician_consultation_node",
        "title": "Physician Agent Node (Consultation)",
        "status": "success",
        "outputs": {
            "conversation": "\n".join(conversation_logs),
            "summary": physician_summary
        },
        "logs": [
            "Physician Agent Consultation activated.",
            "Screening patient for respiratory and systemic symptoms...",
            *conversation_logs,
            "Synthesizing anamnesis clinical records...",
            physician_summary
        ]
    })

    # ------------------ STEP 3: PHYSICIAN - PHYSICAL EXAM ------------------
    fever = patient.get('Fever', 'Unknown')
    bp = patient.get('Blood Pressure', 'Unknown')
    chol = patient.get('Cholesterol Level', 'Unknown')
    
    exam_summary = f"Examination Results: Fever: {fever}, Blood Pressure: {bp} and Cholesterol level: {chol}"
    
    steps.append({
        "node": "physician_examination_node",
        "title": "Physician Agent Node (Examination)",
        "status": "success",
        "outputs": {
            "fever": fever,
            "blood_pressure": bp,
            "cholesterol_level": chol,
            "examination_patient": exam_summary
        },
        "logs": [
            "Physician Agent Physical Examination activated.",
            "Measuring patient vitals and cardiovascular attributes...",
            f"Vitals check completed - Fever: {fever}, BP: {bp}, Cholesterol: {chol}.",
            exam_summary
        ]
    })

    # ------------------ STEP 4: PHYSICIAN - DIAGNOSIS & ROUTING ------------------
    # Routing logic matches the project rule:
    # If severe symptoms or 'Outcome Variable' == 'Positive' in the records, route for X-Ray
    disease = patient.get('Disease', 'Unknown')
    outcome = patient.get('Outcome Variable', 'Unknown')
    
    # We trigger Radiologist if outcome == 'Positive' or difficulty breathing is present
    should_do_xray = (outcome == 'Positive') or (patient.get('Difficulty Breathing') == 'Yes')
    diagnosis_route = "Make X-ray for Chest" if should_do_xray else "Rest to Recover"
    diagnosis_desc = f"Disease: {disease} Outcome Variable: {outcome} and Final diagnosis: {diagnosis_route}"
    
    steps.append({
        "node": "physician_diagnosis_node",
        "title": "Physician Agent Node (Diagnosis & Routing)",
        "status": "success",
        "outputs": {
            "disease": disease,
            "outcome_variable": outcome,
            "diagnosis_patient": diagnosis_desc,
            "diagnosis": diagnosis_route,
            "should_do_xray": should_do_xray
        },
        "logs": [
            "Physician Agent Clinical Diagnosis activated.",
            f"Analyzing clinical correlation (Suspected Disease: {disease}, Database Outcome Code: {outcome}).",
            f"Applying routing rule condition: {diagnosis_route}",
            f"Routing instruction: {'FORWARD TO RADIOLOGIST' if should_do_xray else 'DISCHARGE PATIENT WITH REST RECOMMENDATION'}"
        ]
    })

    # ------------------ STEP 5: RADIOLOGIST DIAGNOSTIC (CONDITIONAL) ------------------
    radiologist_triggered = False
    prediction = "N/A"
    
    if should_do_xray:
        radiologist_triggered = True
        # If the dataset Outcome Variable is Positive, predict Pneumonia. Else predict Normal.
        # This keeps consistency with standard diagnostic mock metrics.
        prediction = "PNEUMONIA" if outcome == "Positive" else "NORMAL"
        
        steps.append({
            "node": "radiologist_agent_node",
            "title": "Radiologist Agent Node",
            "status": "success",
            "outputs": {
                "prediction": prediction,
                "scan_image": "chest_xray_placeholder.png"
            },
            "logs": [
                "Radiologist Agent Node activated via conditional edge.",
                "Loading Chest X-Ray scan image...",
                "Running lxyuan/vit-xray-pneumonia-classification Vision Transformer pipeline...",
                "Running image preprocessing (resize to 224x224, normalization)...",
                f"Classification model outcome: {prediction}",
                f"Diagnostic result finalized: Chest scan indicates status {prediction}."
            ]
        })
    
    return {
        "patient": patient,
        "verified": True,
        "radiologist_triggered": radiologist_triggered,
        "steps": steps,
        "final_status": "Completed"
    }

# Mount static files (will hold index.html, style.css, app.js)
static_dir = os.path.join(BASE_DIR, "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

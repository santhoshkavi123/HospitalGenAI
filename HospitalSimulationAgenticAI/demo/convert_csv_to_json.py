import os
import json
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
CSV_PATH = os.path.join(PROJECT_DIR, "datasets", "input", "processed", "updated_patient_df.csv")
JSON_OUTPUT_PATH = os.path.join(BASE_DIR, "static", "patients.json")

def convert():
    print(f"Reading CSV from: {CSV_PATH}")
    if not os.path.exists(CSV_PATH):
        if os.path.exists(JSON_OUTPUT_PATH):
            print(f"Source CSV not found at {CSV_PATH}, but {JSON_OUTPUT_PATH} already exists. Using existing JSON.")
            return
        raise FileNotFoundError(f"Source CSV not found at {CSV_PATH}")

    patients = []
    with open(CSV_PATH, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            # Parse Age as integer
            try:
                age = int(row.get("Age", 30))
            except ValueError:
                age = 30

            patient_id = row.get("Patient_ID")
            if not patient_id:
                # If Patient_ID is blank, skip or assign mock
                patient_id = f"PAT_{idx}"

            # Make a clean dictionary representing the patient
            patient = {
                "Patient_ID": patient_id,
                "First_Name": row.get("First_Name", "Unknown"),
                "Last_Name": row.get("Last_Name", "Unknown"),
                "Full_Name": row.get("Full_Name", f"{row.get('First_Name', '')}_{row.get('Last_Name', '')}".strip()),
                "Gender": row.get("Gender", "Unknown"),
                "Age": age,
                "Disease": row.get("Disease", "Unknown"),
                "Fever": row.get("Fever", "No"),
                "Cough": row.get("Cough", "No"),
                "Fatigue": row.get("Fatigue", "No"),
                "Difficulty Breathing": row.get("Difficulty Breathing", "No"),
                "Blood Pressure": row.get("Blood Pressure", "Normal"),
                "Cholesterol Level": row.get("Cholesterol Level", "Normal"),
                "Outcome Variable": row.get("Outcome Variable", "Negative")
            }
            patients.append(patient)

    # Ensure static directory exists
    os.makedirs(os.path.dirname(JSON_OUTPUT_PATH), exist_ok=True)
    
    with open(JSON_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(patients, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully wrote {len(patients)} records to {JSON_OUTPUT_PATH}")

if __name__ == "__main__":
    convert()

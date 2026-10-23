import os
import io
import json
import base64
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .ai_engine import (
    preprocess_image,
    check_image_quality,
    analyze_cataract_image,
    generate_gradcam_overlay,
    compare_longitudinal_images,
    calculate_risk_and_followup
)
from .seed_data import init_sample_images
from .report_generator import generate_clinical_report_html, generate_patient_summary_html

app = FastAPI(title="AI-Based Cataract Clinical Decision Support System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "samples")
UPLOADS_DIR = os.path.join(BASE_DIR, "data", "uploads")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Initialize sample data
sample_paths = init_sample_images(DATA_DIR)

# In-memory storage with seed demo patients
PATIENTS_DB = {
    "PAT-2026-0842": {
        "id": "PAT-2026-0842",
        "name": "Rajesh Kumar",
        "age": 62,
        "gender": "Male",
        "monitored_eye": "OD (Right Eye)",
        "has_diabetes": True,
        "steroid_use": False,
        "risk_factors": "Type 2 Diabetes (HbA1c: 7.1%), Age > 60",
        "visits": [
            {
                "visit_id": "V-001",
                "visit_number": 1,
                "visit_date": "2026-08-12",
                "image_filename": "sample_early.jpg",
                "image_path": sample_paths["early"],
                "analysis": {
                    "detection": "Cataract Detected",
                    "confidence": 94.2,
                    "severity": "Early (Grade 1 Nuclear Opacity)",
                    "severity_grade": 1,
                    "opacity_percentage": 28.5,
                    "clarity_score": 96.0,
                    "progression_trend": "Baseline Visit Established",
                    "trend_badge": "baseline",
                    "ssim_score": 1.0,
                    "risk_score": 0.28,
                    "risk_level": "Low Risk",
                    "risk_badge": "low",
                    "follow_up_interval_months": 9,
                    "recommended_follow_up_month": "May 2027",
                    "delta_description": "Initial baseline established. Crystalline lens exhibits early nuclear opacity."
                },
                "doctor_notes": "Mild nuclear sclerosis. Visual acuity 20/30. Advised UV protection. Schedule regular follow-up.",
                "doctor_approved": True
            }
        ]
    },
    "PAT-2026-0193": {
        "id": "PAT-2026-0193",
        "name": "Sunita Devi",
        "age": 58,
        "gender": "Female",
        "monitored_eye": "OS (Left Eye)",
        "has_diabetes": False,
        "steroid_use": False,
        "risk_factors": "No major systemic risk factors",
        "visits": [
            {
                "visit_id": "V-002",
                "visit_number": 1,
                "visit_date": "2026-05-10",
                "image_filename": "sample_normal.jpg",
                "image_path": sample_paths["normal"],
                "analysis": {
                    "detection": "Normal",
                    "confidence": 96.4,
                    "severity": "Normal (No Cataract)",
                    "severity_grade": 0,
                    "opacity_percentage": 4.2,
                    "clarity_score": 98.0,
                    "progression_trend": "Baseline Visit Established",
                    "trend_badge": "baseline",
                    "ssim_score": 1.0,
                    "risk_score": 0.10,
                    "risk_level": "Low Risk",
                    "risk_badge": "low",
                    "follow_up_interval_months": 12,
                    "recommended_follow_up_month": "May 2027",
                    "delta_description": "Crystalline lens is transparent with no nuclear or cortical opacification."
                },
                "doctor_notes": "Healthy clear ocular lens. Visual acuity 20/20 in left eye.",
                "doctor_approved": True
            }
        ]
    }
}

# -------------------------------------------------------------------------
# API ENDPOINTS
# -------------------------------------------------------------------------

@app.get("/api/patients")
def list_patients():
    return list(PATIENTS_DB.values())

@app.get("/api/patients/{patient_id}")
def get_patient(patient_id: str):
    if patient_id not in PATIENTS_DB:
        raise HTTPException(status_code=404, detail="Patient not found")
    return PATIENTS_DB[patient_id]

@app.post("/api/patients")
def create_patient(
    patient_id: str = Form(...),
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    monitored_eye: str = Form("OD (Right Eye)"),
    has_diabetes: bool = Form(False),
    risk_factors: str = Form("None")
):
    if patient_id in PATIENTS_DB:
        raise HTTPException(status_code=400, detail="Patient ID already exists")
    
    rf_lower = (risk_factors or "").lower()
    inferred_diabetes = has_diabetes or ("diabetes" in rf_lower)
    inferred_steroids = "steroid" in rf_lower or "corticosteroid" in rf_lower

    PATIENTS_DB[patient_id] = {
        "id": patient_id,
        "name": name,
        "age": age,
        "gender": gender,
        "monitored_eye": monitored_eye,
        "has_diabetes": inferred_diabetes,
        "steroid_use": inferred_steroids,
        "risk_factors": risk_factors,
        "visits": []
    }
    return PATIENTS_DB[patient_id]

@app.get("/api/image-preview")
def get_image_preview(path: str, gradcam: bool = False, severity_grade: int = 1):
    """
    Returns image or Grad-CAM overlay encoded as base64 for direct browser rendering.
    """
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image file not found")
    img = Image.open(path)
    if gradcam:
        img = generate_gradcam_overlay(img, is_cataract=(severity_grade > 0), severity_grade=severity_grade)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return {"base64": f"data:image/jpeg;base64,{img_b64}"}

@app.post("/api/visits/analyze")
async def analyze_and_record_visit(
    patient_id: str = Form(...),
    visit_date: str = Form(...),
    demo_stage: str = Form(None),
    file: UploadFile = File(None)
):
    if patient_id not in PATIENTS_DB:
        raise HTTPException(status_code=404, detail="Patient not found")
        
    patient = PATIENTS_DB[patient_id]
    
    # Ingest image
    if file and file.filename:
        filename = f"{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        save_path = os.path.join(UPLOADS_DIR, filename)
        content = await file.read()
        image_pil = Image.open(io.BytesIO(content))
        image_pil.save(save_path)
    elif demo_stage and demo_stage in sample_paths:
        save_path = sample_paths[demo_stage]
        image_pil = Image.open(save_path)
        filename = os.path.basename(save_path)
    else:
        # Default fallback to moderate sample
        save_path = sample_paths["moderate"]
        image_pil = Image.open(save_path)
        filename = "sample_moderate.jpg"

    # 1. Quality Check & Preprocessing
    quality = check_image_quality(image_pil)
    preprocessed = preprocess_image(image_pil)
    
    # 2. Cataract Detection & Severity
    cataract_analysis = analyze_cataract_image(preprocessed)
    cataract_analysis["clarity_score"] = quality["clarity_score"]
    
    # 3. Retrieve Previous Visit & Longitudinal Comparison
    prev_visit = patient["visits"][-1] if patient["visits"] else None
    prev_img_pil = Image.open(prev_visit["image_path"]) if prev_visit else None
    prev_grade = prev_visit["analysis"]["severity_grade"] if prev_visit else 0
    
    longitudinal_res = compare_longitudinal_images(
        prev_img_pil,
        preprocessed,
        prev_grade,
        cataract_analysis["severity_grade"]
    )
    
    # 4. Multimodal Progression Risk & Follow-up Calculation
    risk_followup = calculate_risk_and_followup(
        cataract_analysis,
        longitudinal_res,
        patient,
        visit_date
    )
    
    # Merge full analysis payload
    full_analysis = {
        **cataract_analysis,
        **longitudinal_res,
        **risk_followup
    }
    
    # Generate Grad-CAM image and save
    gradcam_img = generate_gradcam_overlay(
        image_pil, 
        is_cataract=cataract_analysis["is_cataract"],
        severity_grade=cataract_analysis["severity_grade"]
    )
    gradcam_filename = f"gradcam_{filename}"
    gradcam_path = os.path.join(UPLOADS_DIR, gradcam_filename)
    gradcam_img.save(gradcam_path, "JPEG")
    full_analysis["gradcam_path"] = gradcam_path
    
    # Append visit record
    visit_number = len(patient["visits"]) + 1
    new_visit = {
        "visit_id": f"V-00{visit_number + len(PATIENTS_DB)}",
        "visit_number": visit_number,
        "visit_date": visit_date,
        "image_filename": filename,
        "image_path": save_path,
        "analysis": full_analysis,
        "doctor_notes": "Clinical review pending. Verified AI image quality.",
        "doctor_approved": False
    }
    
    patient["visits"].append(new_visit)
    
    return {
        "status": "success",
        "patient_id": patient_id,
        "visit": new_visit,
        "previous_visit": prev_visit
    }

@app.post("/api/visits/{patient_id}/{visit_id}/approve")
def approve_visit(patient_id: str, visit_id: str, notes: str = Form(...)):
    if patient_id not in PATIENTS_DB:
        raise HTTPException(status_code=404, detail="Patient not found")
    patient = PATIENTS_DB[patient_id]
    for v in patient["visits"]:
        if v["visit_id"] == visit_id:
            v["doctor_approved"] = True
            v["doctor_notes"] = notes
            return {"status": "approved", "visit": v}
    raise HTTPException(status_code=404, detail="Visit not found")

@app.get("/api/reports/clinical/{patient_id}/{visit_id}")
def get_clinical_report(patient_id: str, visit_id: str):
    if patient_id not in PATIENTS_DB:
        raise HTTPException(status_code=404, detail="Patient not found")
    patient = PATIENTS_DB[patient_id]
    target_visit = next((v for v in patient["visits"] if v["visit_id"] == visit_id), None)
    if not target_visit:
        raise HTTPException(status_code=404, detail="Visit not found")
        
    html = generate_clinical_report_html(patient, target_visit, target_visit["analysis"], target_visit.get("doctor_notes", ""))
    return HTMLResponse(content=html)

@app.get("/api/reports/patient/{patient_id}/{visit_id}")
def get_patient_summary(patient_id: str, visit_id: str):
    if patient_id not in PATIENTS_DB:
        raise HTTPException(status_code=404, detail="Patient not found")
    patient = PATIENTS_DB[patient_id]
    target_visit = next((v for v in patient["visits"] if v["visit_id"] == visit_id), None)
    if not target_visit:
        raise HTTPException(status_code=404, detail="Visit not found")
        
    html = generate_patient_summary_html(patient, target_visit, target_visit["analysis"], target_visit.get("doctor_notes", ""))
    return HTMLResponse(content=html)

# Serve Frontend static directory
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

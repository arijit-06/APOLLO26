import uuid
import os
import json
import logging
import shutil
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List

# Ensure working directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("reports", exist_ok=True)

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ml_runner import execute_ml_pipeline

router = APIRouter()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# In-memory mock database to track job status. 
# Result will be fetched from disk.
JOBS_DB: Dict[str, Dict[str, Any]] = {}

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str

@router.post("/upload-ate-log", response_model=JobResponse)
async def upload_ate_log(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
    
    job_id = str(uuid.uuid4())
    file_path = os.path.join("uploads", f"{job_id}_{file.filename}")
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.info(f"Saved file {file.filename} physically to {file_path}. Assigned Job ID: {job_id}")
    except Exception as e:
        logging.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")
    finally:
        file.file.close()
        
    JOBS_DB[job_id] = {"status": "uploaded", "file_path": file_path}
    
    return JobResponse(job_id=job_id, status="uploaded", message=f"File saved successfully. Job ID: {job_id}")

@router.post("/run-analysis/{job_id}", response_model=JobResponse)
async def run_analysis(job_id: str, background_tasks: BackgroundTasks):
    if job_id not in JOBS_DB:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    
    if JOBS_DB[job_id]["status"] in ["processing", "completed"]:
        raise HTTPException(status_code=400, detail="Analysis is already running or completed.")
    
    file_path = JOBS_DB[job_id]["file_path"]
    JOBS_DB[job_id]["status"] = "processing"
    
    # We pass the update_job_status callback logic effectively into the background task
    def background_ml_task():
        try:
            execute_ml_pipeline(file_path, job_id)
            JOBS_DB[job_id]["status"] = "completed"
        except Exception as e:
            logging.error(f"Background ML task failed for job {job_id}: {e}")
            JOBS_DB[job_id]["status"] = "failed"
            
    background_tasks.add_task(background_ml_task)
    
    return JobResponse(job_id=job_id, status="processing", message="Real ML Pipeline executing asynchronously.")

@router.get("/lot-summary/{job_id}")
async def get_lot_summary(job_id: str):
    if job_id not in JOBS_DB:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    
    job_info = JOBS_DB[job_id]
    
    if job_info["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job is currently {job_info['status']}. Wait for completion.")
    
    # Read the actual physical JSON output file from ML core
    report_path = os.path.join("reports", f"{job_id}_final_report.json")
    if not os.path.exists(report_path):
        JOBS_DB[job_id]["status"] = "failed"
        raise HTTPException(status_code=500, detail="Report file not found. Pipeline may have crashed silently.")
        
    try:
        with open(report_path, "r") as f:
            result = json.load(f)
        return result
    except Exception as e:
        logging.error(f"Failed to read report {report_path}: {e}")
        raise HTTPException(status_code=500, detail="Failed to parse final report.")

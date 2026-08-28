import requests
import time
import json
import os

API_BASE_URL = "http://localhost:8000/api"

def run_simulation():
    print("=== ChronoDrift-AI E2E Integration Simulation ===")
    
    # 1. Generate Synthetic CSV Payload
    print("\n[1] Generating synthetic ATE CSV test log...")
    csv_filename = "synthetic_ate_log.csv"
    with open(csv_filename, "w") as f:
        f.write("die_id,time_h,I_DDQ,Leakage,Delay\n")
        f.write("COMP_1,0,5.1,10.2,201.0\n")
        f.write("COMP_1,168,5.4,14.5,205.0\n")
    
    # 2. POST the file to FastAPI
    print("\n[2] Uploading CSV to FastAPI backend...")
    try:
        with open(csv_filename, "rb") as f:
            files = {"file": (csv_filename, f, "text/csv")}
            response = requests.post(f"{API_BASE_URL}/upload-ate-log", files=files)
            
        if response.status_code != 200:
            print(f"Upload failed: {response.text}")
            return
            
        job_data = response.json()
        job_id = job_data["job_id"]
        print(f"Upload Success. Job ID: {job_id}")
        
        # 3. Trigger Analysis
        print("\n[3] Triggering ML Analysis Pipeline...")
        trigger_resp = requests.post(f"{API_BASE_URL}/run-analysis/{job_id}")
        if trigger_resp.status_code != 200:
            print(f"Failed to trigger analysis: {trigger_resp.text}")
            return
            
        print("Analysis triggered. Background ML tasks are executing asynchronously.")
        
        # 4. Polling for final results
        print("\n[4] Polling for final Lot Summary Report...")
        max_attempts = 15
        for attempt in range(max_attempts):
            time.sleep(2) # Poll every 2 seconds
            poll_resp = requests.get(f"{API_BASE_URL}/lot-summary/{job_id}")
            
            if poll_resp.status_code == 200:
                summary = poll_resp.json()
                if summary["status"] == "completed":
                    print("\n" + "="*40)
                    print("         INTEGRATION SUCCESS")
                    print("="*40)
                    print(f"Total Components: {summary['total_components_processed']}")
                    print(f"Gross Failures:   {summary['gross_failures']}")
                    print(f"Latent Defects:   {summary['latent_defects']}")
                    print("-" * 40)
                    print("Sample Anomaly Root-Cause Explanation:")
                    if summary["anomalies"]:
                        print(json.dumps(summary["anomalies"][0], indent=4))
                    else:
                        print("No anomalies flagged.")
                    break
            elif poll_resp.status_code == 400 and "Wait for completion" in poll_resp.text:
                print(f"Attempt {attempt+1}/{max_attempts}: ML Core still processing...")
            else:
                print(f"Unexpected response: {poll_resp.text}")
                break
        else:
            print("Polling timed out. Job did not complete in time.")
            
    except requests.exceptions.ConnectionError:
        print("\nCRITICAL ERROR: Could not connect to FastAPI server.")
        print("Please ensure you run 'uvicorn backend.main:app --reload' in a separate terminal before executing this script.")
    finally:
        # Cleanup
        if os.path.exists(csv_filename):
            os.remove(csv_filename)

if __name__ == "__main__":
    run_simulation()

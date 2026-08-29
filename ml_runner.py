import numpy as np
import pandas as pd
import json
import logging
import traceback
import sys
import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

try:
    from ml_preprocessing import robust_dpat_scaling, interpolate_trajectories
    from tier1_statistical import run_tier1_filter
    from tier2_autoencoder import ChronoDriftAnomalyEngine
    from tier3_explainability import calculate_feature_attribution, generate_root_cause_json
except ImportError as e:
    logging.warning(f"Could not import ML modules. Ensure they are in root. Error: {e}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_csv_to_tensor(csv_path: str):
    """
    Ingests raw ATE CSV test logs and transforms them into a strictly ordered 
    (num_samples, 4, 3) 3D tensor suitable for the PBAA - Pre-flight Burn-In Anomaly Analysis ML pipeline.
    """
    expected_times = [0, 24, 96, 168]
    df = pd.read_csv(csv_path)
    
    required_cols = ['die_id', 'lot_id', 'timestamp', 'iddq', 'leakage_current', 'prop_delay']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Malformed CSV: Missing required column '{col}'")
            
    df = df[df['timestamp'].isin(expected_times)].copy()
    df.sort_values(by=['die_id', 'timestamp'], inplace=True)
    
    die_counts = df['die_id'].value_counts()
    complete_dies = die_counts[die_counts == 4].index
    
    df_clean = df[df['die_id'].isin(complete_dies)].copy()
    if df_clean.empty:
        raise ValueError("FATAL: No complete component sequences (0h, 24h, 96h, 168h) found in the CSV.")
        
    metadata = df_clean.iloc[::4][['die_id', 'lot_id']]
    die_ids = metadata['die_id'].values
    lot_ids = metadata['lot_id'].values
    
    feature_cols = ['iddq', 'leakage_current', 'prop_delay']
    flat_features = df_clean[feature_cols].values
    
    num_samples = len(die_ids)
    raw_tensor = flat_features.reshape((num_samples, 4, 3))
    
    return raw_tensor, lot_ids, die_ids

def execute_ml_pipeline(csv_path: str, job_id: str):
    logging.info(f"[Job {job_id}] Starting REAL ML Pipeline execution for {csv_path}")
    
    DATA_REPORTS_DIR = os.path.join(BASE_DIR, "data", "reports")
    os.makedirs(DATA_REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(DATA_REPORTS_DIR, f"{job_id}_final_report.json")
    
    try:
        logging.info(f"[Job {job_id}] Step 0: Parsing physical CSV {csv_path}...")
        raw_data, lot_ids, die_ids = parse_csv_to_tensor(csv_path)
        discrete_times = np.array([0, 24, 96, 168])
        
        logging.info(f"[Job {job_id}] Step 1: Running DPAT & PCHIP Interpolation...")
        continuous_times = np.arange(0, 169)
        scaled_data = robust_dpat_scaling(raw_data, lot_ids)
        dense_data = interpolate_trajectories(scaled_data, discrete_times, continuous_times)
        velocities = np.gradient(dense_data, axis=1)
        tensor_3d = np.concatenate([dense_data, velocities], axis=2)
        
        num_features = tensor_3d.shape[2]
        time_steps = tensor_3d.shape[1]
        
        logging.info(f"[Job {job_id}] Step 2: Executing Tier 1 Mahalanobis Filter...")
        t1_results = run_tier1_filter(tensor_3d)
        gross_fail_mask = t1_results["routing_mask"]
        gross_fail_indices = np.where(gross_fail_mask)[0]
        passing_indices = np.where(~gross_fail_mask)[0]
        
        logging.info(f"[Job {job_id}] Step 3: Pushing passing components through LSTM...")
        passing_tensor = tensor_3d[passing_indices]
        engine = ChronoDriftAnomalyEngine(seq_len=time_steps, n_features=num_features, hidden_dim=32, latent_dim=8)
        latent_mask, errors, threshold = engine.evaluate_anomalies(passing_tensor, threshold_percentile=95.0)
        latent_defect_indices = passing_indices[latent_mask]
        
        logging.info(f"[Job {job_id}] Step 4: Generating XAI Explainability...")
        feature_names = ["I_DDQ_Magnitude", "Leakage_Magnitude", "Delay_Magnitude", "I_DDQ_Velocity", "Leakage_Velocity", "Delay_Velocity"]
        anomalies_list = []
        
        if len(latent_defect_indices) > 0:
            flagged_tensor = torch.tensor(tensor_3d[latent_defect_indices], dtype=torch.float32)
            baseline_tensor = torch.zeros((1, time_steps, num_features))
            
            try:
                attributions = calculate_feature_attribution(
                    model=engine.model, 
                    background_data=baseline_tensor, 
                    flagged_samples=flagged_tensor, 
                    device=engine.device
                )
                component_ids = [str(die_ids[idx]) for idx in latent_defect_indices]
                json_string = generate_root_cause_json(attributions, component_ids, feature_names)
                anomalies_list = json.loads(json_string)
            except Exception as e:
                logging.error(f"[Job {job_id}] Explainability failed: {str(e)}")
                anomalies_list = [{"die_id": str(die_ids[idx]), "primary_failure": "Unknown", "contribution_percentage": 0.0} for idx in latent_defect_indices]
        
        final_report = {
            "job_id": job_id,
            "status": "completed",
            "total_components_processed": len(die_ids),
            "gross_failures": len(gross_fail_indices),
            "latent_defects": len(latent_defect_indices),
            "anomalies": anomalies_list
        }
        
        with open(report_path, "w") as f:
            json.dump(final_report, f, indent=4)
            
        logging.info(f"[Job {job_id}] Execution SUCCESS. Report saved to {report_path}")
        return final_report
        
    except MemoryError:
        logging.error(f"[Job {job_id}] FATAL OOM ERROR: Tensor shapes exceeded available RAM.")
        raise
    except ValueError as ve:
        logging.error(f"[Job {job_id}] SHAPE MISMATCH ERROR: {ve}")
        raise
    except Exception as e:
        logging.error(f"[Job {job_id}] UNEXPECTED PIPELINE FAILURE:\n{traceback.format_exc()}")
        raise

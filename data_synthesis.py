import numpy as np
import pandas as pd
import json

def generate_aerospace_burn_in_data(num_samples=10000, anomaly_rate=0.05):
    np.random.seed(42)
    
    # Simulation Parameters
    time_steps = np.array([0, 24, 96, 168])
    num_time_steps = len(time_steps)
    num_features = 3 # I_DDQ (uA), I_Leakage (nA), Prop_Delay (ps)
    
    # 1. Generate Baseline Lot-to-Lot Variations (T=0)
    # I_DDQ: Normal(5.0 uA, 0.2)
    baseline_iddq = np.random.normal(loc=5.0, scale=0.2, size=(num_samples, 1))
    # I_Leakage: Log-Normal to ensure strict positivity, centered around 10.0 nA
    baseline_leak = np.random.lognormal(mean=np.log(10.0), sigma=0.1, size=(num_samples, 1))
    # Prop_Delay: Normal(200.0 ps, 5.0)
    baseline_delay = np.random.normal(loc=200.0, scale=5.0, size=(num_samples, 1))
    
    baselines = np.concatenate([baseline_iddq, baseline_leak, baseline_delay], axis=1)
    
    # Initialize the 3D array: (samples, time_steps, features)
    data_3d = np.zeros((num_samples, num_time_steps, num_features))
    
    # 2. Populate Normal Degradation Physics
    for t_idx, t in enumerate(time_steps):
        # Thermal noise / measurement jitter (Gaussian)
        # We ensure noise is small enough not to create negative currents.
        # For Leakage, we use multiplicative noise to preserve positivity.
        noise_iddq = np.random.normal(0, 0.02, size=num_samples)
        noise_leak_mult = np.random.normal(1.0, 0.01, size=num_samples) # 1% variance
        noise_delay = np.random.normal(0, 0.5, size=num_samples)
        
        # Minor, stable baseline shift over time for normal aging
        aging_factor_iddq = 0.001 * t
        aging_factor_delay = 0.02 * t
        
        data_3d[:, t_idx, 0] = baselines[:, 0] + aging_factor_iddq + noise_iddq
        data_3d[:, t_idx, 1] = baselines[:, 1] * noise_leak_mult + (0.005 * t)
        data_3d[:, t_idx, 2] = baselines[:, 2] + aging_factor_delay + noise_delay

    # Enforce strict positivity for all physics constraints (safety catch)
    data_3d = np.abs(data_3d)

    # 3. Inject Latent Defects (Anomalies)
    num_anomalies = int(num_samples * anomaly_rate)
    anomaly_indices = np.random.choice(num_samples, num_anomalies, replace=False)
    
    # Split anomalies into three distinct failure mechanisms
    mech_1 = anomaly_indices[:num_anomalies//3]         # Slow exponential drift (I_DDQ)
    mech_2 = anomaly_indices[num_anomalies//3:2*num_anomalies//3] # 96h Step-jump (I_Leakage)
    mech_3 = anomaly_indices[2*num_anomalies//3:]       # Accelerating delay
    
    for t_idx, t in enumerate(time_steps):
        if t == 0:
            continue
            
        # Mechanism 1: Monotonic Exponential Drift (Subtle enough to stay within 3-sigma limit)
        # e.g., max drift at 168h is ~0.4 uA, which keeps it largely within the general population bounds
        data_3d[mech_1, t_idx, 0] += 0.005 * (np.exp(0.025 * t) - 1)
        
        # Mechanism 2: Sudden Step-Jump at 96h mark
        if t >= 96:
            # Jump by ~1.5 nA. Small enough to avoid static thresholds, big enough for sequence models
            data_3d[mech_2, t_idx, 1] += 1.5 + np.random.normal(0, 0.1, size=len(mech_2))
            
        # Mechanism 3: Accelerating Propagation Delay (Quadratic)
        # Monotonically increasing degradation curve
        data_3d[mech_3, t_idx, 2] += 0.0003 * (t ** 2)

    # Ensure strictly monotonic degradation for the anomalous parameters to pass Physics Check
    for t_idx in range(1, num_time_steps):
        data_3d[mech_1, t_idx, 0] = np.maximum(data_3d[mech_1, t_idx, 0], data_3d[mech_1, t_idx-1, 0])
        data_3d[mech_2, t_idx, 1] = np.maximum(data_3d[mech_2, t_idx, 1], data_3d[mech_2, t_idx-1, 1])
        data_3d[mech_3, t_idx, 2] = np.maximum(data_3d[mech_3, t_idx, 2], data_3d[mech_3, t_idx-1, 2])

    labels = np.zeros(num_samples, dtype=int)
    labels[anomaly_indices] = 1

    # 4. Formatting - JSON Export preparation
    json_export_list = []
    for i in range(num_samples):
        measurements = []
        for t_idx, t in enumerate(time_steps):
            measurements.append({
                "time_h": int(t),
                "I_DDQ_uA": float(data_3d[i, t_idx, 0]),
                "I_Leakage_nA": float(data_3d[i, t_idx, 1]),
                "Delay_ps": float(data_3d[i, t_idx, 2])
            })
        
        json_export_list.append({
            "component_id": f"SN_{i:06d}",
            "is_defective": bool(labels[i]), # Ground truth for validation only
            "measurements": measurements
        })

    json_payload = {
        "metadata": {
            "description": "Aerospace Component Burn-in Test Data",
            "total_samples": num_samples,
            "time_intervals_h": time_steps.tolist(),
            "features": ["I_DDQ_uA", "I_Leakage_nA", "Delay_ps"]
        },
        "components": json_export_list
    }

    return data_3d, labels, json_payload

if __name__ == "__main__":
    print("Generating synthetic aerospace burn-in dataset...")
    
    # Generate data
    X_3d, y_labels, json_data = generate_aerospace_burn_in_data()
    
    # 1. Save 3D NumPy array for TensorFlow/PyTorch sequence models
    np.save("burn_in_sequences.npy", X_3d)
    np.save("burn_in_labels.npy", y_labels)
    print(f"NumPy arrays saved. X shape: {X_3d.shape}, y shape: {y_labels.shape}")
    
    # 2. Save Structured JSON for React/FastAPI frontend
    with open("burn_in_frontend.json", "w") as f:
        json.dump(json_data, f, indent=2)
    print("JSON payload generated and saved to 'burn_in_frontend.json'.")

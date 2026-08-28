import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import logging
import sys

# Configure logging for production readiness
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def load_data(npy_file="burn_in_sequences.npy", num_lots=50):
    """Loads sparse burn-in data and assigns mock lots."""
    logging.info(f"Loading raw sequences from {npy_file}...")
    try:
        data_3d = np.load(npy_file)
        num_samples = data_3d.shape[0]
        # Randomly assign components to lots for DPAT
        lot_ids = np.random.randint(0, num_lots, size=num_samples)
        time_steps = np.array([0, 24, 96, 168])
        logging.info(f"Loaded {num_samples} components.")
        return data_3d, lot_ids, time_steps
    except FileNotFoundError:
        logging.error(f"File {npy_file} not found. Generating mock data instead.")
        num_samples = 10000
        lot_ids = np.random.randint(0, num_lots, size=num_samples)
        time_steps = np.array([0, 24, 96, 168])
        data_3d = np.zeros((num_samples, len(time_steps), 3))
        for t_idx, t in enumerate(time_steps):
            data_3d[:, t_idx, 0] = 5.0 + 0.001 * t + np.random.normal(0, 0.1, num_samples)
            data_3d[:, t_idx, 1] = 10.0 + 0.05 * t + np.random.normal(0, 0.5, num_samples)
            data_3d[:, t_idx, 2] = 200.0 + 0.1 * t + np.random.normal(0, 2.0, num_samples)
        return data_3d, lot_ids, time_steps

def robust_dpat_scaling(data_3d, lot_ids):
    """
    Applies Dynamic Part Average Testing (DPAT) using Intra-lot Robust Z-Score.
    Avoids data leakage by isolating median/IQR calculations per lot.
    """
    logging.info("Applying Dynamic Part Average Testing (DPAT) Scaling...")
    num_samples, num_time_steps, num_features = data_3d.shape
    
    # Reshape to 2D for pandas processing
    flattened_data = data_3d.reshape((num_samples * num_time_steps, num_features))
    
    # Expand lot_ids to match the flattened time steps
    expanded_lots = np.repeat(lot_ids, num_time_steps)
    
    df = pd.DataFrame(flattened_data, columns=['IDDQ', 'Leakage', 'Delay'])
    df['lot_id'] = expanded_lots
    
    # Define robust scaling function with epsilon for numerical stability
    def scale_group(group):
        eps = 1e-8
        median = group.median()
        iqr = group.quantile(0.75) - group.quantile(0.25)
        return (group - median) / (iqr + eps)
    
    # Apply grouped scaling (prevents data leakage between lots)
    scaled_df = df.groupby('lot_id')[['IDDQ', 'Leakage', 'Delay']].transform(scale_group)
    
    # Reshape back to 3D tensor
    scaled_3d = scaled_df.values.reshape((num_samples, num_time_steps, num_features))
    return scaled_3d

def interpolate_trajectories(scaled_data, original_times, target_times):
    """
    Upsamples sparse time intervals into continuous dense trajectories.
    Uses PCHIP to preserve monotonicity and eliminate Runge's phenomenon overshoots.
    """
    logging.info(f"Interpolating discrete times {original_times.tolist()} to {len(target_times)} continuous steps...")
    num_samples, _, num_features = scaled_data.shape
    
    dense_data = np.zeros((num_samples, len(target_times), num_features))
    
    for i in range(num_features):
        feature_slice = scaled_data[:, :, i]
        interpolator = PchipInterpolator(original_times, feature_slice, axis=1)
        dense_data[:, :, i] = interpolator(target_times)
        
    return dense_data

def build_lstm_tensor_pipeline():
    """Main pipeline execution function."""
    
    # 1. Load Data
    raw_data, lot_ids, discrete_times = load_data()
    continuous_times = np.arange(0, 169) # Hours 0 through 168 (169 steps)
    
    # 2. DPAT Robust Normalization
    scaled_data = robust_dpat_scaling(raw_data, lot_ids)
    
    # 3. Continuous Trajectory Interpolation (PCHIP)
    dense_data = interpolate_trajectories(scaled_data, discrete_times, continuous_times)
    
    # 4. Derivative Feature Extraction (Drift Velocity)
    logging.info("Extracting derivative features (drift velocity d/dt)...")
    velocities = np.gradient(dense_data, axis=1)
    
    # 5. Final Tensor Construction
    logging.info("Constructing final 3D tensor...")
    final_tensor = np.concatenate([dense_data, velocities], axis=2)
    
    num_samples = raw_data.shape[0]
    expected_shape = (num_samples, 169, 6)
    if final_tensor.shape != expected_shape:
        logging.error(f"Shape mismatch! Expected {expected_shape}, got {final_tensor.shape}")
        raise ValueError("Pipeline tensor construction failed.")
    
    if np.isnan(final_tensor).any():
        logging.error("NaNs detected in the final tensor!")
        raise ValueError("Mathematical instability (NaNs) in pipeline.")
        
    logging.info(f"Pipeline executed successfully. Final Tensor Shape: {final_tensor.shape}")
    return final_tensor

if __name__ == "__main__":
    ml_ready_tensor = build_lstm_tensor_pipeline()
    np.save("burn_in_lstm_autoencoder_tensor.npy", ml_ready_tensor)
    logging.info("Saved final tensor to burn_in_lstm_autoencoder_tensor.npy")

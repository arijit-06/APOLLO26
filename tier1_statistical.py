import numpy as np
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet
from scipy.stats import chi2
import logging
import sys
from typing import Tuple, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def extract_temporal_summaries(tensor_3d: np.ndarray) -> np.ndarray:
    """
    Flattens a 3D temporal tensor by extracting summary statistics per component.
    
    Args:
        tensor_3d: Input tensor of shape (num_samples, time_steps, num_features).
        
    Returns:
        np.ndarray: A 2D matrix of shape (num_samples, num_summary_features).
    """
    logging.info(f"Extracting temporal summaries from tensor of shape {tensor_3d.shape}...")
    
    # Feature 1: Final absolute magnitude (the state at 168h)
    final_magnitude = np.abs(tensor_3d[:, -1, :])
    
    # Feature 2: Maximum absolute drift/magnitude across the entire sequence
    max_magnitude = np.max(np.abs(tensor_3d), axis=1)
    
    # Feature 3: Variance across the time steps
    variance = np.var(tensor_3d, axis=1)
    
    # Concatenate all summaries along the feature axis
    summarized_2d = np.hstack([final_magnitude, max_magnitude, variance])
    logging.info(f"Summarization complete. Output shape: {summarized_2d.shape}")
    
    return summarized_2d

def apply_pca(summarized_2d: np.ndarray, variance_ratio: float = 0.95) -> Tuple[np.ndarray, PCA]:
    """
    Applies PCA dimensionality reduction to eliminate multi-collinearity.
    
    Args:
        summarized_2d: The extracted 2D summary features.
        variance_ratio: The percentage of variance to retain.
        
    Returns:
        Tuple containing the PCA-transformed matrix and the fitted PCA model.
    """
    logging.info(f"Applying PCA to retain {variance_ratio*100}% of variance...")
    pca = PCA(n_components=variance_ratio, svd_solver='full')
    pca_transformed = pca.fit_transform(summarized_2d)
    logging.info(f"PCA reduced dimensions from {summarized_2d.shape[1]} to {pca_transformed.shape[1]}")
    
    return pca_transformed, pca

def compute_mahalanobis_mask(pca_data: np.ndarray, percentile: float = 0.99) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Computes robust Mahalanobis distances and applies a Chi-Square dynamic threshold.
    
    Args:
        pca_data: The PCA-reduced 2D matrix.
        percentile: The Chi-Square distribution percentile for thresholding.
        
    Returns:
        Tuple containing the boolean mask (True = Gross Fail), distances, and threshold limit.
    """
    logging.info("Fitting Minimum Covariance Determinant (MinCovDet) for robust centroid estimation...")
    # Using MinCovDet prevents extreme outliers from skewing the covariance matrix
    robust_cov = MinCovDet()
    robust_cov.fit(pca_data)
    
    logging.info("Calculating Mahalanobis distances...")
    # mahalanobis() returns the squared Mahalanobis distance when used in sklearn MinCovDet
    squared_distances = robust_cov.mahalanobis(pca_data)
    
    # Degrees of freedom equals the number of features in the PCA transformed data
    df = pca_data.shape[1]
    
    # Set dynamic threshold using the Chi-Square distribution
    threshold = chi2.ppf(percentile, df)
    logging.info(f"Chi-Square Threshold ({percentile*100}th percentile, df={df}): {threshold:.2f}")
    
    # Create mask: True if distance > threshold (Gross Fail)
    routing_mask = squared_distances > threshold
    
    fail_count = np.sum(routing_mask)
    logging.info(f"Statistical filter identified {fail_count} Gross Failures ({(fail_count/len(routing_mask))*100:.2f}% of lot).")
    
    return routing_mask, squared_distances, threshold

def run_tier1_filter(tensor_3d: np.ndarray) -> Dict[str, Any]:
    """
    Executes the complete Tier 1 Statistical Filter pipeline.
    
    Args:
        tensor_3d: Raw input tensor from preprocessor (num_samples, time_steps, features).
        
    Returns:
        A dictionary containing the routing mask and pipeline metadata.
    """
    logging.info("--- Starting Tier 1 Statistical Filter ---")
    summarized_data = extract_temporal_summaries(tensor_3d)
    pca_data, pca_model = apply_pca(summarized_data)
    routing_mask, distances, threshold = compute_mahalanobis_mask(pca_data)
    
    logging.info("--- Tier 1 Statistical Filter Complete ---")
    return {
        "routing_mask": routing_mask,
        "distances": distances,
        "threshold": threshold,
        "pca_model": pca_model,
        "pca_data": pca_data
    }

if __name__ == '__main__':
    # Generate dummy 3D data simulating (10000 components, 169 time steps, 6 features)
    np.random.seed(42)
    num_samples = 10000
    time_steps = 169
    num_features = 6
    
    logging.info("Generating dummy input tensor for end-to-end testing...")
    dummy_tensor = np.random.normal(loc=0, scale=1.0, size=(num_samples, time_steps, num_features))
    
    # Inject synthetic "Gross Failures" (catastrophic anomalies)
    num_anomalies = 50
    anomaly_indices = np.random.choice(num_samples, num_anomalies, replace=False)
    # Add huge spikes to the anomalous components at the final time step
    dummy_tensor[anomaly_indices, -1, :] += np.random.normal(loc=20.0, scale=5.0, size=(num_anomalies, num_features))
    
    # Execute Pipeline
    results = run_tier1_filter(dummy_tensor)
    
    routing_mask = results["routing_mask"]
    
    # Validation
    true_positives = np.sum(routing_mask[anomaly_indices])
    logging.info(f"Test Validation: Filter caught {true_positives}/{num_anomalies} injected gross failures.")

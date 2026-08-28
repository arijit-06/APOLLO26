import torch
import torch.nn as nn
import numpy as np
import json
import logging
import sys
from typing import List, Dict, Any, Union

# Attempt to import captum, handle gracefully if missing
try:
    from captum.attr import IntegratedGradients
except ImportError:
    logging.warning("Captum library not found. Please install via: pip install captum")
    IntegratedGradients = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

class ReconstructionErrorWrapper(nn.Module):
    """
    A PyTorch wrapper module that returns the reconstruction MSE error for each sequence.
    This scalar output is required by Captum's Integrated Gradients to trace attributions
    back to the original input features.
    """
    def __init__(self, autoencoder_model: nn.Module):
        super().__init__()
        self.model = autoencoder_model
        self.model.eval() # Ensure the model is in evaluation mode
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
        Returns:
            torch.Tensor: The MSE reconstruction error per sequence, shape (batch_size, 1)
        """
        # Get reconstructed sequence
        reconstructed = self.model(x)
        # Calculate MSE per sequence (average across time steps and features)
        # Shape: (batch_size, seq_len, num_features) -> (batch_size,)
        error = torch.mean((x - reconstructed) ** 2, dim=[1, 2])
        # We need a 2D output for Captum target attribution, shape (batch_size, 1)
        return error.unsqueeze(1)

def calculate_feature_attribution(
    model: nn.Module, 
    background_data: torch.Tensor, 
    flagged_samples: torch.Tensor,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Computes Integrated Gradients attribution scores for flagged anomalies.
    
    Args:
        model: The trained PyTorch LSTM Autoencoder.
        background_data: A baseline reference tensor (e.g., median of normal data).
        flagged_samples: The anomalous sequences needing explanation.
        device: 'cuda' or 'cpu'.
        
    Returns:
        torch.Tensor: The attribution matrix matching the shape of flagged_samples.
    """
    if IntegratedGradients is None:
        raise ImportError("Captum is required for this module. Install with 'pip install captum'.")
        
    logging.info(f"Calculating attributions for {flagged_samples.shape[0]} flagged components...")
    
    # Wrap model to output reconstruction error
    wrapper = ReconstructionErrorWrapper(model).to(device)
    
    # Initialize Integrated Gradients
    ig = IntegratedGradients(wrapper)
    
    flagged_samples = flagged_samples.to(device).requires_grad_(True)
    background_data = background_data.to(device)
    
    # Attribute with respect to target=0 (the single loss value returned by the wrapper)
    attributions, delta = ig.attribute(
        inputs=flagged_samples,
        baselines=background_data,
        target=0,
        return_convergence_delta=True,
        n_steps=50 # Number of steps in the Riemann approximation of the integral
    )
    
    logging.info("Attribution calculation complete.")
    return attributions

def generate_root_cause_json(
    attributions: torch.Tensor, 
    component_ids: List[str], 
    feature_names: List[str]
) -> str:
    """
    Aggregates temporal attributions and formats them into a frontend-ready JSON structure.
    
    Args:
        attributions: Tensor of shape (batch, seq_len, num_features) containing gradient scores.
        component_ids: List of string identifiers for each component.
        feature_names: List of names corresponding to the 6 features.
        
    Returns:
        str: JSON string containing root-cause analysis for each component.
    """
    logging.info("Aggregating temporal attributions into Root-Cause JSON...")
    
    # Aggregate over the time dimension by taking the sum of absolute attributions
    # Shape becomes (batch, num_features)
    abs_attributions = torch.abs(attributions).detach().cpu().numpy()
    feature_importance = np.sum(abs_attributions, axis=1)
    
    results = []
    
    for i, component_id in enumerate(component_ids):
        # Calculate percentage contribution for each feature
        total_importance = np.sum(feature_importance[i])
        
        # Handle zero-division case safely
        if total_importance == 0:
            percentages = np.zeros_like(feature_importance[i])
        else:
            percentages = (feature_importance[i] / total_importance) * 100.0
            
        # Find the primary culprit
        primary_idx = int(np.argmax(percentages))
        primary_feature = feature_names[primary_idx]
        primary_score = percentages[primary_idx]
        
        # Build the feature breakdown dict
        breakdown = {feature_names[j]: round(float(percentages[j]), 2) for j in range(len(feature_names))}
        
        result_dict = {
            "die_id": component_id,
            "primary_failure": primary_feature,
            "contribution_percentage": round(float(primary_score), 2),
            "feature_breakdown": breakdown
        }
        results.append(result_dict)
        
    return json.dumps(results, indent=4)

if __name__ == '__main__':
    # We need the LSTM model structure to test this natively
    # Simulating a mock autoencoder just for the execution test
    class MockAutoencoder(nn.Module):
        def forward(self, x):
            # Simulated reconstruction: just adds some noise so error > 0
            # A real anomalous component would have a massive difference
            return x * 0.5 + 0.5 

    logging.info("Starting Tier 3 Explainability Test...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mock_model = MockAutoencoder().to(device)
    
    # Simulate a flagged component: shape (1, 169, 6)
    # The anomaly is injected heavily into feature index 1 (Leakage Current)
    flagged_tensor = torch.zeros((1, 169, 6))
    flagged_tensor[0, :, 1] = torch.linspace(0, 10, 169) # Runaway leakage
    flagged_tensor[0, :, 4] = torch.linspace(0, 2, 169)  # Minor leakage velocity
    
    # Baseline normal background (e.g. median of training data, here using zeros)
    baseline_tensor = torch.zeros((1, 169, 6))
    
    feature_names = [
        "I_DDQ_Magnitude", "Leakage_Magnitude", "Delay_Magnitude",
        "I_DDQ_Velocity", "Leakage_Velocity", "Delay_Velocity"
    ]
    component_ids = ["W2_D42"]
    
    try:
        attributions = calculate_feature_attribution(
            model=mock_model,
            background_data=baseline_tensor,
            flagged_samples=flagged_tensor,
            device=device
        )
        
        json_output = generate_root_cause_json(
            attributions=attributions,
            component_ids=component_ids,
            feature_names=feature_names
        )
        
        logging.info(f"Root-Cause Breakdown:\n{json_output}")
        
    except ImportError as e:
        logging.error(f"Test aborted: {e}")

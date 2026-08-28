import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
import sys
from typing import Tuple, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for temporal anomaly detection.
    Compresses input sequence to a latent vector, then reconstructs it.
    """
    def __init__(self, seq_len: int, n_features: int, hidden_dim: int = 64, latent_dim: int = 16):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder_lstm1 = nn.LSTM(input_size=n_features, hidden_size=hidden_dim, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(input_size=hidden_dim, hidden_size=latent_dim, batch_first=True)
        
        # Decoder
        self.decoder_lstm1 = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(input_size=hidden_dim, hidden_size=n_features, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, n_features)
        
        # Encode
        encoded, _ = self.encoder_lstm1(x)
        _, (hidden, _) = self.encoder_lstm2(encoded)
        
        # hidden is (1, batch_size, latent_dim) for single-direction LSTM
        latent_vector = hidden[-1] # (batch_size, latent_dim)
        
        # Decode: repeat latent vector for each time step
        decoded_input = latent_vector.unsqueeze(1).repeat(1, self.seq_len, 1) # (batch, seq_len, latent)
        decoded, _ = self.decoder_lstm1(decoded_input)
        reconstructed, _ = self.decoder_lstm2(decoded) # (batch, seq_len, n_features)
        
        return reconstructed

class ChronoDriftAnomalyEngine:
    """
    Manages the training and evaluation of the LSTM Autoencoder for ChronoDrift-AI.
    """
    def __init__(self, seq_len: int, n_features: int, hidden_dim: int = 64, latent_dim: int = 16, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Initializing Tier 2 Autoencoder on device: {self.device}")
        self.model = LSTMAutoencoder(seq_len, n_features, hidden_dim, latent_dim).to(self.device)
        self.criterion = nn.MSELoss()
        
    def train_model(self, train_data: np.ndarray, epochs: int = 50, batch_size: int = 32, learning_rate: float = 1e-3, patience: int = 5) -> None:
        """
        Trains the autoencoder on strictly normal (Tier 1 passed) data.
        Includes early stopping based on training loss.
        """
        logging.info(f"Starting training for up to {epochs} epochs with batch size {batch_size}...")
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        tensor_data = torch.tensor(train_data, dtype=torch.float32)
        dataset = TensorDataset(tensor_data, tensor_data) # Autoencoder maps input to itself
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        best_loss = float('inf')
        patience_counter = 0
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * batch_x.size(0)
                
            avg_loss = epoch_loss / len(dataset)
            logging.info(f"Epoch [{epoch+1:03d}/{epochs:03d}] - Loss (MSE): {avg_loss:.6f}")
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping triggered at epoch {epoch+1}.")
                    break

    def evaluate_anomalies(self, eval_data: np.ndarray, threshold_percentile: float = 95.0) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Evaluates the reconstruction error and flags anomalies based on dynamic thresholding.
        
        Args:
            eval_data: Data to evaluate (including potential defects).
            threshold_percentile: Percentile of reconstruction errors to use as the cut-off threshold.
            
        Returns:
            Tuple containing boolean mask (True = Latent Defect), reconstruction errors, and threshold.
        """
        logging.info("Evaluating anomalies via sequence reconstruction...")
        self.model.eval()
        
        tensor_data = torch.tensor(eval_data, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            reconstructed = self.model(tensor_data)
        
        # Calculate Mean Squared Error per sequence (averaging across time steps and features)
        errors = torch.mean((tensor_data - reconstructed) ** 2, dim=[1, 2]).cpu().numpy()
        
        # Dynamic Threshold calculation (based on empirical distribution)
        threshold = np.percentile(errors, threshold_percentile)
        logging.info(f"Dynamic Anomaly Threshold ({threshold_percentile}th percentile): {threshold:.6f}")
        
        # Mask: True if error > threshold (Latent Defect Flagged)
        anomaly_mask = errors > threshold
        
        return anomaly_mask, errors, threshold

if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    
    num_samples = 1000
    time_steps = 169
    num_features = 6
    
    logging.info("Generating dummy data for Tier 2 Autoencoder testing...")
    
    # Generate entirely normal baseline data (representing items that passed Tier 1 safely)
    train_data = np.random.normal(0, 1.0, (800, time_steps, num_features))
    
    # Generate evaluation data: mix of normals and some subtle anomalies
    eval_data_normal = np.random.normal(0, 1.0, (180, time_steps, num_features))
    
    # Inject 20 subtle anomalies: slight linear drifts applied over time.
    # Barely noticeable per step, but structurally distinct enough that 
    # the LSTM reconstruction will fail to map it accurately.
    eval_data_anomalous = np.random.normal(0, 1.0, (20, time_steps, num_features))
    drift = np.linspace(0, 0.5, time_steps).reshape(1, time_steps, 1) # Drifting up to 0.5 sigma over 168h
    eval_data_anomalous += drift
    
    # Combine evaluation data (Anomalies are stored at index 180 to 199)
    eval_data = np.concatenate([eval_data_normal, eval_data_anomalous], axis=0)
    
    # 1. Instantiate the Engine
    engine = ChronoDriftAnomalyEngine(seq_len=time_steps, n_features=num_features, hidden_dim=32, latent_dim=8)
    
    # 2. Train purely on 'passed' normal components
    engine.train_model(train_data, epochs=10, batch_size=64, learning_rate=0.005, patience=3)
    
    # 3. Evaluate and Flag
    # Setting threshold at 90th percentile to easily catch the top 10% (our 20 anomalies)
    mask, errors, threshold = engine.evaluate_anomalies(eval_data, threshold_percentile=90.0)
    
    flagged_count = np.sum(mask)
    logging.info(f"Evaluation complete. Flagged {flagged_count}/{len(eval_data)} components as Latent Defects.")
    logging.info(f"Expected anomalies: 20 (indices 180 to 199)")
    
    flagged_indices = np.where(mask)[0]
    logging.info(f"Sample of flagged indices: {flagged_indices[:20]}")

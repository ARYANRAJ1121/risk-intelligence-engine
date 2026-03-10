"""
lstm_sequence.py — LSTM (Long Short-Term Memory) for Credit Deterioration
========================================================================
WHY THIS EXISTS:
    Traditional tabular models (Layer 2) look at a single snapshot in time.
    LSTMs look at the SEQUENCE of a borrower's behavior over time to catch
    credit deterioration (the "velocity" of risk) much earlier.

WHAT THIS DOES:
    Accepts 3D tensors: [batch_size, sequence_length, num_features]
    Sequence length is usually N months of historical payment/balance data.
    The LSTM processes this temporal data to output a Probability of Default (PD).
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


logger = logging.getLogger("models.lstm_sequence")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ============================================================
#  1. PyTorch LSTM Architecture
# ============================================================

if PYTORCH_AVAILABLE:
    class CreditDeteriorationLSTM(nn.Module):
        """
        An LSTM network for predicting loan defaults based on a temporal sequence
        of financial snapshorts.
        """
        def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            
            # The core recurrent layer tracking state across time
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,    # Inputs are [batch, seq_len, features]
                dropout=dropout if num_layers > 1 else 0
            )
            
            # The final classification head parsing the final LSTM state
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),    # Binary Output (Default / No Default)
                nn.Sigmoid()         # Push prediction to [0, 1] probability range
            )

        def forward(self, x):
            """
            x shape: (batch_size, sequence_length, input_dim)
            """
            # out: all hidden states across all time steps
            # (h_n, c_n): final hidden and cell states for the sequence
            out, (h_n, c_n) = self.lstm(x)
            
            # We care about the *last* time step's output to make our final prediction
            # Depending on PyTorch version/settings, out[:, -1, :] holds the final relevant vector.
            # We'll use the top layer's final hidden state h_n[-1]
            last_hidden_state = h_n[-1] 
            
            # Pass the final temporal understanding through the classifying dense layers
            probability_of_default = self.classifier(last_hidden_state)
            return probability_of_default


# ============================================================
#  2. Data Preparation Helpers for Sequential Data
# ============================================================

def create_sequences(df: pd.DataFrame, target_col: str, seq_length: int = 3) -> tuple:
    """
    Convert a flat 2D tabular dataset into a 3D sequential dataset
    [samples, sequence_length, features] using a sliding window approach.

    In production with true time-series data (e.g. Freddie Mac monthly logs),
    you would group by Loan ID, sort by Date, and slide a window. Here we
    simulate temporal variation by treating consecutive rows as sequential
    snapshots and adding small Gaussian noise to represent month-over-month
    fluctuations in the borrower's financial profile.

    Args:
        df: DataFrame with numeric features (target excluded)
        target_col: Name of target column (used for label generation)
        seq_length: Number of time steps per sequence

    Returns:
        (X_3d, y_seq): X_3d shape [n_sequences, seq_length, n_features],
                        y_seq shape [n_sequences, 1]
    """
    features = [c for c in df.columns if c != target_col]
    X_values = df[features].values.astype(np.float32)

    n_samples = len(X_values)
    n_features = X_values.shape[1]

    # Number of non-overlapping sequences we can create
    n_sequences = n_samples // seq_length

    if n_sequences == 0:
        raise ValueError(f"Not enough rows ({n_samples}) for seq_length={seq_length}")

    # Trim to exact multiple of seq_length
    X_trimmed = X_values[:n_sequences * seq_length]

    # Reshape into [n_sequences, seq_length, n_features]
    X_3d = X_trimmed.reshape(n_sequences, seq_length, n_features)

    # Add small Gaussian noise to each time step to simulate temporal variation
    # Step 0 = original data, Step 1 = slight drift, Step 2 = more drift
    noise_scale = 0.02  # 2% standard deviation
    for t in range(1, seq_length):
        noise = np.random.normal(0, noise_scale, size=(n_sequences, n_features)).astype(np.float32)
        X_3d[:, t, :] = X_3d[:, t, :] + noise * t

    # Generate binary labels: use the mean of features in the last time step
    # as a proxy signal — in production this would be the actual default flag
    # from the next month's observation
    last_step_mean = np.mean(X_3d[:, -1, :], axis=1)
    threshold = np.percentile(last_step_mean, 90)  # Top 10% = "default"
    y_seq = (last_step_mean > threshold).astype(np.float32).reshape(-1, 1)

    logger.info(f"Created {n_sequences} sequences | Shape: {X_3d.shape} | Default rate: {y_seq.mean():.1%}")

    return X_3d, y_seq


# ============================================================
#  3. Wrapper for Training
# ============================================================

class LSTMDetector:
    """Wrapper class to handle PyTorch LSTM training."""
    
    def __init__(self, input_dim: int, epochs: int = 20, batch_size: int = 128, lr: float = 0.001):
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Run: pip install torch")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CreditDeteriorationLSTM(input_dim=input_dim).to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        
        # BCELoss is standard for Binary Classification
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
    def fit(self, X_train_3d: np.ndarray, y_train: np.ndarray):
        """Train the LSTM on 3D sequence arrays."""
        self.model.train()
        
        X_tensor = torch.FloatTensor(X_train_3d)
        y_tensor = torch.FloatTensor(y_train).view(-1, 1) # Ensure Shape [Batch, 1]
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        logger.info(f"Training LSTM on {self.device} for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # Forward
                pred_prob = self.model(batch_x)
                loss = self.criterion(pred_prob, batch_y)
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            if (epoch + 1) % 5 == 0 or epoch == 0:
                avg_loss = total_loss / len(loader)
                logger.info(f"  Epoch [{epoch+1}/{self.epochs}] Loss: {avg_loss:.6f}")
                
    def predict_proba(self, X_3d: np.ndarray) -> np.ndarray:
        """Output the final PD for the sequences."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X_3d).to(self.device)
        
        # Use simple batching to avoid memory blow-ups on large evaluations
        loader = DataLoader(TensorDataset(X_tensor), batch_size=self.batch_size, shuffle=False)
        all_preds = []
        
        with torch.no_grad():
            for batch_x in loader:
                batch_x = batch_x[0].to(self.device)
                pred_prob = self.model(batch_x)
                all_preds.append(pred_prob.cpu().numpy())
                
        return np.vstack(all_preds)

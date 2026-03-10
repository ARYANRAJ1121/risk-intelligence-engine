"""
dl_anomaly.py — Deep Learning Anomaly Detection (Autoencoder + Isolation Forest)
==============================================================================
WHY THIS EXISTS:
    Standard ML models (XGBoost/Random Forest) are great at expected patterns 
    (e.g. low income + high debt = default). They are BAD at unexpected patterns
    (e.g., fraud, synthetic identities, or impossible combinations of data).

WHAT THIS DOES:
    An Autoencoder compresses healthy loan data into a narrow "bottleneck" and
    reconstructs it. When fed a fraudulent/weird loan, it fails to reconstruct
    it accurately. High Reconstruction Error = High Anomaly Score.
    
    We combine this with Scikit-Learn's Isolation Forest to create an 
    Ensemble Anomaly Detector.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Scikit-learn for Isolation Forest
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# PyTorch for the Autoencoder
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


logger = logging.getLogger("models.dl_anomaly")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ============================================================
#  1. PyTorch Autoencoder Architecture
# ============================================================

if PYTORCH_AVAILABLE:
    class CreditAutoencoder(nn.Module):
        """
        A deep autoencoder for compressing and reconstructing tabular credit data.
        The "bottleneck" forces the network to learn the core mathematical
        components of a healthy loan.
        """
        def __init__(self, input_dim: int, hidden_dim1: int = 32, hidden_dim2: int = 16, bottleneck_dim: int = 8):
            super().__init__()
            
            # Encoder: Compress the data down
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim1),
                nn.ELU(),  # ELU handles negative values better than ReLU for tabular data
                nn.Dropout(0.2), # Prevent overfitting
                nn.Linear(hidden_dim1, hidden_dim2),
                nn.ELU(),
                nn.Linear(hidden_dim2, bottleneck_dim) # The Bottleneck
            )
            
            # Decoder: Try to reconstruct the original data
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck_dim, hidden_dim2),
                nn.ELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim2, hidden_dim1),
                nn.ELU(),
                nn.Linear(hidden_dim1, input_dim)
            )

        def forward(self, x):
            bottleneck = self.encoder(x)
            reconstructed = self.decoder(bottleneck)
            return reconstructed


# ============================================================
#  2. The Autoencoder Training & Scoring Logic
# ============================================================

class AutoencoderDetector:
    """Wrapper class to handle PyTorch training, scoring, and saving."""
    
    def __init__(self, input_dim: int, epochs: int = 50, batch_size: int = 256, lr: float = 1e-3):
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for AutoencoderDetector. Run: pip install torch")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CreditAutoencoder(input_dim=input_dim).to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        
    def fit(self, X_train_scaled: np.ndarray):
        """Train the autoencoder on (ideally) ONLY healthy non-default loans."""
        self.model.train()
        
        dataset = TensorDataset(torch.FloatTensor(X_train_scaled))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Learning rate scheduler — reduce LR when loss plateaus
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )
        
        logger.info(f"Training Autoencoder on {self.device} for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x in loader:
                batch_x = batch_x[0].to(self.device)
                
                # Forward
                reconstructed = self.model(batch_x)
                loss = self.criterion(reconstructed, batch_x)
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            scheduler.step(avg_loss)
                
            if (epoch + 1) % 10 == 0 or epoch == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"  Epoch [{epoch+1}/{self.epochs}] Loss: {avg_loss:.6f} | LR: {current_lr:.6f}")
                
    def score_anomalies(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction error for each row. 
        Higher error = Higher probability of anomaly.
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            # Calculate Mean Squared Error per row (mean across features)
            mse_per_row = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            
        return mse_per_row.cpu().numpy()


# ============================================================
#  3. Ensemble Anomaly Detector (Combining PyTorch + sklearn)
# ============================================================

class EnsembleAnomalyDetector:
    """
    Combines the Deep Learning Autoencoder with a Tree-based Isolation Forest.
    Isolation Forest catches "simple" extreme value outlies.
    Autoencoder catches "complex" relational outliers (e.g., high income + zero credit lines).
    """
    def __init__(self, contamination: float = 0.01):
        """
        Args:
            contamination: Expected percentage of anomalies in the dataset (e.g. 1%)
        """
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.iso_forest = IsolationForest(
            contamination=self.contamination, 
            random_state=42, 
            n_jobs=-1
        )
        self.autoencoder = None # Will be initialized during fit
        
    def fit(self, X_train: pd.DataFrame):
        """
        Fit the ensemble on the training data.
        NOTE: For best results, X_train should ideally only contain healthy (non-default) loans.
        """
        logger.info(f"Fitting Ensemble Anomaly Detector on {len(X_train)} rows.")
        
        # 1. Scale data
        X_scaled = self.scaler.fit_transform(X_train)
        
        # 2. Fit Isolation Forest
        logger.info("Training Isolation Forest...")
        self.iso_forest.fit(X_scaled)
        
        # 3. Fit Autoencoder
        if PYTORCH_AVAILABLE:
            self.autoencoder = AutoencoderDetector(input_dim=X_scaled.shape[1], epochs=30)
            self.autoencoder.fit(X_scaled)
        else:
            logger.warning("PyTorch not found. Ensemble will ONLY use Isolation Forest.")
            
    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Score a new dataset and return the anomaly scores.
        """
        X_scaled = self.scaler.transform(X_test)
        results = pd.DataFrame(index=X_test.index)
        
        # 1. Isolation Forest Scoring (-1 is anomaly, 1 is normal)
        # We invert the score so that Higher = More Anomalous
        iso_scores = self.iso_forest.score_samples(X_scaled)
        results["iso_forest_score"] = -iso_scores 
        results["is_iso_anomaly"] = self.iso_forest.predict(X_scaled) == -1
        
        # 2. Autoencoder Scoring (Reconstruction Error)
        if self.autoencoder:
            results["reconstruction_error"] = self.autoencoder.score_anomalies(X_scaled)
            
            # Define an anomaly as the top X% of reconstruction errors
            threshold = np.percentile(results["reconstruction_error"], 100 * (1 - self.contamination))
            results["is_ae_anomaly"] = results["reconstruction_error"] > threshold
            
            # Combine them: If BOTH models say it's an anomaly, it's highly suspicious!
            results["is_ensemble_anomaly"] = results["is_iso_anomaly"] | results["is_ae_anomaly"]
            results["strong_anomaly_flag"] = results["is_iso_anomaly"] & results["is_ae_anomaly"]
            
        else:
            results["is_ensemble_anomaly"] = results["is_iso_anomaly"]
            
        return results

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm

class MLP(nn.Module):
    """Multi-layer perceptron for binary classification"""
    
    def __init__(self, input_dim: int, hidden_layers: List[int], dropout: float = 0.3, use_dp: bool = False):
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            
            # Use GroupNorm for DP compatibility, BatchNorm otherwise
            if use_dp:
                # GroupNorm: divide channels into groups (use 1 group = LayerNorm behavior for 1D)
                num_groups = min(32, hidden_dim)  # Ensure divisibility
                while hidden_dim % num_groups != 0:
                    num_groups -= 1
                layers.append(nn.GroupNorm(num_groups, hidden_dim))
            else:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def get_confidence(self, x):
        """Get prediction confidence (probability)"""
        with torch.no_grad():
            return self.forward(x)
    
    def get_loss(self, x, y):
        """Get per-sample loss"""
        pred = self.forward(x)
        loss = nn.BCELoss(reduction='none')(pred, y.unsqueeze(1).float())
        return loss.squeeze()


class StandardTrainer:
    """Standard (non-private) model trainer"""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.01,
        batch_size: int = 256,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        self.training_history = {'loss': [], 'accuracy': []}
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        verbose: bool = True
    ) -> Dict:
        """Train the model"""
        
        # Convert to tensors
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor).squeeze()
                val_loss = self.criterion(val_outputs, y_val_tensor).item()
                val_preds = (val_outputs > 0.5).float()
                val_acc = (val_preds == y_val_tensor).float().mean().item()
            
            self.training_history['loss'].append(epoch_loss / len(train_loader))
            self.training_history['accuracy'].append(val_acc)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {epoch_loss/len(train_loader):.4f} - "
                      f"Val Acc: {val_acc:.4f}")
        
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor).squeeze()
            return (outputs > 0.5).cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor).squeeze()
            return outputs.cpu().numpy()
    
    def get_sample_losses(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Get per-sample losses for attack"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            losses = self.model.get_loss(X_tensor, y_tensor)
            return losses.cpu().numpy()


class DPSGDTrainer:
    """Differential Privacy SGD trainer using Opacus"""
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        learning_rate: float = 0.01,
        batch_size: int = 256,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.training_history = {'loss': [], 'accuracy': [], 'epsilon': []}
        
        try:
            from opacus import PrivacyEngine
            self.privacy_engine = PrivacyEngine()
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            self.use_opacus = True
        except ImportError:
            print("Warning: Opacus not available. Using manual DP-SGD implementation.")
            self.use_opacus = False
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            self.noise_multiplier = self._calculate_noise_multiplier()
        
        self.criterion = nn.BCELoss()
    
    def _calculate_noise_multiplier(self) -> float:
        """Calculate noise multiplier for target epsilon"""
        # Simplified calculation - use proper privacy accounting in production
        return np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def _clip_gradients(self):
        """Manually clip gradients"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
    
    def _add_noise(self):
        """Add Gaussian noise to gradients"""
        for p in self.model.parameters():
            if p.grad is not None:
                noise = torch.randn_like(p.grad) * self.noise_multiplier * self.max_grad_norm
                p.grad.data.add_(noise)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        verbose: bool = True
    ) -> Dict:
        """Train with differential privacy"""
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        if self.use_opacus:
            # Attach privacy engine
            self.model, self.optimizer, train_loader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=train_loader,
                noise_multiplier=self._calculate_noise_multiplier(),
                max_grad_norm=self.max_grad_norm,
            )
        
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                
                if not self.use_opacus:
                    self._clip_gradients()
                    self._add_noise()
                
                self.optimizer.step()
                epoch_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor).squeeze()
                val_preds = (val_outputs > 0.5).float()
                val_acc = (val_preds == y_val_tensor).float().mean().item()
            
            # Track privacy budget
            if self.use_opacus:
                epsilon_spent = self.privacy_engine.get_epsilon(self.delta)
            else:
                # Approximate epsilon accounting
                epsilon_spent = self.epsilon * (epoch + 1) / epochs
            
            self.training_history['loss'].append(epoch_loss / len(train_loader))
            self.training_history['accuracy'].append(val_acc)
            self.training_history['epsilon'].append(epsilon_spent)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {epoch_loss/len(train_loader):.4f} - "
                      f"Val Acc: {val_acc:.4f} - "
                      f"ε: {epsilon_spent:.4f}")
        
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor).squeeze()
            return (outputs > 0.5).cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor).squeeze()
            return outputs.cpu().numpy()
    
    def get_sample_losses(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Get per-sample losses"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            
            # Access underlying model if wrapped by Opacus
            if hasattr(self.model, '_module'):
                # Opacus wrapped model
                base_model = self.model._module
            else:
                base_model = self.model
            
            # Calculate loss directly since get_loss might not be accessible
            outputs = base_model(X_tensor).squeeze()
            loss = nn.BCELoss(reduction='none')(outputs, y_tensor)
            return loss.cpu().numpy()


def create_model(input_dim: int, hidden_layers: List[int], dropout: float = 0.3, use_dp: bool = False) -> MLP:
    """Factory function to create a new model"""
    return MLP(input_dim, hidden_layers, dropout, use_dp)

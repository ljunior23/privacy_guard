import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from models.neural_nets import MLP, StandardTrainer

class PATE:
    """
    Private Aggregation of Teacher Ensembles
    Based on Papernot et al. (ICLR 2017)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        num_teachers: int = 10,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.num_teachers = num_teachers
        self.epsilon = epsilon
        self.delta = delta
        self.device = device
        
        self.teachers = []
        self.student = None
        self.privacy_budget_spent = 0.0
        
    def _partition_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Partition data for teacher models"""
        
        n_samples = len(X)
        partition_size = n_samples // self.num_teachers
        
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        partitions = []
        for i in range(self.num_teachers):
            start_idx = i * partition_size
            end_idx = start_idx + partition_size if i < self.num_teachers - 1 else n_samples
            
            X_partition = X_shuffled[start_idx:end_idx]
            y_partition = y_shuffled[start_idx:end_idx]
            partitions.append((X_partition, y_partition))
        
        return partitions
    
    def train_teachers(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 30,
        learning_rate: float = 0.01,
        batch_size: int = 256,
        verbose: bool = True
    ):
        """Train ensemble of teacher models"""
        
        if verbose:
            print(f"Training {self.num_teachers} teacher models...")
        
        partitions = self._partition_data(X_train, y_train)
        
        for i, (X_part, y_part) in enumerate(partitions):
            # Create validation split for each teacher
            X_train_t, X_val_t, y_train_t, y_val_t = train_test_split(
                X_part, y_part, test_size=0.2, random_state=i
            )
            
            # Create and train teacher (use DP-compatible architecture)
            teacher = MLP(self.input_dim, self.hidden_layers, use_dp=False).to(self.device)
            trainer = StandardTrainer(
                teacher,
                learning_rate=learning_rate,
                batch_size=batch_size,
                device=self.device
            )
            
            trainer.train(
                X_train_t, y_train_t,
                X_val_t, y_val_t,
                epochs=epochs,
                verbose=False
            )
            
            self.teachers.append(teacher)
            
            if verbose:
                print(f"  Teacher {i+1}/{self.num_teachers} trained")
    
    def _noisy_aggregation(self, votes: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Aggregate teacher predictions with Laplacian noise
        
        Args:
            votes: Array of shape (n_samples, n_classes) with teacher votes
            
        Returns:
            predictions: Noisy aggregated predictions
            privacy_cost: Privacy budget spent
        """
        
        # Count votes for each class
        vote_counts = votes.sum(axis=1)
        
        # Add Laplacian noise for privacy
        # Sensitivity is 1 (changing one teacher's vote changes count by 1)
        noise_scale = 1.0 / self.epsilon
        noise = np.random.laplace(0, noise_scale, size=vote_counts.shape)
        noisy_counts = vote_counts + noise
        
        # Make predictions based on noisy counts
        predictions = (noisy_counts > self.num_teachers / 2).astype(int)
        
        # Calculate privacy cost (simplified - use moments accountant for accuracy)
        privacy_cost = self.epsilon * len(votes) / 10000  # Approximate
        
        return predictions, privacy_cost
    
    def generate_student_labels(
        self,
        X_public: np.ndarray,
        verbose: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        Generate labels for student training using noisy aggregation
        
        Args:
            X_public: Public data for student training
            
        Returns:
            labels: Private labels for student
            privacy_spent: Total privacy budget spent
        """
        
        if not self.teachers:
            raise ValueError("Teachers must be trained first!")
        
        if verbose:
            print("Generating private labels for student...")
        
        # Get predictions from all teachers
        teacher_preds = []
        for teacher in self.teachers:
            teacher.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_public).to(self.device)
                preds = (teacher(X_tensor).squeeze() > 0.5).cpu().numpy()
                teacher_preds.append(preds)
        
        teacher_preds = np.array(teacher_preds).T  # Shape: (n_samples, n_teachers)
        
        # Aggregate with noise
        labels, privacy_cost = self._noisy_aggregation(teacher_preds)
        self.privacy_budget_spent += privacy_cost
        
        if verbose:
            print(f"  Privacy budget spent: {privacy_cost:.6f}")
            print(f"  Total privacy spent: {self.privacy_budget_spent:.6f}")
        
        return labels, privacy_cost
    
    def train_student(
        self,
        X_public: np.ndarray,
        epochs: int = 50,
        learning_rate: float = 0.01,
        batch_size: int = 256,
        verbose: bool = True
    ):
        """Train student model on privately labeled public data"""
        
        if verbose:
            print("\nTraining student model...")
        
        # Generate private labels
        y_private, _ = self.generate_student_labels(X_public, verbose=False)
        
        # Split for validation
        X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(
            X_public, y_private, test_size=0.2, random_state=42
        )
        
        # Create and train student (use DP-compatible architecture)
        self.student = MLP(self.input_dim, self.hidden_layers, use_dp=False).to(self.device)
        trainer = StandardTrainer(
            self.student,
            learning_rate=learning_rate,
            batch_size=batch_size,
            device=self.device
        )
        
        history = trainer.train(
            X_train_s, y_train_s,
            X_val_s, y_val_s,
            epochs=epochs,
            verbose=verbose
        )
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using student model"""
        if self.student is None:
            raise ValueError("Student must be trained first!")
        
        self.student.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.student(X_tensor).squeeze()
            return (outputs > 0.5).cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using student model"""
        if self.student is None:
            raise ValueError("Student must be trained first!")
        
        self.student.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.student(X_tensor).squeeze()
            return outputs.cpu().numpy()
    
    def get_sample_losses(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Get per-sample losses from student"""
        if self.student is None:
            raise ValueError("Student must be trained first!")
        
        self.student.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            losses = self.student.get_loss(X_tensor, y_tensor)
            return losses.cpu().numpy()
    
    def get_privacy_spent(self) -> float:
        """Return total privacy budget spent"""
        return self.privacy_budget_spent


class PATETrainer:
    """Wrapper for PATE to match the interface of other trainers"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        num_teachers: int = 10,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.pate = PATE(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            num_teachers=num_teachers,
            epsilon=epsilon,
            delta=delta,
            device=device
        )
        self.training_history = {'loss': [], 'accuracy': []}
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        teacher_epochs: int = 30,
        verbose: bool = True
    ) -> Dict:
        """Train PATE model"""
        
        # Train teachers on private data
        self.pate.train_teachers(
            X_train, y_train,
            epochs=teacher_epochs,
            verbose=verbose
        )
        
        # Use validation set as "public" data for student
        # In practice, you would use actual public data
        history = self.pate.train_student(
            X_val,
            epochs=epochs,
            verbose=verbose
        )
        
        self.training_history = history
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pate.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pate.predict_proba(X)
    
    def get_sample_losses(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.pate.get_sample_losses(X, y)

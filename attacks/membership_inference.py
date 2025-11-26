
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Optional
import torch

class BaseAttack:
    """Base class for membership inference attacks"""
    
    def __init__(self, name: str):
        self.name = name
        self.attack_model = None
        self.metrics = {}
    
    def train(self, member_features, nonmember_features):
        """Train attack model"""
        raise NotImplementedError
    
    def predict(self, features) -> np.ndarray:
        """Predict membership"""
        raise NotImplementedError
    
    def evaluate(self, member_preds, nonmember_preds) -> Dict:
        """
        Evaluate attack performance
        
        Args:
            member_preds: Already computed predictions for members
            nonmember_preds: Already computed predictions for non-members
        """
        # Ensure predictions are 1D arrays
        if len(member_preds.shape) > 1:
            member_preds = member_preds.flatten()
        if len(nonmember_preds.shape) > 1:
            nonmember_preds = nonmember_preds.flatten()
        
        # Combine predictions and labels
        y_true = np.concatenate([
            np.ones(len(member_preds)),
            np.zeros(len(nonmember_preds))
        ])
        y_pred = np.concatenate([member_preds, nonmember_preds])
        
        # Calculate metrics
        auc = roc_auc_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred > 0.5)
        
        # Calculate precision and recall
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        
        self.metrics = {
            'auc': auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'member_predictions': member_preds,
            'nonmember_predictions': nonmember_preds
        }
        
        return self.metrics


class ConfidenceBasedAttack(BaseAttack):
    """
    Attack 1: Confidence-Based (Shokri et al., 2017)
    Uses prediction confidence scores to infer membership
    """
    
    def __init__(self):
        super().__init__("Confidence-Based (Shokri et al.)")
        self.attack_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    
    def _extract_features(self, confidences: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Extract features from model predictions
        
        Args:
            confidences: Prediction probabilities
            labels: True labels
            
        Returns:
            Feature matrix for attack model
        """
        features = []
        
        for conf, label in zip(confidences, labels):
            # Feature 1: Confidence for predicted class
            pred_conf = conf if label == 1 else 1 - conf
            
            # Feature 2: Confidence for true class
            true_conf = conf if label == 1 else 1 - conf
            
            # Feature 3: Entropy (uncertainty)
            epsilon = 1e-7
            entropy = -(conf * np.log(conf + epsilon) + 
                       (1-conf) * np.log(1-conf + epsilon))
            
            # Feature 4: Modified entropy
            modified_entropy = -np.log(pred_conf + epsilon)
            
            features.append([
                pred_conf,
                true_conf,
                entropy,
                modified_entropy
            ])
        
        return np.array(features)
    
    def train(
        self,
        member_confidences: np.ndarray,
        member_labels: np.ndarray,
        nonmember_confidences: np.ndarray,
        nonmember_labels: np.ndarray
    ):
        """Train attack model using shadow model approach"""
        
        # Extract features
        member_features = self._extract_features(member_confidences, member_labels)
        nonmember_features = self._extract_features(nonmember_confidences, nonmember_labels)
        
        # Create training data
        X_attack = np.vstack([member_features, nonmember_features])
        y_attack = np.concatenate([
            np.ones(len(member_features)),
            np.zeros(len(nonmember_features))
        ])
        
        # Train attack model
        self.attack_model.fit(X_attack, y_attack)
    
    def predict(self, confidences: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Predict membership probability"""
        features = self._extract_features(confidences, labels)
        return self.attack_model.predict_proba(features)[:, 1]


class LabelOnlyAttack(BaseAttack):
    """
    Attack 2: Label-Only (Choquette-Choo et al., 2021)
    Uses only predicted labels without confidence scores
    More realistic threat model
    """
    
    def __init__(self):
        super().__init__("Label-Only (Choquette-Choo et al.)")
        self.class_statistics = {}
    
    def train(
        self,
        member_predictions: np.ndarray,
        member_labels: np.ndarray,
        nonmember_predictions: np.ndarray,
        nonmember_labels: np.ndarray
    ):
        """
        Learn label statistics from shadow models
        """
        
        # Calculate agreement rates for members and non-members
        member_agreement = (member_predictions == member_labels).astype(float)
        nonmember_agreement = (nonmember_predictions == nonmember_labels).astype(float)
        
        # Per-class statistics
        for class_label in [0, 1]:
            member_mask = member_labels == class_label
            nonmember_mask = nonmember_labels == class_label
            
            self.class_statistics[class_label] = {
                'member_agreement': np.mean(member_agreement[member_mask]),
                'nonmember_agreement': np.mean(nonmember_agreement[nonmember_mask]),
                'member_count': np.sum(member_mask),
                'nonmember_count': np.sum(nonmember_mask)
            }
    
    def predict(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Predict membership based on label agreement
        """
        membership_scores = np.zeros(len(predictions))
        
        for i, (pred, label) in enumerate(zip(predictions, labels)):
            agrees = (pred == label)
            
            if label in self.class_statistics:
                stats = self.class_statistics[label]
                
                # Calculate likelihood ratio
                if agrees:
                    member_prob = stats['member_agreement']
                    nonmember_prob = stats['nonmember_agreement']
                else:
                    member_prob = 1 - stats['member_agreement']
                    nonmember_prob = 1 - stats['nonmember_agreement']
                
                # Avoid division by zero
                if nonmember_prob > 0:
                    likelihood_ratio = member_prob / nonmember_prob
                    membership_scores[i] = likelihood_ratio / (likelihood_ratio + 1)
                else:
                    membership_scores[i] = 0.5
            else:
                membership_scores[i] = 0.5
        
        return membership_scores


class MetricBasedAttack(BaseAttack):
    """
    Attack 3: Metric-Based (Song & Shmatikov)
    Uses per-sample loss values as attack signal
    """
    
    def __init__(self):
        super().__init__("Metric-Based (Song & Shmatikov)")
        self.threshold = None
        self.attack_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
    
    def _extract_features(self, losses: np.ndarray) -> np.ndarray:
        """Extract features from loss values"""
        features = []
        
        for loss in losses:
            # Feature 1: Raw loss
            # Feature 2: Log loss
            # Feature 3: Normalized loss (if statistics available)
            features.append([
                loss,
                np.log(loss + 1e-7),
                loss  # Placeholder for normalized loss
            ])
        
        return np.array(features)
    
    def train(
        self,
        member_losses: np.ndarray,
        nonmember_losses: np.ndarray
    ):
        """Train attack using loss distributions"""
        
        # Extract features
        member_features = self._extract_features(member_losses)
        nonmember_features = self._extract_features(nonmember_losses)
        
        # Create training data
        X_attack = np.vstack([member_features, nonmember_features])
        y_attack = np.concatenate([
            np.ones(len(member_features)),
            np.zeros(len(nonmember_features))
        ])
        
        # Train attack model
        self.attack_model.fit(X_attack, y_attack)
        
        # Also calculate simple threshold
        self.threshold = (np.mean(member_losses) + np.mean(nonmember_losses)) / 2
    
    def predict(self, losses: np.ndarray) -> np.ndarray:
        """Predict membership from losses"""
        features = self._extract_features(losses)
        return self.attack_model.predict_proba(features)[:, 1]


class AdaptiveThresholdAttack(BaseAttack):
    """
    Attack 4: Adaptive Threshold
    Per-class threshold optimization for different privacy budgets
    """
    
    def __init__(self):
        super().__init__("Adaptive Threshold")
        self.class_thresholds = {}
        self.global_threshold = None
    
    def train(
        self,
        member_confidences: np.ndarray,
        member_labels: np.ndarray,
        nonmember_confidences: np.ndarray,
        nonmember_labels: np.ndarray
    ):
        """Learn optimal thresholds per class"""
        
        # Global threshold
        all_member_conf = member_confidences
        all_nonmember_conf = nonmember_confidences
        
        # Find threshold that maximizes accuracy
        thresholds = np.linspace(0.3, 0.9, 100)
        best_acc = 0
        best_threshold = 0.5
        
        for thresh in thresholds:
            member_correct = np.sum(member_confidences > thresh)
            nonmember_correct = np.sum(nonmember_confidences <= thresh)
            acc = (member_correct + nonmember_correct) / (len(member_confidences) + len(nonmember_confidences))
            
            if acc > best_acc:
                best_acc = acc
                best_threshold = thresh
        
        self.global_threshold = best_threshold
        
        # Per-class thresholds
        for class_label in [0, 1]:
            member_mask = member_labels == class_label
            nonmember_mask = nonmember_labels == class_label
            
            if np.sum(member_mask) > 0 and np.sum(nonmember_mask) > 0:
                member_conf_class = member_confidences[member_mask]
                nonmember_conf_class = nonmember_confidences[nonmember_mask]
                
                # Find optimal threshold for this class
                best_class_acc = 0
                best_class_threshold = 0.5
                
                for thresh in thresholds:
                    member_correct = np.sum(member_conf_class > thresh)
                    nonmember_correct = np.sum(nonmember_conf_class <= thresh)
                    acc = (member_correct + nonmember_correct) / (len(member_conf_class) + len(nonmember_conf_class))
                    
                    if acc > best_class_acc:
                        best_class_acc = acc
                        best_class_threshold = thresh
                
                self.class_thresholds[class_label] = best_class_threshold
            else:
                self.class_thresholds[class_label] = self.global_threshold
    
    def predict(self, confidences: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Predict membership using adaptive thresholds"""
        membership_scores = np.zeros(len(confidences))
        
        for i, (conf, label) in enumerate(zip(confidences, labels)):
            # Use per-class threshold if available
            if label in self.class_thresholds:
                threshold = self.class_thresholds[label]
            else:
                threshold = self.global_threshold
            
            # Soft decision: distance from threshold
            distance = conf - threshold
            # Convert to probability using sigmoid
            membership_scores[i] = 1 / (1 + np.exp(-5 * distance))
        
        return membership_scores


class AttackOrchestrator:
    """Manages and runs all attacks"""
    
    def __init__(self):
        self.attacks = {
            'confidence_based': ConfidenceBasedAttack(),
            'label_only': LabelOnlyAttack(),
            'metric_based': MetricBasedAttack(),
            'adaptive_threshold': AdaptiveThresholdAttack()
        }
        self.results = {}
    
    def train_attacks(
        self,
        target_model,
        shadow_models,
        X_shadow: np.ndarray,
        y_shadow: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ):
        """
        Train all attacks using shadow models
        
        Args:
            target_model: Model to attack
            shadow_models: List of shadow models trained on similar data
            X_shadow: Shadow training data
            y_shadow: Shadow training labels
            X_test: Test data (non-members)
            y_test: Test labels
        """
        
        print("Training attacks using shadow models...")
        
        # Collect data from shadow models
        member_confidences = []
        member_predictions = []
        member_losses = []
        member_labels = []
        
        nonmember_confidences = []
        nonmember_predictions = []
        nonmember_losses = []
        nonmember_labels = []
        
        for i, shadow in enumerate(shadow_models):
            # Member data (training set of shadow model)
            n_train = len(X_shadow) // 2
            X_member = X_shadow[:n_train]
            y_member = y_shadow[:n_train]
            
            # Non-member data (test set)
            X_nonmember = X_test[:len(X_member)]
            y_nonmember = y_test[:len(X_member)]
            
            # Get predictions
            member_conf = shadow.predict_proba(X_member)
            nonmember_conf = shadow.predict_proba(X_nonmember)
            
            member_pred = shadow.predict(X_member)
            nonmember_pred = shadow.predict(X_nonmember)
            
            member_loss = shadow.get_sample_losses(X_member, y_member)
            nonmember_loss = shadow.get_sample_losses(X_nonmember, y_nonmember)
            
            member_confidences.append(member_conf)
            member_predictions.append(member_pred)
            member_losses.append(member_loss)
            member_labels.append(y_member)
            
            nonmember_confidences.append(nonmember_conf)
            nonmember_predictions.append(nonmember_pred)
            nonmember_losses.append(nonmember_loss)
            nonmember_labels.append(y_nonmember)
        
        # Concatenate all data
        member_confidences = np.concatenate(member_confidences)
        member_predictions = np.concatenate(member_predictions)
        member_losses = np.concatenate(member_losses)
        member_labels = np.concatenate(member_labels)
        
        nonmember_confidences = np.concatenate(nonmember_confidences)
        nonmember_predictions = np.concatenate(nonmember_predictions)
        nonmember_losses = np.concatenate(nonmember_losses)
        nonmember_labels = np.concatenate(nonmember_labels)
        
        # Train each attack
        print("\n1. Training Confidence-Based Attack...")
        self.attacks['confidence_based'].train(
            member_confidences, member_labels,
            nonmember_confidences, nonmember_labels
        )
        
        print("2. Training Label-Only Attack...")
        self.attacks['label_only'].train(
            member_predictions, member_labels,
            nonmember_predictions, nonmember_labels
        )
        
        print("3. Training Metric-Based Attack...")
        self.attacks['metric_based'].train(
            member_losses, nonmember_losses
        )
        
        print("4. Training Adaptive Threshold Attack...")
        self.attacks['adaptive_threshold'].train(
            member_confidences, member_labels,
            nonmember_confidences, nonmember_labels
        )
        
        print("\nAll attacks trained successfully!")
    
    def evaluate_attacks(
        self,
        target_model,
        X_member: np.ndarray,
        y_member: np.ndarray,
        X_nonmember: np.ndarray,
        y_nonmember: np.ndarray
    ) -> Dict:
        """Evaluate all attacks against target model"""
        
        print("\nEvaluating attacks on target model...")
        
        # Get target model predictions
        member_conf = target_model.predict_proba(X_member)
        member_pred = target_model.predict(X_member)
        member_loss = target_model.get_sample_losses(X_member, y_member)
        
        nonmember_conf = target_model.predict_proba(X_nonmember)
        nonmember_pred = target_model.predict(X_nonmember)
        nonmember_loss = target_model.get_sample_losses(X_nonmember, y_nonmember)
        
        results = {}
        
        # Confidence-based
        conf_member = self.attacks['confidence_based'].predict(member_conf, y_member)
        conf_nonmember = self.attacks['confidence_based'].predict(nonmember_conf, y_nonmember)
        results['confidence_based'] = self.attacks['confidence_based'].evaluate(
            conf_member, conf_nonmember
        )
        
        # Label-only
        label_member = self.attacks['label_only'].predict(member_pred, y_member)
        label_nonmember = self.attacks['label_only'].predict(nonmember_pred, y_nonmember)
        results['label_only'] = self.attacks['label_only'].evaluate(
            label_member, label_nonmember
        )
        
        # Metric-based
        metric_member = self.attacks['metric_based'].predict(member_loss)
        metric_nonmember = self.attacks['metric_based'].predict(nonmember_loss)
        results['metric_based'] = self.attacks['metric_based'].evaluate(
            metric_member, metric_nonmember
        )
        
        # Adaptive threshold
        adapt_member = self.attacks['adaptive_threshold'].predict(member_conf, y_member)
        adapt_nonmember = self.attacks['adaptive_threshold'].predict(nonmember_conf, y_nonmember)
        results['adaptive_threshold'] = self.attacks['adaptive_threshold'].evaluate(
            adapt_member, adapt_nonmember
        )
        
        self.results = results
        return results
    
    def print_results(self):
        """Print attack results summary"""
        print("\n" + "="*70)
        print("ATTACK RESULTS SUMMARY")
        print("="*70)
        
        for attack_name, metrics in self.results.items():
            print(f"\n{attack_name.replace('_', ' ').title()}:")
            print(f"  AUC: {metrics['auc']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")

from dataclasses import dataclass
from typing import List, Dict
import numpy as np

@dataclass
class ExperimentConfig:
    """Configuration for privacy experiments"""
    
    # Privacy budgets to evaluate
    epsilon_values: List[float] = None
    
    # Model parameters
    batch_size: int = 256
    epochs: int = 50
    learning_rate: float = 0.01
    hidden_layers: List[int] = None
    
    # DP-SGD parameters
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.1
    delta: float = 1e-5
    
    # PATE parameters
    num_teachers: int = 10
    teacher_epochs: int = 30
    
    # Attack parameters
    shadow_models: int = 10
    attack_train_size: int = 5000
    
    # Data splits
    train_ratio: float = 0.6
    test_ratio: float = 0.2
    member_ratio: float = 0.2
    
    # Fairness groups
    sensitive_attributes: List[str] = None
    
    # Random seed
    random_seed: int = 42
    
    def __post_init__(self):
        if self.epsilon_values is None:
            self.epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        if self.hidden_layers is None:
            self.hidden_layers = [128, 64, 32]
        if self.sensitive_attributes is None:
            self.sensitive_attributes = ['race', 'sex', 'age']
    
    def get_noise_multiplier(self, epsilon: float) -> float:
        """Calculate noise multiplier for given epsilon"""
        # Simplified calculation - in practice use privacy accounting
        return np.sqrt(2 * np.log(1.25 / self.delta)) / epsilon


# Model architectures
MODEL_CONFIGS = {
    'small': [64, 32],
    'medium': [128, 64, 32],
    'large': [256, 128, 64, 32]
}

# Attack configurations
ATTACK_CONFIGS = {
    'confidence_based': {
        'name': 'Confidence-Based (Shokri et al.)',
        'uses_confidence': True,
        'threshold_optimization': False
    },
    'label_only': {
        'name': 'Label-Only (Choquette-Choo et al.)',
        'uses_confidence': False,
        'threshold_optimization': False
    },
    'metric_based': {
        'name': 'Metric-Based (Song & Shmatikov)',
        'uses_loss': True,
        'threshold_optimization': False
    },
    'adaptive_threshold': {
        'name': 'Adaptive Threshold',
        'uses_confidence': True,
        'threshold_optimization': True
    }
}

# Defense configurations
DEFENSE_CONFIGS = {
    'baseline': {'type': 'none', 'epsilon': None},
    'dpsgd_gaussian_5': {'type': 'dpsgd', 'epsilon': 5.0, 'mechanism': 'gaussian'},
    'dpsgd_gaussian_1': {'type': 'dpsgd', 'epsilon': 1.0, 'mechanism': 'gaussian'},
    'dpsgd_gaussian_05': {'type': 'dpsgd', 'epsilon': 0.5, 'mechanism': 'gaussian'},
    'dpsgd_laplace_1': {'type': 'dpsgd', 'epsilon': 1.0, 'mechanism': 'laplace'},
    'pate': {'type': 'pate', 'epsilon': 1.0}
}

# Fairness metrics to compute
FAIRNESS_METRICS = [
    'demographic_parity_difference',
    'equalized_odds_difference',
    'equal_opportunity_difference',
    'disparate_impact'
]

# Visualization settings
VIZ_CONFIG = {
    'style': 'seaborn-v0_8-darkgrid',
    'color_palette': 'Set2',
    'figure_dpi': 150,
    'save_format': 'png'
}

# Paths
DATA_DIR = './data'
MODELS_DIR = './models'
RESULTS_DIR = './results'
PLOTS_DIR = './results/plots'

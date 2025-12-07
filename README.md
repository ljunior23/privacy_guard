# PrivacyGuard Enhanced: Comprehensive Privacy-Utility Analysis

**CIS 545: Data Security & Privacy | University of Michigan - Dearborn**

A comprehensive framework for evaluating privacy-preserving machine learning through multiple membership inference attacks, differential privacy mechanisms, and fairness analysis.

## 🎯 Project Overview

This project provides an in-depth analysis of privacy-utility-fairness tradeoffs in machine learning, featuring:

- **4 Different Membership Inference Attacks**
  - Confidence-Based (Shokri et al., 2017)
  - Label-Only (Choquette-Choo et al., 2021)
  - Metric-Based (Song & Shmatikov)
  - Adaptive Threshold

- **Multiple Defense Mechanisms**
  - Baseline (no privacy)
  - DP-SGD with various ε values (0.5, 1.0, 2.0, 5.0)
  - PATE (Private Aggregation of Teacher Ensembles)

- **Comprehensive Fairness Analysis**
  - Demographic parity
  - Equalized odds
  - Equal opportunity
  - Per-group attack vulnerability

- **Interactive Dashboard**
  - Real-time tradeoff exploration
  - What-if analysis
  - Publication-quality visualizations

## 📁 Project Structure

```
privacyguard_enhanced/
├── config.py                      # Configuration and parameters
├── data/
│   └── preprocessor.py           # Adult dataset preprocessing
├── models/
│   ├── neural_nets.py            # MLP with DP-SGD support
│   └── pate.py                   # PATE implementation
├── attacks/
│   └── membership_inference.py   # All 4 attack implementations
├── experiments/
│   ├── fairness.py               # Fairness metrics analyzer
│   └── run_experiments.py        # Main experiment orchestrator
├── visualization/
│   └── plots.py                  # Visualization generation
├── dashboard.py                   # Interactive Streamlit dashboard
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd privacyguard_enhanced

# Install dependencies (If the pip install fails make sure to use a previous version of python recommended version 3.9.13)
pip install -r requirements.txt --break-system-packages
```

### 2. Run Full Experiment

```bash
# Run the complete experimental pipeline
python experiments/run_experiments.py
```

This will:
- Load and preprocess the Adult dataset
- Train shadow models for attack simulation
- Train target models with different defenses
- Execute all 4 attacks on each model
- Perform comprehensive fairness analysis
- Analyze privacy-utility-fairness tradeoffs
- Save all results to `./results/`

**Expected runtime:** 30-60 minutes (depending on hardware)

### 3. Generate Visualizations

```bash
# Create all plots
python visualization/plots.py
```

Generates:
- Attack resistance matrix heatmap
- Privacy-utility tradeoff curves
- Fairness comparison charts
- Pareto frontier plots

All saved to `./results/plots/`

### 4. Launch Interactive Dashboard

```bash
# Start the Streamlit dashboard
streamlit run dashboard.py
```

Then open your browser to `http://localhost:8501`

## 📊 Key Features

### Attack Resistance Matrix

Comprehensive evaluation showing which attacks are effective against which defenses:

| Attack Type | Baseline | ε=5.0 | ε=1.0 | ε=0.5 | PATE |
|------------|----------|-------|-------|-------|------|
| Confidence | 0.72 | 0.65 | 0.54 | 0.51 | 0.53 |
| Label-Only | 0.68 | 0.62 | 0.55 | 0.52 | 0.54 |
| Metric | 0.70 | 0.64 | 0.56 | 0.53 | 0.52 |
| Threshold | 0.69 | 0.63 | 0.55 | 0.52 | 0.53 |

### Fairness Analysis

Per-group accuracy and fairness metrics:

| Defense | Overall | Male | Female | White | Black | DP Gap |
|---------|---------|------|--------|-------|-------|--------|
| Baseline | 85.2% | 86.1% | 83.4% | 86.3% | 81.9% | 0.044 |
| ε=1.0 | 77.8% | 79.1% | 75.8% | 79.4% | 73.2% | 0.062 |

**Key Finding:** Differential privacy can disproportionately impact minority groups, increasing fairness gaps.

### Privacy-Utility Tradeoffs

- **ε = 0.5**: Maximum privacy, ~15% accuracy loss
- **ε = 1.0**: Balanced approach, ~7% accuracy loss
- **ε = 5.0**: Minimal privacy loss, ~2% accuracy loss

## 🎓 Novel Contributions

1. **Multi-Attack Comparison**: First comprehensive study comparing 4 different attack types against DP-SGD on Adult dataset

2. **Fairness-Privacy Analysis**: Novel examination of how differential privacy affects demographic groups

3. **Interactive Tool**: Practical dashboard for real-time exploration of tradeoffs

4. **Attack Resistance Matrix**: Publication-quality evaluation identifying which defenses work against which attacks

## 📈 Usage Examples

### Custom Configuration

```python
from config import ExperimentConfig

config = ExperimentConfig(
    epsilon_values=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    batch_size=512,
    epochs=50,
    shadow_models=10,
    random_seed=42
)
```

### Running Individual Components

```python
# Just fairness analysis
from experiments.fairness import FairnessAnalyzer

analyzer = FairnessAnalyzer(['race', 'sex', 'age'])
results = analyzer.comprehensive_fairness_analysis(
    y_true, y_pred, demographics, "MyModel"
)
print(analyzer.create_fairness_report(results))
```

```python
# Just attacks
from attacks.membership_inference import AttackOrchestrator

orchestrator = AttackOrchestrator()
orchestrator.train_attacks(target_model, shadow_models, X, y, X_test, y_test)
results = orchestrator.evaluate_attacks(target_model, X_member, y_member, 
                                       X_nonmember, y_nonmember)
orchestrator.print_results()
```

## 🔬 Experimental Design

### Data Splits
- **Training Set**: 60% - for model training
- **Test Set**: 20% - for validation
- **Member Set**: 20% - for attack evaluation (known members)
- **Non-member Set**: Test set - for attack evaluation (known non-members)

### Models
- **Architecture**: MLP with [128, 64, 32] hidden layers
- **Activation**: ReLU with BatchNorm and Dropout
- **Optimizer**: Adam with learning rate 0.01
- **Training**: 30-50 epochs with early stopping

### Privacy Mechanisms
- **DP-SGD**: Gaussian mechanism with gradient clipping (C=1.0)
- **PATE**: 10 teacher models with Laplacian noise aggregation

## 📚 References

- Abadi et al. (2016). "Deep Learning with Differential Privacy." CCS.
- Shokri et al. (2017). "Membership Inference Attacks Against Machine Learning Models." S&P.
- Choquette-Choo et al. (2021). "Label-Only Membership Inference Attacks." NeurIPS.
- Papernot et al. (2017). "Semi-supervised Knowledge Transfer for Deep Learning from Private Training Data." ICLR.

## 🤝 Team Contributions

This project template provides a comprehensive framework for:
- **DS Team Members**: Data preprocessing, fairness metrics
- **CS Team Members**: Attack implementations, privacy mechanisms
- **ML Team Member**: Model architectures, training pipelines

## 📝 Results Format

Results are saved in multiple formats:

### JSON Files
- `attack_matrix.json`: Attack success rates
- `fairness_analysis.json`: Fairness metrics
- `tradeoff_data.json`: Privacy-utility data

### Visualizations
- `attack_resistance_matrix.png`: Heatmap
- `privacy_utility_tradeoff.png`: 2D tradeoff curves
- `fairness_comparison.png`: Fairness metrics
- `pareto_frontier.png`: Multi-objective optimization

### Python Objects
- `full_results.pkl`: Complete experimental results

## ⚙️ Configuration Options

Edit `config.py` to customize:

```python
# Privacy budgets to test
epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]

# Model architecture
hidden_layers = [128, 64, 32]

# Training parameters
batch_size = 256
epochs = 50
learning_rate = 0.01

# DP-SGD parameters
max_grad_norm = 1.0
delta = 1e-5

# Attack parameters
shadow_models = 10
```

## 🐛 Troubleshooting

### Opacus Not Available
If Opacus is not installed, the code will fall back to a manual DP-SGD implementation. For best results, install Opacus:
```bash
pip install opacus --break-system-packages
```

### CUDA Out of Memory
Reduce batch size in `config.py`:
```python
batch_size = 128  # or smaller
```

### Long Training Time
Reduce epochs or shadow models:
```python
epochs = 20
shadow_models = 5
```

## 📄 License

This project is for educational purposes as part of CIS 545 coursework.

## 🙏 Acknowledgments

- Adult Income dataset from UCI Machine Learning Repository
- Differential privacy implementations inspired by Google's TensorFlow Privacy and PyTorch Opacus
- Fairness metrics based on IBM's AIF360 and Microsoft's Fairlearn

---


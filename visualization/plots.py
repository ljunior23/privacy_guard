import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (10, 6)

class PrivacyVisualizer:
    """Generate visualizations for privacy analysis"""
    
    def __init__(self, results_dir: str = './results'):
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        
    def plot_attack_resistance_matrix(self, attack_matrix: Dict, save: bool = True):
        """
        Create heatmap showing attack success rates across defenses
        """
        # Check if data is available
        if not attack_matrix or not any(attack_matrix.values()):
            print("Skipping attack resistance matrix - no data available")
            return None
        
        # Filter out empty defenses
        attack_matrix = {k: v for k, v in attack_matrix.items() if v}
        
        if not attack_matrix:
            print("Skipping attack resistance matrix - all defenses empty")
            return None
        # Prepare data
        defenses = list(attack_matrix.keys())
        attacks = list(attack_matrix[defenses[0]].keys())
        
        # Create matrix
        matrix_data = np.zeros((len(attacks), len(defenses)))
        for j, defense in enumerate(defenses):
            for i, attack in enumerate(attacks):
                matrix_data[i, j] = attack_matrix[defense][attack]['auc']
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Format attack names
        attack_labels = [name.replace('_', ' ').title() for name in attacks]
        defense_labels = [name.replace('_', ' ').title() for name in defenses]
        
        # Create heatmap
        im = ax.imshow(matrix_data, cmap='RdYlGn_r', aspect='auto', vmin=0.5, vmax=0.75)
        
        # Set ticks
        ax.set_xticks(np.arange(len(defenses)))
        ax.set_yticks(np.arange(len(attacks)))
        ax.set_xticklabels(defense_labels, rotation=45, ha='right')
        ax.set_yticklabels(attack_labels)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attack AUC', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(attacks)):
            for j in range(len(defenses)):
                text = ax.text(j, i, f'{matrix_data[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=10)
        
        ax.set_title('Membership Inference Attack Success Rates\n(AUC Scores)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Defense Mechanism', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attack Type', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.plots_dir / 'attack_resistance_matrix.png',
                       dpi=300, bbox_inches='tight')
            print(f"✓ Saved: attack_resistance_matrix.png")
        
        return fig
    
    def plot_privacy_utility_tradeoff(self, tradeoff_data: Dict, save: bool = True):
        """
        Plot privacy-utility tradeoff curves
        """
        # Check if data is complete
        if not tradeoff_data or not all(k in tradeoff_data for k in ['epsilon', 'accuracy', 'avg_attack_auc']):
            print("Skipping privacy-utility tradeoff - incomplete data")
            return None
        
        if not tradeoff_data['epsilon']:
            print("Skipping privacy-utility tradeoff - no data points")
            return None
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        epsilon = tradeoff_data['epsilon']
        accuracy = tradeoff_data['accuracy']
        attack_auc = tradeoff_data['avg_attack_auc']
        
        # Left plot: Accuracy vs Privacy
        ax1.plot(epsilon, accuracy, 'o-', linewidth=2, markersize=8,
                color='#2E86AB', label='Model Accuracy')
        ax1.axhline(y=accuracy[0] if len(accuracy) > 0 else 0.8,
                   color='gray', linestyle='--', alpha=0.5, label='Baseline')
        ax1.set_xlabel('Privacy Budget (ε)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Privacy vs. Utility Tradeoff', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.set_xscale('log')
        
        # Right plot: Attack Success vs Privacy
        ax2.plot(epsilon, attack_auc, 's-', linewidth=2, markersize=8,
                color='#A23B72', label='Avg Attack AUC')
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5,
                   label='Random Guess')
        ax2.set_xlabel('Privacy Budget (ε)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Attack Success (AUC)', fontsize=12, fontweight='bold')
        ax2.set_title('Privacy Protection Effectiveness', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        ax2.set_xscale('log')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.plots_dir / 'privacy_utility_tradeoff.png',
                       dpi=300, bbox_inches='tight')
            print(f"✓ Saved: privacy_utility_tradeoff.png")
        
        return fig
    
    def plot_fairness_comparison(self, fairness_data: Dict, save: bool = True):
        """
        Compare fairness metrics across different defenses
        """
        # Check if data is complete
        if not fairness_data or not any(fairness_data.values()):
            print("Skipping fairness comparison - no data available")
            return None
        
        # Extract data
        defenses = []
        dp_diffs = []
        eo_diffs = []
        accuracies = []
        
        for defense_name, results in fairness_data.items():
            if not results or not isinstance(results, dict):
                continue
                
            defenses.append(defense_name.replace('_', '\n'))
            
            # Check if we have the complete structure
            if 'overall_accuracy' in results:
                accuracies.append(results['overall_accuracy'])
            else:
                accuracies.append(0)
            
            # Average across attributes if available
            if 'by_attribute' in results and results['by_attribute']:
                dp_vals = []
                eo_vals = []
                for attr_results in results['by_attribute'].values():
                    if 'demographic_parity' in attr_results:
                        dp_vals.append(attr_results['demographic_parity']['dp_difference'])
                    if 'equalized_odds' in attr_results:
                        eo_vals.append(attr_results['equalized_odds']['eo_difference'])
                
                dp_diffs.append(np.mean(dp_vals) if dp_vals else 0)
                eo_diffs.append(np.mean(eo_vals) if eo_vals else 0)
            else:
                dp_diffs.append(0)
                eo_diffs.append(0)
        
        if not defenses:
            print("Skipping fairness comparison - no valid defense data")
            return None
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        x = np.arange(len(defenses))
        width = 0.35
        
        # Left plot: Fairness metrics
        bars1 = ax1.bar(x - width/2, dp_diffs, width, label='DP Difference',
                       color='#F18F01', alpha=0.8)
        bars2 = ax1.bar(x + width/2, eo_diffs, width, label='EO Difference',
                       color='#C73E1D', alpha=0.8)
        
        ax1.set_xlabel('Defense Mechanism', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Fairness Gap', fontsize=12, fontweight='bold')
        ax1.set_title('Fairness Metrics Comparison\n(Lower is Better)',
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(defenses, fontsize=9)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Right plot: Accuracy
        bars = ax2.bar(x, accuracies, color='#2E86AB', alpha=0.8)
        ax2.set_xlabel('Defense Mechanism', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title('Model Accuracy by Defense', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(defenses, fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([min(accuracies)-0.05 if accuracies else 0.7, 1.0])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.plots_dir / 'fairness_comparison.png',
                       dpi=300, bbox_inches='tight')
            print(f"✓ Saved: fairness_comparison.png")
        
        return fig
    
    def plot_pareto_frontier(self, tradeoff_data: Dict, save: bool = True):
        """
        Plot Pareto frontier for multi-objective optimization
        """
        # Check if data is complete
        if not tradeoff_data or not all(k in tradeoff_data for k in ['epsilon', 'accuracy', 'avg_attack_auc', 'avg_dp_difference']):
            print("Skipping Pareto frontier - incomplete data")
            return None
        
        if not tradeoff_data['epsilon']:
            print("Skipping Pareto frontier - no data points")
            return None
        fig = plt.figure(figsize=(14, 6))
        
        # 2D Pareto: Privacy vs Utility
        ax1 = fig.add_subplot(121)
        accuracy = np.array(tradeoff_data['accuracy'])
        attack_auc = np.array(tradeoff_data['avg_attack_auc'])
        epsilon = np.array(tradeoff_data['epsilon'])
        
        # Privacy protection = 1 - attack_auc (higher is better)
        privacy_protection = 1 - attack_auc
        
        scatter = ax1.scatter(privacy_protection, accuracy, c=epsilon,
                            s=200, cmap='viridis', edgecolors='black', linewidth=2)
        
        # Add epsilon labels
        for i, eps in enumerate(epsilon):
            ax1.annotate(f'ε={eps:.1f}', (privacy_protection[i], accuracy[i]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=9, bbox=dict(boxstyle='round,pad=0.5',
                        fc='yellow', alpha=0.7))
        
        ax1.set_xlabel('Privacy Protection (1 - Attack AUC)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Utility (Accuracy)', fontsize=12, fontweight='bold')
        ax1.set_title('Privacy-Utility Pareto Frontier', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Privacy Budget (ε)', rotation=270, labelpad=20)
        
        # 3D-style plot: Privacy vs Utility vs Fairness
        ax2 = fig.add_subplot(122)
        dp_diff = np.array(tradeoff_data['avg_dp_difference'])
        
        # Fairness score (inverted so higher is better)
        fairness_score = 1 - dp_diff
        
        scatter2 = ax2.scatter(privacy_protection, accuracy, c=fairness_score,
                             s=200, cmap='RdYlGn', edgecolors='black', linewidth=2,
                             vmin=0.85, vmax=1.0)
        
        # Add epsilon labels
        for i, eps in enumerate(epsilon):
            ax2.annotate(f'ε={eps:.1f}', (privacy_protection[i], accuracy[i]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=9, bbox=dict(boxstyle='round,pad=0.5',
                        fc='lightblue', alpha=0.7))
        
        ax2.set_xlabel('Privacy Protection', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Utility (Accuracy)', fontsize=12, fontweight='bold')
        ax2.set_title('Privacy-Utility-Fairness Tradeoff', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Fairness Score', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.plots_dir / 'pareto_frontier.png',
                       dpi=300, bbox_inches='tight')
            print(f"✓ Saved: pareto_frontier.png")
        
        return fig
    
    def generate_all_plots(self):
        """Generate all visualization plots"""
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80 + "\n")
        
        # Load results
        try:
            with open(self.results_dir / 'attack_matrix.json', 'r') as f:
                attack_matrix = json.load(f)
            
            with open(self.results_dir / 'fairness_analysis.json', 'r') as f:
                fairness_data = json.load(f)
            
            with open(self.results_dir / 'tradeoff_data.json', 'r') as f:
                tradeoff_data = json.load(f)
        except FileNotFoundError as e:
            print(f"Error: Results files not found: {e}")
            print("\nPlease run the experiment first:")
            print("  python experiments/run_experiments.py")
            return
        
        # Generate plots
        plots_generated = 0
        
        print("1. Attack Resistance Matrix...")
        if self.plot_attack_resistance_matrix(attack_matrix) is not None:
            plots_generated += 1
        
        print("\n2. Privacy-Utility Tradeoff...")
        if self.plot_privacy_utility_tradeoff(tradeoff_data) is not None:
            plots_generated += 1
        
        print("\n3. Fairness Comparison...")
        if self.plot_fairness_comparison(fairness_data) is not None:
            plots_generated += 1
        
        print("\n4. Pareto Frontier...")
        if self.plot_pareto_frontier(tradeoff_data) is not None:
            plots_generated += 1
        
        print("\n" + "="*80)
        print(f"✓ {plots_generated}/4 plots generated successfully")
        if plots_generated < 4:
            print("\nSome plots were skipped due to incomplete data.")
            print("Run the full experiment to generate all visualizations.")
        print(f"✓ Saved to: {self.plots_dir}")
        print("="*80)


if __name__ == "__main__":
    visualizer = PrivacyVisualizer()
    visualizer.generate_all_plots()

"""
Experiment Orchestrator
Main script to run comprehensive privacy-utility-fairness analysis
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import project modules
import os
import sys
sys.path.append(os.getcwd())  

from config import ExperimentConfig, DEFENSE_CONFIGS, ATTACK_CONFIGS
from data.preprocessor import AdultDataProcessor, calculate_data_sensitivity
from models.neural_nets import create_model, StandardTrainer, DPSGDTrainer
from models.pate import PATETrainer
from attacks.membership_inference import AttackOrchestrator
from experiments.fairness import FairnessAnalyzer

class PrivacyGuardExperiment:
    """Main experiment orchestrator"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results_dir = Path('./results')
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_processor = AdultDataProcessor(config.random_seed)
        self.fairness_analyzer = FairnessAnalyzer(config.sensitive_attributes)
        self.attack_orchestrator = AttackOrchestrator()
        
        # Storage for results
        self.experiment_results = {
            'attack_matrix': {},
            'fairness_analysis': {},
            'utility_metrics': {},
            'privacy_analysis': {}
        }
        
    def prepare_data(self):
        """Load and preprocess Adult dataset"""
        print("="*80)
        print("STEP 1: DATA PREPARATION")
        print("="*80)
        
        # Load data
        print("\nLoading Adult dataset...")
        df = self.data_processor.load_data()
        print(f"✓ Loaded {len(df)} records")
        
        # Preprocess
        print("\nPreprocessing data...")
        X, y, metadata = self.data_processor.preprocess(df)
        print(f"✓ Features: {X.shape[1]}")
        print(f"✓ Samples: {X.shape[0]}")
        print(f"✓ Class distribution: {metadata['class_distribution']}")
        
        # Create splits
        print("\nCreating train/test/member splits...")
        self.data_splits = self.data_processor.create_train_test_member_splits(
            X, y, metadata,
            train_ratio=self.config.train_ratio,
            test_ratio=self.config.test_ratio,
            member_ratio=self.config.member_ratio
        )
        
        print(f"✓ Train set: {len(self.data_splits['X_train'])} samples")
        print(f"✓ Test set: {len(self.data_splits['X_test'])} samples")
        print(f"✓ Member set: {len(self.data_splits['X_member'])} samples")
        print(f"✓ Non-member set: {len(self.data_splits['X_nonmember'])} samples")
        
        # Calculate sensitivity
        sensitivity = calculate_data_sensitivity(X)
        print(f"\n✓ Data L2 sensitivity: {sensitivity:.4f}")
        
        self.input_dim = X.shape[1]
        
    def train_shadow_models(self, n_shadow: int = 10):
        """Train shadow models for attack training"""
        print("\n" + "="*80)
        print("STEP 2: TRAINING SHADOW MODELS")
        print("="*80)
        
        shadow_models = []
        
        # Create shadow data from available data
        X_available = np.vstack([
            self.data_splits['X_train'],
            self.data_splits['X_test']
        ])
        y_available = np.concatenate([
            self.data_splits['y_train'],
            self.data_splits['y_test']
        ])
        
        for i in tqdm(range(n_shadow), desc="Training shadow models"):
            # Sample data for this shadow model
            n_samples = len(X_available)
            indices = np.random.choice(n_samples, size=n_samples//2, replace=False)
            X_shadow = X_available[indices]
            y_shadow = y_available[indices]
            
            # Split into train/val
            split_idx = int(len(X_shadow) * 0.8)
            X_train_s = X_shadow[:split_idx]
            y_train_s = y_shadow[:split_idx]
            X_val_s = X_shadow[split_idx:]
            y_val_s = y_shadow[split_idx:]
            
            # Train shadow model
            model = create_model(self.input_dim, self.config.hidden_layers)
            trainer = StandardTrainer(
                model,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size
            )
            trainer.train(X_train_s, y_train_s, X_val_s, y_val_s, 
                         epochs=self.config.epochs, verbose=False)
            
            shadow_models.append(trainer)
        
        print(f"\n✓ Trained {len(shadow_models)} shadow models")
        self.shadow_models = shadow_models
        
    def train_target_models(self):
        """Train all target models with different defense mechanisms"""
        print("\n" + "="*80)
        print("STEP 3: TRAINING TARGET MODELS")
        print("="*80)
        
        self.target_models = {}
        
        X_train = self.data_splits['X_train']
        y_train = self.data_splits['y_train']
        X_val = self.data_splits['X_test']
        y_val = self.data_splits['y_test']
        
        for defense_name, defense_config in DEFENSE_CONFIGS.items():
            print(f"\n{defense_name}:")
            print("-" * 40)
            
            if defense_config['type'] == 'none':
                # Baseline model
                model = create_model(self.input_dim, self.config.hidden_layers)
                trainer = StandardTrainer(
                    model,
                    learning_rate=self.config.learning_rate,
                    batch_size=self.config.batch_size
                )
                print("Training baseline model (no privacy)...")
                trainer.train(X_train, y_train, X_val, y_val, 
                            epochs=self.config.epochs, verbose=True)
                
            elif defense_config['type'] == 'dpsgd':
                # DP-SGD model (use GroupNorm instead of BatchNorm)
                epsilon = defense_config['epsilon']
                model = create_model(self.input_dim, self.config.hidden_layers, use_dp=True)
                trainer = DPSGDTrainer(
                    model,
                    epsilon=epsilon,
                    delta=self.config.delta,
                    max_grad_norm=self.config.max_grad_norm,
                    learning_rate=self.config.learning_rate,
                    batch_size=self.config.batch_size
                )
                print(f"Training DP-SGD model (ε={epsilon})...")
                trainer.train(X_train, y_train, X_val, y_val,
                            epochs=self.config.epochs, verbose=True)
                
            elif defense_config['type'] == 'pate':
                # PATE model
                trainer = PATETrainer(
                    input_dim=self.input_dim,
                    hidden_layers=self.config.hidden_layers,
                    num_teachers=self.config.num_teachers,
                    epsilon=defense_config['epsilon'],
                    delta=self.config.delta
                )
                print("Training PATE model...")
                trainer.train(X_train, y_train, X_val, y_val,
                            epochs=self.config.epochs,
                            teacher_epochs=self.config.teacher_epochs,
                            verbose=True)
            
            self.target_models[defense_name] = trainer
            
            # Evaluate utility
            y_pred = trainer.predict(X_val)
            accuracy = np.mean(y_pred == y_val)
            print(f"✓ Validation Accuracy: {accuracy:.4f}")
        
        print(f"\n✓ Trained {len(self.target_models)} target models")
    
    def run_attacks(self):
        """Run all attacks on all target models"""
        print("\n" + "="*80)
        print("STEP 4: RUNNING MEMBERSHIP INFERENCE ATTACKS")
        print("="*80)
        
        # Train attacks using shadow models
        self.attack_orchestrator.train_attacks(
            target_model=None,  # Not needed for training
            shadow_models=self.shadow_models,
            X_shadow=self.data_splits['X_train'],
            y_shadow=self.data_splits['y_train'],
            X_test=self.data_splits['X_test'],
            y_test=self.data_splits['y_test']
        )
        
        # Evaluate attacks on each target model
        X_member = self.data_splits['X_member']
        y_member = self.data_splits['y_member']
        X_nonmember = self.data_splits['X_nonmember']
        y_nonmember = self.data_splits['y_nonmember']
        
        for defense_name, target_model in self.target_models.items():
            print(f"\n{defense_name}:")
            print("-" * 40)
            
            results = self.attack_orchestrator.evaluate_attacks(
                target_model,
                X_member, y_member,
                X_nonmember, y_nonmember
            )
            
            self.experiment_results['attack_matrix'][defense_name] = results
            
            # Print results
            for attack_name, metrics in results.items():
                print(f"  {attack_name}: AUC={metrics['auc']:.4f}, "
                      f"Acc={metrics['accuracy']:.4f}")
    
    def run_fairness_analysis(self):
        """Analyze fairness across all models"""
        print("\n" + "="*80)
        print("STEP 5: FAIRNESS ANALYSIS")
        print("="*80)
        
        X_test = self.data_splits['X_test']
        y_test = self.data_splits['y_test']
        demographics_test = self.data_splits['demographics_test']
        
        for defense_name, target_model in self.target_models.items():
            print(f"\n{defense_name}:")
            print("-" * 40)
            
            # Get predictions
            y_pred = target_model.predict(X_test)
            
            # Comprehensive fairness analysis
            fairness_results = self.fairness_analyzer.comprehensive_fairness_analysis(
                y_test, y_pred, demographics_test, defense_name
            )
            
            self.experiment_results['fairness_analysis'][defense_name] = fairness_results
            
            # Print report
            report = self.fairness_analyzer.create_fairness_report(fairness_results)
            print(report)
    
    def analyze_privacy_utility_tradeoffs(self):
        """Analyze privacy-utility-fairness tradeoffs"""
        print("\n" + "="*80)
        print("STEP 6: PRIVACY-UTILITY-FAIRNESS TRADEOFFS")
        print("="*80)
        
        # Collect data for different epsilon values
        tradeoff_data = {
            'epsilon': [],
            'accuracy': [],
            'avg_attack_auc': [],
            'avg_dp_difference': [],
            'avg_eo_difference': []
        }
        
        for defense_name in DEFENSE_CONFIGS.keys():
            if defense_name == 'baseline':
                continue
            
            # Extract epsilon
            config = DEFENSE_CONFIGS[defense_name]
            if 'epsilon' in config:
                epsilon = config['epsilon']
            else:
                continue
            
            # Get accuracy
            y_pred = self.target_models[defense_name].predict(
                self.data_splits['X_test']
            )
            accuracy = np.mean(y_pred == self.data_splits['y_test'])
            
            # Get attack performance
            attack_results = self.experiment_results['attack_matrix'][defense_name]
            avg_auc = np.mean([m['auc'] for m in attack_results.values()])
            
            # Get fairness metrics
            fairness = self.experiment_results['fairness_analysis'][defense_name]
            dp_diffs = []
            eo_diffs = []
            for attr_results in fairness['by_attribute'].values():
                dp_diffs.append(attr_results['demographic_parity']['dp_difference'])
                eo_diffs.append(attr_results['equalized_odds']['eo_difference'])
            
            tradeoff_data['epsilon'].append(epsilon)
            tradeoff_data['accuracy'].append(accuracy)
            tradeoff_data['avg_attack_auc'].append(avg_auc)
            tradeoff_data['avg_dp_difference'].append(np.mean(dp_diffs))
            tradeoff_data['avg_eo_difference'].append(np.mean(eo_diffs))
        
        self.experiment_results['tradeoff_data'] = tradeoff_data
        
        # Print summary
        print("\nPrivacy-Utility-Fairness Tradeoff Summary:")
        print("-" * 80)
        print(f"{'Epsilon':<12}{'Accuracy':<12}{'Attack AUC':<12}"
              f"{'DP Diff':<12}{'EO Diff':<12}")
        print("-" * 80)
        
        for i in range(len(tradeoff_data['epsilon'])):
            print(f"{tradeoff_data['epsilon'][i]:<12.2f}"
                  f"{tradeoff_data['accuracy'][i]:<12.4f}"
                  f"{tradeoff_data['avg_attack_auc'][i]:<12.4f}"
                  f"{tradeoff_data['avg_dp_difference'][i]:<12.4f}"
                  f"{tradeoff_data['avg_eo_difference'][i]:<12.4f}")
    
    def save_results(self):
        """Save all experimental results"""
        print("\n" + "="*80)
        print("STEP 7: SAVING RESULTS")
        print("="*80)
        
        # Save attack matrix
        with open(self.results_dir / 'attack_matrix.json', 'w') as f:
            # Convert numpy types to Python types for JSON
            attack_matrix_serializable = {}
            for defense, attacks in self.experiment_results['attack_matrix'].items():
                attack_matrix_serializable[defense] = {}
                for attack_name, metrics in attacks.items():
                    attack_matrix_serializable[defense][attack_name] = {
                        'auc': float(metrics['auc']),
                        'accuracy': float(metrics['accuracy'])
                    }
            json.dump(attack_matrix_serializable, f, indent=2)
        print("✓ Saved attack matrix")
        
        # Save fairness analysis
        with open(self.results_dir / 'fairness_analysis.json', 'w') as f:
            fairness_serializable = {}
            for defense, results in self.experiment_results['fairness_analysis'].items():
                fairness_serializable[defense] = {
                    'model_name': results['model_name'],
                    'overall_accuracy': float(results['overall_accuracy'])
                }
            json.dump(fairness_serializable, f, indent=2)
        print("✓ Saved fairness analysis")
        
        # Save tradeoff data
        with open(self.results_dir / 'tradeoff_data.json', 'w') as f:
            tradeoff_serializable = {
                k: [float(v) for v in vals]
                for k, vals in self.experiment_results['tradeoff_data'].items()
            }
            json.dump(tradeoff_serializable, f, indent=2)
        print("✓ Saved tradeoff data")
        
        # Save full results as pickle
        with open(self.results_dir / 'full_results.pkl', 'wb') as f:
            pickle.dump(self.experiment_results, f)
        print("✓ Saved full results")
        
        print(f"\nAll results saved to: {self.results_dir}")
    
    def run_full_experiment(self):
        """Run the complete experimental pipeline"""
        print("\n")
        print("="*80)
        print("PRIVACYGUARD ENHANCED: COMPREHENSIVE PRIVACY ANALYSIS")
        print("="*80)
        print("\nThis will run a comprehensive analysis including:")
        print("  • 4 different membership inference attacks")
        print("  • Multiple defense mechanisms (Baseline, DP-SGD, PATE)")
        print("  • Fairness analysis across demographic groups")
        print("  • Privacy-utility-fairness tradeoff analysis")
        print("\n" + "="*80 + "\n")
        
        # Run all steps
        self.prepare_data()
        self.train_shadow_models(n_shadow=self.config.shadow_models)
        self.train_target_models()
        self.run_attacks()
        self.run_fairness_analysis()
        self.analyze_privacy_utility_tradeoffs()
        self.save_results()
        
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nNext steps:")
        print("  1. Review results in ./results/ directory")
        print("  2. Generate visualizations using visualization module")
        print("  3. Launch interactive dashboard for exploration")
        print("\n")


def main():
    """Main entry point"""
    # Create configuration
    config = ExperimentConfig(
        epsilon_values=[0.5, 1.0, 2.0, 5.0],
        batch_size=256,
        epochs=30,  # Reduced for faster testing
        shadow_models=5,  # Reduced for faster testing
        num_teachers=10,
        random_seed=42
    )
    
    # Create and run experiment
    experiment = PrivacyGuardExperiment(config)
    experiment.run_full_experiment()


if __name__ == "__main__":
    main()

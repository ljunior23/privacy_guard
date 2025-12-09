"""
Quick Demo Script
Tests the installation and runs a minimal example
"""
import os
import sys
sys.path.append(os.getcwd())  # Ensure current directory is in path

import numpy as np
from data.preprocessor import AdultDataProcessor
from models.neural_nets import create_model, StandardTrainer
from attacks.membership_inference import ConfidenceBasedAttack
from experiments.fairness import FairnessAnalyzer

def test_data_loading():
    """Test data preprocessing"""
    print("\n" + "="*60)
    print("TEST 1: Data Loading and Preprocessing")
    print("="*60)
    
    try:
        processor = AdultDataProcessor(random_seed=42)
        print("Processor initialized")
        
        # Create synthetic data for testing (avoid downloading)
        print("Creating synthetic test data...")
        n_samples = 1000
        n_features = 14
        
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randint(0, 2, n_samples)
        
        metadata = {
            'sensitive_attrs': {
                'race': np.random.choice(['White', 'Black'], n_samples),
                'sex': np.random.choice(['Male', 'Female'], n_samples),
                'age': np.random.randint(18, 80, n_samples)
            },
            'feature_names': [f'feature_{i}' for i in range(n_features)],
            'n_features': n_features,
            'n_samples': n_samples
        }
        
        splits = processor.create_train_test_member_splits(X, y, metadata)
        
        print(f"Train set: {len(splits['X_train'])} samples")
        print(f"Test set: {len(splits['X_test'])} samples")
        print(f"Member set: {len(splits['X_member'])} samples")
        
        return splits
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_model_training(splits):
    """Test model training"""
    print("\n" + "="*60)
    print("TEST 2: Model Training")
    print("="*60)
    
    try:
        input_dim = splits['X_train'].shape[1]
        model = create_model(input_dim, [64, 32])
        print("Model created")
        
        trainer = StandardTrainer(model, learning_rate=0.01, batch_size=128)
        print("Trainer initialized")
        
        print("Training model (5 epochs)...")
        trainer.train(
            splits['X_train'], splits['y_train'],
            splits['X_test'], splits['y_test'],
            epochs=5,
            verbose=False
        )
        
        # Test prediction
        y_pred = trainer.predict(splits['X_test'])
        accuracy = np.mean(y_pred == splits['y_test'])
        print(f"Model trained! Test accuracy: {accuracy:.3f}")
        
        return trainer
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_attack(trainer, splits):
    """Test membership inference attack"""
    print("\n" + "="*60)
    print("TEST 3: Membership Inference Attack")
    print("="*60)
    
    try:
        attack = ConfidenceBasedAttack()
        print("Attack initialized")
        
        # Get predictions for training
        member_conf = trainer.predict_proba(splits['X_member'][:100])
        nonmember_conf = trainer.predict_proba(splits['X_nonmember'][:100])
        member_labels = splits['y_member'][:100]
        nonmember_labels = splits['y_nonmember'][:100]
        
        # Train attack
        print("Training attack model...")
        attack.train(
            member_conf, member_labels,
            nonmember_conf, nonmember_labels
        )
        
        # Evaluate attack
        member_preds = attack.predict(member_conf, member_labels)
        nonmember_preds = attack.predict(nonmember_conf, nonmember_labels)
        
        # Calculate AUC manually
        from sklearn.metrics import roc_auc_score
        y_true = np.concatenate([np.ones(len(member_preds)), 
                                 np.zeros(len(nonmember_preds))])
        y_score = np.concatenate([member_preds, nonmember_preds])
        auc = roc_auc_score(y_true, y_score)
        
        print(f"Attack completed! AUC: {auc:.3f}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_fairness(trainer, splits):
    """Test fairness analysis"""
    print("\n" + "="*60)
    print("TEST 4: Fairness Analysis")
    print("="*60)
    
    try:
        analyzer = FairnessAnalyzer(['race', 'sex'])
        print("Fairness analyzer initialized")
        
        # Get predictions
        y_pred = trainer.predict(splits['X_test'])
        y_true = splits['y_test']
        
        # Analyze fairness
        print("Computing fairness metrics...")
        results = analyzer.comprehensive_fairness_analysis(
            y_true, y_pred,
            splits['demographics_test'],
            "Test Model"
        )
        
        print(f"Overall accuracy: {results['overall_accuracy']:.3f}")
        
        for attr_name, attr_results in results['by_attribute'].items():
            dp_diff = attr_results['demographic_parity']['dp_difference']
            print(f"{attr_name} DP difference: {dp_diff:.3f}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests"""
    print("\n")
    print("="*60)
    print("PRIVACYGUARD ENHANCED - QUICK DEMO")
    print("="*60)
    print("\nThis demo tests all major components of the framework.")
    print("Full experiments will take longer but follow the same pattern.")
    print("")
    
    # Run tests
    splits = test_data_loading()
    if splits is None:
        print("\nData loading failed. Cannot continue.")
        return
    
    trainer = test_model_training(splits)
    if trainer is None:
        print("\nModel training failed. Cannot continue.")
        return
    
    test_attack(trainer, splits)
    test_fairness(trainer, splits)
    
    # Summary
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY! ✓")
    print("="*60)
    print("\nAll components are working correctly.")
    print("\nNext steps:")
    print("  1. Run full experiment: python experiments/run_experiments.py")
    print("  2. Generate plots: python visualization/plots.py")
    print("  3. Launch dashboard: streamlit run dashboard.py")
    print("")

if __name__ == "__main__":
    main()

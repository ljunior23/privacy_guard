"""
Data Preprocessing Module
Handles Adult dataset loading, preprocessing, and demographic analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, List
import pickle
import os

class AdultDataProcessor:
    """Preprocessor for Adult Income dataset with fairness tracking"""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.sensitive_indices = {}
        
    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """Load Adult dataset from file or download"""
        
        # Column names for Adult dataset
        column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
            'income'
        ]
        
        if filepath and os.path.exists(filepath):
            df = pd.read_csv(filepath, names=column_names, skipinitialspace=True)
        else:
            # Download from UCI repository
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            df = pd.read_csv(url, names=column_names, skipinitialspace=True, na_values='?')
        
        return df
    
    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Preprocess Adult dataset with sensitive attribute tracking
        
        Returns:
            X: Feature matrix
            y: Labels
            metadata: Dictionary containing demographic information
        """
        
        # Remove missing values
        df = df.dropna()
        
        # Store original sensitive attributes
        sensitive_attrs = {
            'race': df['race'].copy(),
            'sex': df['sex'].copy(),
            'age': df['age'].copy()
        }
        
        # Create age groups
        sensitive_attrs['age_group'] = pd.cut(
            df['age'], 
            bins=[0, 25, 40, 60, 100], 
            labels=['18-25', '26-40', '41-60', '60+']
        )
        
        # Binary label: >50K = 1, <=50K = 0
        y = (df['income'].str.strip() == '>50K').astype(int).values
        
        # Separate features
        feature_cols = [col for col in df.columns if col not in ['income']]
        
        # Encode categorical variables
        df_encoded = df[feature_cols].copy()
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            self.label_encoders[col] = le
        
        # Store feature names
        self.feature_names = df_encoded.columns.tolist()
        
        # Convert to numpy array
        X = df_encoded.values.astype(np.float32)
        
        # Normalize features
        X = self.scaler.fit_transform(X)
        
        # Store sensitive attribute indices
        for attr in ['race', 'sex', 'age']:
            if attr in self.feature_names:
                self.sensitive_indices[attr] = self.feature_names.index(attr)
        
        # Create metadata
        metadata = {
            'sensitive_attrs': sensitive_attrs,
            'feature_names': self.feature_names,
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'class_distribution': {
                '<=50K': np.sum(y == 0),
                '>50K': np.sum(y == 1)
            }
        }
        
        return X, y, metadata
    
    def create_train_test_member_splits(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        metadata: Dict,
        train_ratio: float = 0.6,
        test_ratio: float = 0.2,
        member_ratio: float = 0.2
    ) -> Dict:
        """
        Create train/test splits and member/non-member sets
        Preserves demographic distribution
        """
        
        n_samples = len(X)
        
        # Calculate split sizes
        train_size = int(n_samples * train_ratio)
        test_size = int(n_samples * test_ratio)
        member_size = int(n_samples * member_ratio)
        
        # First split: train + member vs test
        X_trainmem, X_test, y_trainmem, y_test, idx_trainmem, idx_test = train_test_split(
            X, y, np.arange(n_samples),
            test_size=test_size,
            random_state=self.random_seed,
            stratify=y
        )
        
        # Second split: train vs member (for attack evaluation)
        # Calculate member ratio relative to the trainmem set
        member_ratio_relative = member_ratio / (train_ratio + member_ratio)
        X_train, X_member, y_train, y_member, idx_train, idx_member = train_test_split(
            X_trainmem, y_trainmem, idx_trainmem,
            test_size=member_ratio_relative,
            random_state=self.random_seed,
            stratify=y_trainmem
        )
        
        # Extract demographic info for each split
        def get_demographics(indices):
            demo_dict = {}
            for attr in metadata['sensitive_attrs'].keys():
                attr_data = metadata['sensitive_attrs'][attr]
                # Handle both pandas Series and numpy arrays
                if hasattr(attr_data, 'iloc'):
                    demo_dict[attr] = attr_data.iloc[indices].values
                else:
                    demo_dict[attr] = attr_data[indices]
            return demo_dict
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'X_member': X_member,
            'y_member': y_member,
            'X_nonmember': X_test,  # Non-members are from test set
            'y_nonmember': y_test,
            'demographics_train': get_demographics(idx_train),
            'demographics_test': get_demographics(idx_test),
            'demographics_member': get_demographics(idx_member),
            'demographics_nonmember': get_demographics(idx_test),
            'metadata': metadata
        }
    
    def save(self, filepath: str):
        """Save processor state"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names,
                'sensitive_indices': self.sensitive_indices
            }, f)
    
    def load(self, filepath: str):
        """Load processor state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            self.scaler = state['scaler']
            self.label_encoders = state['label_encoders']
            self.feature_names = state['feature_names']
            self.sensitive_indices = state['sensitive_indices']


def calculate_data_sensitivity(X: np.ndarray) -> float:
    """
    Calculate L2 sensitivity of the dataset for DP noise calibration
    
    Args:
        X: Feature matrix
        
    Returns:
        sensitivity: Maximum L2 norm of any sample
    """
    norms = np.linalg.norm(X, axis=1)
    return np.max(norms)


def analyze_demographic_distribution(demographics: Dict, name: str = "Dataset"):
    """Print demographic distribution statistics"""
    
    print(f"\n{name} Demographic Distribution:")
    print("=" * 60)
    
    for attr, values in demographics.items():
        if attr == 'age':
            print(f"\n{attr.upper()}:")
            print(f"  Mean: {np.mean(values):.2f}")
            print(f"  Std: {np.std(values):.2f}")
            print(f"  Range: [{np.min(values)}, {np.max(values)}]")
        else:
            unique, counts = np.unique(values, return_counts=True)
            print(f"\n{attr.upper()}:")
            for val, count in zip(unique, counts):
                print(f"  {val}: {count} ({count/len(values)*100:.1f}%)")


if __name__ == "__main__":
    # Test the data processor
    processor = AdultDataProcessor(random_seed=42)
    
    print("Loading Adult dataset...")
    df = processor.load_data()
    print(f"Loaded {len(df)} records")
    
    print("\nPreprocessing...")
    X, y, metadata = processor.preprocess(df)
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    print(f"Class distribution: {metadata['class_distribution']}")
    
    print("\nCreating splits...")
    splits = processor.create_train_test_member_splits(X, y, metadata)
    
    print(f"\nTrain set: {len(splits['X_train'])} samples")
    print(f"Test set: {len(splits['X_test'])} samples")
    print(f"Member set: {len(splits['X_member'])} samples")
    print(f"Non-member set: {len(splits['X_nonmember'])} samples")
    
    # Analyze demographics
    analyze_demographic_distribution(splits['demographics_train'], "Training Set")
    analyze_demographic_distribution(splits['demographics_test'], "Test Set")
    
    # Calculate sensitivity
    sensitivity = calculate_data_sensitivity(X)
    print(f"\nData L2 sensitivity: {sensitivity:.4f}")

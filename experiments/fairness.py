"""
Fairness Analysis Module
Evaluates fairness metrics across demographic groups
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix

class FairnessAnalyzer:
    """
    Analyzes fairness metrics for model predictions
    Implements demographic parity, equalized odds, and equal opportunity
    """
    
    def __init__(self, sensitive_attributes: List[str]):
        self.sensitive_attributes = sensitive_attributes
        self.metrics_history = []
    
    def calculate_demographic_parity(
        self,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> Dict:
        """
        Calculate demographic parity difference
        P(Y_pred=1 | Group A) - P(Y_pred=1 | Group B)
        """
        groups = np.unique(sensitive_attr)
        positive_rates = {}
        
        for group in groups:
            mask = sensitive_attr == group
            if np.sum(mask) > 0:
                positive_rate = np.mean(y_pred[mask] == 1)
                positive_rates[str(group)] = positive_rate
        
        if len(positive_rates) >= 2:
            rates = list(positive_rates.values())
            dp_difference = max(rates) - min(rates)
        else:
            dp_difference = 0.0
        
        return {
            'positive_rates': positive_rates,
            'dp_difference': dp_difference
        }
    
    def calculate_equalized_odds(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> Dict:
        """
        Calculate equalized odds difference
        Difference in TPR and FPR across groups
        """
        groups = np.unique(sensitive_attr)
        tpr_by_group = {}
        fpr_by_group = {}
        
        for group in groups:
            mask = sensitive_attr == group
            if np.sum(mask) > 0:
                y_true_group = y_true[mask]
                y_pred_group = y_pred[mask]
                
                # Calculate TPR and FPR
                cm = confusion_matrix(y_true_group, y_pred_group)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                else:
                    tpr, fpr = 0, 0
                
                tpr_by_group[str(group)] = tpr
                fpr_by_group[str(group)] = fpr
        
        if len(tpr_by_group) >= 2:
            tpr_values = list(tpr_by_group.values())
            fpr_values = list(fpr_by_group.values())
            tpr_diff = max(tpr_values) - min(tpr_values)
            fpr_diff = max(fpr_values) - min(fpr_values)
            eo_difference = max(tpr_diff, fpr_diff)
        else:
            eo_difference = 0.0
        
        return {
            'tpr_by_group': tpr_by_group,
            'fpr_by_group': fpr_by_group,
            'eo_difference': eo_difference
        }
    
    def calculate_equal_opportunity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> Dict:
        """
        Calculate equal opportunity difference
        Difference in TPR (recall) across groups
        """
        groups = np.unique(sensitive_attr)
        tpr_by_group = {}
        
        for group in groups:
            mask = sensitive_attr == group
            if np.sum(mask) > 0:
                y_true_group = y_true[mask]
                y_pred_group = y_pred[mask]
                
                # Calculate TPR
                cm = confusion_matrix(y_true_group, y_pred_group)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                else:
                    tpr = 0
                
                tpr_by_group[str(group)] = tpr
        
        if len(tpr_by_group) >= 2:
            tpr_values = list(tpr_by_group.values())
            eop_difference = max(tpr_values) - min(tpr_values)
        else:
            eop_difference = 0.0
        
        return {
            'tpr_by_group': tpr_by_group,
            'eop_difference': eop_difference
        }
    
    def calculate_disparate_impact(
        self,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> Dict:
        """
        Calculate disparate impact ratio
        min(P(Y_pred=1|group)) / max(P(Y_pred=1|group))
        """
        groups = np.unique(sensitive_attr)
        positive_rates = {}
        
        for group in groups:
            mask = sensitive_attr == group
            if np.sum(mask) > 0:
                positive_rate = np.mean(y_pred[mask] == 1)
                positive_rates[str(group)] = positive_rate
        
        if len(positive_rates) >= 2:
            rates = list(positive_rates.values())
            di_ratio = min(rates) / max(rates) if max(rates) > 0 else 1.0
        else:
            di_ratio = 1.0
        
        return {
            'positive_rates': positive_rates,
            'di_ratio': di_ratio
        }
    
    def calculate_accuracy_by_group(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> Dict:
        """Calculate per-group accuracy"""
        groups = np.unique(sensitive_attr)
        accuracy_by_group = {}
        
        for group in groups:
            mask = sensitive_attr == group
            if np.sum(mask) > 0:
                acc = np.mean(y_true[mask] == y_pred[mask])
                accuracy_by_group[str(group)] = acc
        
        return accuracy_by_group
    
    def comprehensive_fairness_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        demographics: Dict,
        model_name: str = "Model"
    ) -> Dict:
        """
        Perform comprehensive fairness analysis across all sensitive attributes
        """
        results = {
            'model_name': model_name,
            'overall_accuracy': np.mean(y_true == y_pred),
            'by_attribute': {}
        }
        
        for attr_name in self.sensitive_attributes:
            if attr_name not in demographics:
                continue
            
            sensitive_attr = demographics[attr_name]
            
            # Calculate all fairness metrics
            dp = self.calculate_demographic_parity(y_pred, sensitive_attr)
            eo = self.calculate_equalized_odds(y_true, y_pred, sensitive_attr)
            eop = self.calculate_equal_opportunity(y_true, y_pred, sensitive_attr)
            di = self.calculate_disparate_impact(y_pred, sensitive_attr)
            acc_by_group = self.calculate_accuracy_by_group(y_true, y_pred, sensitive_attr)
            
            results['by_attribute'][attr_name] = {
                'demographic_parity': dp,
                'equalized_odds': eo,
                'equal_opportunity': eop,
                'disparate_impact': di,
                'accuracy_by_group': acc_by_group
            }
        
        self.metrics_history.append(results)
        return results
    
    def analyze_attack_vulnerability_by_group(
        self,
        attack_predictions: np.ndarray,
        is_member: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> Dict:
        """
        Analyze how vulnerable different demographic groups are to attacks
        """
        groups = np.unique(sensitive_attr)
        vulnerability_by_group = {}
        
        for group in groups:
            mask = sensitive_attr == group
            if np.sum(mask) > 0:
                group_attack_preds = attack_predictions[mask]
                group_is_member = is_member[mask]
                
                # Attack success rate for this group
                # Higher score for members = successful attack
                member_mask = group_is_member == 1
                nonmember_mask = group_is_member == 0
                
                if np.sum(member_mask) > 0 and np.sum(nonmember_mask) > 0:
                    # Average attack score for members vs non-members
                    member_scores = np.mean(group_attack_preds[member_mask])
                    nonmember_scores = np.mean(group_attack_preds[nonmember_mask])
                    vulnerability = member_scores - nonmember_scores
                else:
                    vulnerability = 0.0
                
                vulnerability_by_group[str(group)] = {
                    'vulnerability_score': vulnerability,
                    'member_avg_score': member_scores if np.sum(member_mask) > 0 else 0,
                    'nonmember_avg_score': nonmember_scores if np.sum(nonmember_mask) > 0 else 0,
                    'group_size': np.sum(mask)
                }
        
        return vulnerability_by_group
    
    def create_fairness_report(self, results: Dict) -> str:
        """Create a formatted fairness report"""
        report = []
        report.append("="*80)
        report.append(f"FAIRNESS ANALYSIS REPORT: {results['model_name']}")
        report.append("="*80)
        report.append(f"\nOverall Accuracy: {results['overall_accuracy']:.4f}")
        
        for attr_name, attr_results in results['by_attribute'].items():
            report.append(f"\n{'-'*80}")
            report.append(f"Sensitive Attribute: {attr_name.upper()}")
            report.append(f"{'-'*80}")
            
            # Accuracy by group
            report.append("\nAccuracy by Group:")
            for group, acc in attr_results['accuracy_by_group'].items():
                report.append(f"  {group}: {acc:.4f}")
            
            # Demographic parity
            dp = attr_results['demographic_parity']
            report.append(f"\nDemographic Parity Difference: {dp['dp_difference']:.4f}")
            report.append("  Positive Prediction Rates:")
            for group, rate in dp['positive_rates'].items():
                report.append(f"    {group}: {rate:.4f}")
            
            # Equalized odds
            eo = attr_results['equalized_odds']
            report.append(f"\nEqualized Odds Difference: {eo['eo_difference']:.4f}")
            report.append("  True Positive Rates:")
            for group, tpr in eo['tpr_by_group'].items():
                report.append(f"    {group}: {tpr:.4f}")
            report.append("  False Positive Rates:")
            for group, fpr in eo['fpr_by_group'].items():
                report.append(f"    {group}: {fpr:.4f}")
            
            # Equal opportunity
            eop = attr_results['equal_opportunity']
            report.append(f"\nEqual Opportunity Difference: {eop['eop_difference']:.4f}")
            
            # Disparate impact
            di = attr_results['disparate_impact']
            report.append(f"\nDisparate Impact Ratio: {di['di_ratio']:.4f}")
            report.append("  (Closer to 1.0 is more fair)")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)
    
    def compare_fairness_across_models(
        self,
        models_results: List[Dict]
    ) -> pd.DataFrame:
        """
        Compare fairness metrics across multiple models
        Returns a comparison DataFrame
        """
        comparison_data = []
        
        for result in models_results:
            model_name = result['model_name']
            overall_acc = result['overall_accuracy']
            
            row = {
                'Model': model_name,
                'Overall_Accuracy': overall_acc
            }
            
            # Extract key fairness metrics
            for attr_name, attr_results in result['by_attribute'].items():
                prefix = attr_name.title()
                row[f'{prefix}_DP_Diff'] = attr_results['demographic_parity']['dp_difference']
                row[f'{prefix}_EO_Diff'] = attr_results['equalized_odds']['eo_difference']
                row[f'{prefix}_DI_Ratio'] = attr_results['disparate_impact']['di_ratio']
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)


def create_fairness_utility_tradeoff_data(
    epsilon_values: List[float],
    accuracies: List[float],
    fairness_metrics: List[Dict],
    metric_name: str = 'dp_difference'
) -> pd.DataFrame:
    """
    Create data for privacy-utility-fairness tradeoff analysis
    """
    data = {
        'epsilon': epsilon_values,
        'accuracy': accuracies,
        metric_name: []
    }
    
    for metric in fairness_metrics:
        # Extract the specific fairness metric (averaged across attributes)
        metric_values = []
        for attr_results in metric['by_attribute'].values():
            if metric_name == 'dp_difference':
                metric_values.append(attr_results['demographic_parity']['dp_difference'])
            elif metric_name == 'eo_difference':
                metric_values.append(attr_results['equalized_odds']['eo_difference'])
            elif metric_name == 'di_ratio':
                metric_values.append(attr_results['disparate_impact']['di_ratio'])
        
        data[metric_name].append(np.mean(metric_values) if metric_values else 0)
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Test fairness analyzer
    np.random.seed(42)
    
    # Simulate data
    n_samples = 1000
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = y_true.copy()
    y_pred[np.random.random(n_samples) < 0.1] = 1 - y_pred[np.random.random(n_samples) < 0.1]
    
    race = np.random.choice(['White', 'Black', 'Asian', 'Hispanic'], n_samples)
    sex = np.random.choice(['Male', 'Female'], n_samples)
    
    demographics = {'race': race, 'sex': sex}
    
    analyzer = FairnessAnalyzer(['race', 'sex'])
    results = analyzer.comprehensive_fairness_analysis(
        y_true, y_pred, demographics, "Test Model"
    )
    
    report = analyzer.create_fairness_report(results)
    print(report)

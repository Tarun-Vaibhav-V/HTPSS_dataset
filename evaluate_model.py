"""
Model Evaluation and Analysis Script
=====================================
This script provides comprehensive evaluation metrics and analysis to help understand:
1. Where the model performs well
2. Where the model lags/struggles
3. Detailed performance metrics per disease
4. Confusion patterns
5. Feature importance analysis

Author: AI Assistant
Date: 2025-12-18
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support,
    top_k_accuracy_score
)
from data_preprocessor import DataPreprocessor
from disease_model import DiseasePredictor
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluation and analysis.
    """
    
    def __init__(self, model_path='models/disease_model.pkl', 
                 mappings_path='models/mappings.pkl'):
        """
        Initialize the evaluator.
        
        Args:
            model_path (str): Path to trained model
            mappings_path (str): Path to mappings file
        """
        self.predictor = DiseasePredictor()
        self.predictor.load_model(model_path)
        self.predictor.load_mappings(mappings_path)
        
        # Load test data
        preprocessor = DataPreprocessor('DiseaseAndSymptoms.csv')
        _, self.X_test, _, self.y_test = preprocessor.preprocess_pipeline(save=False)
        
        # Generate predictions
        self.y_pred = self.predictor.model.predict(self.X_test)
        self.y_pred_proba = self.predictor.model.predict_proba(self.X_test)
        
        # Disease mappings
        self.idx_to_disease = self.predictor.idx_to_disease
        self.disease_to_idx = self.predictor.disease_to_idx
        
    def overall_metrics(self):
        """
        Calculate and display overall model performance metrics.
        """
        print("="*80)
        print("📊 OVERALL MODEL PERFORMANCE METRICS")
        print("="*80)
        
        # Basic accuracy
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f"\n✅ Overall Accuracy: {accuracy*100:.2f}%")
        
        # Top-k accuracy
        top3_acc = top_k_accuracy_score(self.y_test, self.y_pred_proba, k=3)
        top5_acc = top_k_accuracy_score(self.y_test, self.y_pred_proba, k=5)
        
        print(f"✅ Top-3 Accuracy: {top3_acc*100:.2f}%")
        print(f"✅ Top-5 Accuracy: {top5_acc*100:.2f}%")
        
        # Precision, Recall, F1-Score
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, self.y_pred, average='weighted'
        )
        
        print(f"\n📈 Weighted Metrics:")
        print(f"   Precision: {precision*100:.2f}%")
        print(f"   Recall: {recall*100:.2f}%")
        print(f"   F1-Score: {f1*100:.2f}%")
        
        return {
            'accuracy': accuracy,
            'top3_accuracy': top3_acc,
            'top5_accuracy': top5_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def per_disease_analysis(self):
        """
        Analyze performance for each disease individually.
        Identifies which diseases the model predicts well vs poorly.
        """
        print("\n" + "="*80)
        print("🔍 PER-DISEASE PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Get classification report as dict
        report = classification_report(
            self.y_test, 
            self.y_pred,
            target_names=[self.idx_to_disease[i] for i in range(len(self.idx_to_disease))],
            output_dict=True,
            zero_division=0
        )
        
        # Convert to DataFrame for easier analysis
        df_report = pd.DataFrame(report).transpose()
        df_report = df_report[df_report['support'] > 0]  # Only diseases in test set
        df_report = df_report[:-3]  # Remove avg rows
        
        # Sort by F1-score
        df_report_sorted = df_report.sort_values('f1-score', ascending=False)
        
        print("\n🏆 TOP 10 BEST PERFORMING DISEASES (by F1-Score):")
        print("-" * 80)
        for idx, (disease, row) in enumerate(df_report_sorted.head(10).iterrows(), 1):
            print(f"{idx:2d}. {disease:30s} | F1: {row['f1-score']:.3f} | "
                  f"Precision: {row['precision']:.3f} | Recall: {row['recall']:.3f} | "
                  f"Support: {int(row['support'])}")
        
        print("\n⚠️  10 WORST PERFORMING DISEASES (by F1-Score):")
        print("-" * 80)
        worst = df_report_sorted.tail(10).sort_values('f1-score')
        for idx, (disease, row) in enumerate(worst.iterrows(), 1):
            print(f"{idx:2d}. {disease:30s} | F1: {row['f1-score']:.3f} | "
                  f"Precision: {row['precision']:.3f} | Recall: {row['recall']:.3f} | "
                  f"Support: {int(row['support'])}")
        
        # Save detailed report to CSV
        df_report_sorted.to_csv('models/per_disease_performance.csv')
        print(f"\n💾 Detailed report saved to: models/per_disease_performance.csv")
        
        return df_report_sorted
    
    def confusion_analysis(self, top_n=10):
        """
        Analyze confusion patterns - which diseases are confused with each other.
        
        Args:
            top_n (int): Number of top confusions to show
        """
        print("\n" + "="*80)
        print("🔀 CONFUSION PATTERN ANALYSIS")
        print("="*80)
        
        # Get confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        # Find top confusions (excluding diagonal)
        confusions = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j and cm[i][j] > 0:  # Exclude correct predictions
                    confusions.append({
                        'actual': self.idx_to_disease[i],
                        'predicted': self.idx_to_disease[j],
                        'count': cm[i][j]
                    })
        
        # Sort by count
        confusions_sorted = sorted(confusions, key=lambda x: x['count'], reverse=True)
        
        print(f"\n🚨 TOP {top_n} CONFUSION PAIRS (Actual → Incorrectly Predicted As):")
        print("-" * 80)
        
        if len(confusions_sorted) == 0:
            print("   ✅ Perfect predictions! No confusions found.")
        else:
            for idx, conf in enumerate(confusions_sorted[:top_n], 1):
                print(f"{idx:2d}. {conf['actual']:25s} → {conf['predicted']:25s} "
                      f"({conf['count']} times)")
        
        # Calculate total misclassifications
        total_errors = sum([c['count'] for c in confusions_sorted])
        print(f"\n📊 Total Misclassifications: {total_errors} out of {len(self.y_test)} "
              f"({total_errors/len(self.y_test)*100:.2f}%)")
        
        return confusions_sorted
    
    def feature_importance_analysis(self, top_n=20):
        """
        Analyze which symptoms are most important for predictions.
        
        Args:
            top_n (int): Number of top features to show
        """
        print("\n" + "="*80)
        print("🎯 FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        top_features = self.predictor.get_feature_importance(top_n=top_n)
        
        print(f"\n🔝 TOP {top_n} MOST IMPORTANT SYMPTOMS:")
        print("-" * 80)
        
        for idx, (symptom, importance) in enumerate(top_features, 1):
            bar_length = int(importance * 50)  # Scale for visualization
            bar = '█' * bar_length
            print(f"{idx:2d}. {symptom:40s} | {importance:.4f} {bar}")
        
        # Save to CSV
        importance_df = pd.DataFrame(top_features, columns=['Symptom', 'Importance'])
        importance_df.to_csv('models/feature_importance.csv', index=False)
        print(f"\n💾 Feature importance saved to: models/feature_importance.csv")
        
        return top_features
    
    def error_analysis(self):
        """
        Deep dive into errors - analyze misclassified samples.
        """
        print("\n" + "="*80)
        print("🔬 ERROR ANALYSIS - MISCLASSIFIED SAMPLES")
        print("="*80)
        
        # Find misclassified indices
        errors = np.where(self.y_pred != self.y_test)[0]
        
        print(f"\n❌ Total Errors: {len(errors)} out of {len(self.y_test)} samples")
        
        if len(errors) == 0:
            print("   ✅ Perfect predictions! No errors to analyze.")
            return
        
        # Analyze first 10 errors
        print(f"\n📋 Analyzing first 10 errors:")
        print("-" * 80)
        
        for i, err_idx in enumerate(errors[:10], 1):
            actual_disease = self.idx_to_disease[self.y_test[err_idx]]
            predicted_disease = self.idx_to_disease[self.y_pred[err_idx]]
            confidence = self.y_pred_proba[err_idx][self.y_pred[err_idx]] * 100
            
            # Get symptoms for this sample
            symptom_indices = np.where(self.X_test[err_idx] == 1)[0]
            symptoms = [self.predictor.idx_to_symptom[idx] for idx in symptom_indices]
            
            print(f"\n{i}. Actual: {actual_disease}")
            print(f"   Predicted: {predicted_disease} (confidence: {confidence:.1f}%)")
            print(f"   Symptoms: {', '.join(symptoms[:5])}" + 
                  (f" ... (+{len(symptoms)-5} more)" if len(symptoms) > 5 else ""))
    
    def performance_by_confidence(self):
        """
        Analyze accuracy by prediction confidence levels.
        """
        print("\n" + "="*80)
        print("📊 PERFORMANCE BY CONFIDENCE LEVEL")
        print("="*80)
        
        # Get max probability for each prediction
        confidences = np.max(self.y_pred_proba, axis=1)
        
        # Define confidence bins
        bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
        
        print("\n🎯 Accuracy by Confidence Range:")
        print("-" * 80)
        
        for low, high in bins:
            mask = (confidences >= low) & (confidences < high)
            if np.sum(mask) > 0:
                acc = accuracy_score(self.y_test[mask], self.y_pred[mask])
                count = np.sum(mask)
                print(f"Confidence {low:.0%}-{high:.0%}: Accuracy = {acc*100:.2f}% "
                      f"({count} samples)")
    
    def generate_visualizations(self):
        """
        Generate visualization plots for model performance.
        """
        print("\n" + "="*80)
        print("📈 GENERATING VISUALIZATIONS")
        print("="*80)
        
        # 1. Feature Importance Plot
        top_features = self.predictor.get_feature_importance(top_n=15)
        symptoms, importances = zip(*top_features)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(symptoms)), importances)
        plt.yticks(range(len(symptoms)), symptoms)
        plt.xlabel('Importance Score')
        plt.title('Top 15 Most Important Symptoms')
        plt.tight_layout()
        plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: models/feature_importance.png")
        
        # 2. Confidence Distribution
        confidences = np.max(self.y_pred_proba, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Confidence')
        plt.axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidences):.3f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig('models/confidence_distribution.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: models/confidence_distribution.png")
        
        plt.close('all')
        
    def run_complete_evaluation(self):
        """
        Run all evaluation analyses in sequence.
        """
        print("\n" + "="*80)
        print("🚀 STARTING COMPLETE MODEL EVALUATION")
        print("="*80)
        
        # 1. Overall metrics
        overall = self.overall_metrics()
        
        # 2. Per-disease analysis
        per_disease = self.per_disease_analysis()
        
        # 3. Confusion analysis
        confusions = self.confusion_analysis(top_n=15)
        
        # 4. Feature importance
        features = self.feature_importance_analysis(top_n=20)
        
        # 5. Error analysis
        self.error_analysis()
        
        # 6. Performance by confidence
        self.performance_by_confidence()
        
        # 7. Generate visualizations
        self.generate_visualizations()
        
        print("\n" + "="*80)
        print("✅ EVALUATION COMPLETE!")
        print("="*80)
        
        print("\n📁 Generated Files:")
        print("   • models/per_disease_performance.csv - Detailed per-disease metrics")
        print("   • models/feature_importance.csv - Symptom importance scores")
        print("   • models/feature_importance.png - Feature importance visualization")
        print("   • models/confidence_distribution.png - Prediction confidence plot")
        
        print("\n💡 KEY INSIGHTS:")
        print(f"   • Overall accuracy: {overall['accuracy']*100:.2f}%")
        print(f"   • Top-3 accuracy: {overall['top3_accuracy']*100:.2f}%")
        print(f"   • Check per_disease_performance.csv to see which diseases lag")
        print(f"   • Review confusion patterns to understand misclassifications")
        
        return {
            'overall': overall,
            'per_disease': per_disease,
            'confusions': confusions,
            'features': features
        }


def main():
    """
    Main execution function.
    """
    print("\n" + "🩺 " + "="*76 + " 🩺")
    print("   AI SYMPTOM ANALYSIS ASSISTANT - MODEL EVALUATION & ANALYSIS")
    print("🩺 " + "="*76 + " 🩺\n")
    
    # Check if model exists
    import os
    if not os.path.exists('models/disease_model.pkl'):
        print("❌ Error: Model not found!")
        print("   Please train the model first by running: python train.py")
        return
    
    # Create evaluator and run complete evaluation
    evaluator = ModelEvaluator()
    results = evaluator.run_complete_evaluation()
    
    print("\n" + "="*80)
    print("🎯 NEXT STEPS:")
    print("="*80)
    print("""
1. Review 'models/per_disease_performance.csv' to identify weak diseases
2. Check visualization plots for feature importance insights
3. Analyze confusion patterns to understand similar diseases
4. Consider collecting more data for poorly performing diseases
5. Experiment with different model hyperparameters
    """)


if __name__ == "__main__":
    main()

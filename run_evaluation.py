"""Quick evaluation runner that saves output to file"""
import sys

# Redirect output to file
with open('evaluation_results.txt', 'w', encoding='utf-8') as f:
    sys.stdout = f
    sys.stderr = f
    
    # Import and run evaluation
    from evaluate_model import ModelEvaluator
    
    evaluator = ModelEvaluator()
    results = evaluator.run_complete_evaluation()

# Restore stdout
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

print("✅ Evaluation complete! Results saved to evaluation_results.txt")
print("📊 Check the following files for detailed analysis:")
print("   • evaluation_results.txt - Full evaluation report")
print("   • models/per_disease_performance.csv - Per-disease metrics")
print("   • models/feature_importance.csv - Feature importance scores")
print("   • models/feature_importance.png - Visualization")
print("   • models/confidence_distribution.png - Confidence plot")

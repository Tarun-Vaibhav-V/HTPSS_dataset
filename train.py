"""
Training Script
===============
Complete pipeline for training the AI Symptom Analysis Assistant.

This script:
1. Preprocesses the disease-symptom dataset
2. Trains the ML model
3. Evaluates performance
4. Saves the model and mappings

Author: AI Assistant
Date: 2025-12-18
"""

import sys
import argparse
from data_preprocessor import DataPreprocessor
from disease_model import DiseasePredictor


def train_model(data_path='DiseaseAndSymptoms.csv', 
                model_type='random_forest',
                test_size=0.2,
                save_model=True):
    """
    Complete training pipeline.
    
    Args:
        data_path (str): Path to the dataset CSV
        model_type (str): Type of ML model ('random_forest' or 'gradient_boosting')
        test_size (float): Proportion of data for testing
        save_model (bool): Whether to save the trained model
    """
    print("\n" + "=" * 70)
    print("🚀 AI SYMPTOM ANALYSIS ASSISTANT - TRAINING PIPELINE")
    print("=" * 70)
    print(f"   Dataset: {data_path}")
    print(f"   Model Type: {model_type}")
    print(f"   Test Size: {test_size * 100}%")
    print("=" * 70)
    
    # ========================================================================
    # STEP 1: Data Preprocessing
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 70)
    
    preprocessor = DataPreprocessor(data_path)
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(save=True)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   - Total samples: {len(X_train) + len(X_test)}")
    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Testing samples: {len(X_test)}")
    print(f"   - Features (symptoms): {X_train.shape[1]}")
    print(f"   - Classes (diseases): {len(preprocessor.diseases)}")
    
    # ========================================================================
    # STEP 2: Model Training
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: MODEL TRAINING")
    print("=" * 70)
    
    predictor = DiseasePredictor(model_type=model_type)
    predictor.load_mappings('models/mappings.pkl')
    predictor.train(X_train, y_train)
    
    # ========================================================================
    # STEP 3: Model Evaluation
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: MODEL EVALUATION")
    print("=" * 70)
    
    eval_results = predictor.evaluate(X_test, y_test)
    
    # Show top important features (if RandomForest)
    if model_type == 'random_forest':
        print("\n" + "=" * 70)
        print("🔝 TOP 15 MOST IMPORTANT SYMPTOMS FOR PREDICTION")
        print("=" * 70)
        
        top_features = predictor.get_feature_importance(top_n=15)
        for i, (symptom, importance) in enumerate(top_features, 1):
            # Create a visual bar
            bar_length = int(importance * 50)
            bar = "█" * bar_length
            
            print(f"{i:2d}. {symptom.replace('_', ' ').title():30s} {bar} {importance:.4f}")
    
    # ========================================================================
    # STEP 4: Save Model
    # ========================================================================
    if save_model:
        print("\n" + "=" * 70)
        print("STEP 4: SAVING MODEL")
        print("=" * 70)
        
        predictor.save_model('models/disease_model.pkl')
        
        print("\n✅ Model files saved:")
        print("   - models/disease_model.pkl")
        print("   - models/mappings.pkl")
        print("   - models/disease_symptom_map.pkl")
    
    # ========================================================================
    # STEP 5: Quick Test
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: QUICK PREDICTION TEST")
    print("=" * 70)
    
    # Test Case 1: Dengue symptoms
    print("\n🧪 Test Case 1: Dengue-like symptoms")
    test_symptoms_1 = ['high_fever', 'joint_pain', 'skin_rash', 'headache', 'back_pain']
    predictions_1 = predictor.predict_with_details(test_symptoms_1, top_k=3)
    
    print(f"Input Symptoms: {', '.join(test_symptoms_1)}")
    print("\nTop 3 Predictions:")
    for i, pred in enumerate(predictions_1, 1):
        print(f"  {i}. {pred['disease']:30s} {pred['confidence']:5.1f}% [{pred['risk']}]")
    
    # Test Case 2: Diabetes symptoms
    print("\n🧪 Test Case 2: Diabetes-like symptoms")
    test_symptoms_2 = ['fatigue', 'weight_loss', 'excessive_hunger', 'increased_appetite', 'polyuria']
    predictions_2 = predictor.predict_with_details(test_symptoms_2, top_k=3)
    
    print(f"Input Symptoms: {', '.join(test_symptoms_2)}")
    print("\nTop 3 Predictions:")
    for i, pred in enumerate(predictions_2, 1):
        print(f"  {i}. {pred['disease']:30s} {pred['confidence']:5.1f}% [{pred['risk']}]")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"   Accuracy: {eval_results['accuracy'] * 100:.2f}%")
    print(f"   Top-3 Accuracy: {eval_results['top3_accuracy'] * 100:.2f}%")
    print("\nNext Steps:")
    print("   1. Run the symptom analyzer: python symptom_analyzer.py")
    print("   2. Start the interactive app: python app.py")
    print("=" * 70)
    
    return predictor, eval_results


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Train the AI Symptom Analysis Assistant'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='DiseaseAndSymptoms.csv',
        help='Path to the dataset CSV file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='random_forest',
        choices=['random_forest', 'gradient_boosting'],
        help='Type of ML model to train'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for testing (0.0 to 1.0)'
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Only evaluate an existing model (skip training)'
    )
    
    args = parser.parse_args()
    
    # Run training
    try:
        if args.evaluate:
            # Load and evaluate existing model
            print("📊 Evaluating existing model...")
            
            predictor = DiseasePredictor(model_type=args.model)
            predictor.load_model('models/disease_model.pkl')
            predictor.load_mappings('models/mappings.pkl')
            
            # Load test data
            from data_preprocessor import DataPreprocessor
            preprocessor = DataPreprocessor(args.data)
            X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(save=False)
            
            # Evaluate
            eval_results = predictor.evaluate(X_test, y_test)
            
            print(f"\n✅ Evaluation completed!")
            print(f"   Accuracy: {eval_results['accuracy'] * 100:.2f}%")
            print(f"   Top-3 Accuracy: {eval_results['top3_accuracy'] * 100:.2f}%")
        else:
            # Train new model
            predictor, eval_results = train_model(
                data_path=args.data,
                model_type=args.model,
                test_size=args.test_size,
                save_model=True
            )
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"   Make sure the dataset file exists at: {args.data}")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

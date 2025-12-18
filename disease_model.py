"""
Disease Prediction Model Module
================================
This module implements machine learning models for disease prediction
based on symptom input.

Author: AI Assistant
Date: 2025-12-18
"""

import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import top_k_accuracy_score
import os


class DiseasePredictor:
    """
    Machine Learning model for predicting diseases from symptoms.
    Uses an ensemble of RandomForest and GradientBoosting classifiers.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the disease predictor.
        
        Args:
            model_type (str): Type of model ('random_forest' or 'gradient_boosting')
        """
        self.model_type = model_type
        self.model = None
        self.idx_to_disease = None
        self.disease_to_idx = None
        self.symptom_to_idx = None
        self.all_symptoms = None
        
        # Initialize the model based on type
        if model_type == 'random_forest':
            # RandomForest: Good for interpretability and handles multi-class well
            self.model = RandomForestClassifier(
                n_estimators=200,      # Number of trees
                max_depth=20,          # Maximum depth of trees
                min_samples_split=5,   # Minimum samples to split a node
                random_state=42,
                n_jobs=-1              # Use all CPU cores
            )
        elif model_type == 'gradient_boosting':
            # GradientBoosting: Often more accurate but slower
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def load_mappings(self, mappings_path='models/mappings.pkl'):
        """
        Load symptom and disease mappings from preprocessed data.
        
        Args:
            mappings_path (str): Path to the mappings pickle file
        """
        print(f"📂 Loading mappings from {mappings_path}...")
        
        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)
        
        self.symptom_to_idx = mappings['symptom_to_idx']
        self.disease_to_idx = mappings['disease_to_idx']
        self.idx_to_disease = mappings['idx_to_disease']
        self.all_symptoms = mappings['all_symptoms']
        
        print(f"✅ Mappings loaded successfully!")
        print(f"   - Symptoms: {len(self.symptom_to_idx)}")
        print(f"   - Diseases: {len(self.disease_to_idx)}")
    
    def train(self, X_train, y_train):
        """
        Train the disease prediction model.
        
        Args:
            X_train (np.array): Training feature matrix
            y_train (np.array): Training labels
        """
        print(f"\n🎓 Training {self.model_type} model...")
        print(f"   - Training samples: {len(X_train)}")
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Calculate training accuracy
        train_preds = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_preds)
        
        print(f"✅ Training completed!")
        print(f"   - Training accuracy: {train_accuracy * 100:.2f}%")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (np.array): Test feature matrix
            y_test (np.array): Test labels
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        print(f"\n📊 Evaluating model on test set...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Top-3 accuracy (useful for medical diagnosis)
        top3_accuracy = top_k_accuracy_score(y_test, y_proba, k=3)
        
        print(f"✅ Evaluation Results:")
        print(f"   - Accuracy: {accuracy * 100:.2f}%")
        print(f"   - Top-3 Accuracy: {top3_accuracy * 100:.2f}%")
        
        # Detailed classification report
        print(f"\n📋 Detailed Classification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=[self.idx_to_disease[i] for i in range(len(self.disease_to_idx))],
            zero_division=0
        ))
        
        return {
            'accuracy': accuracy,
            'top3_accuracy': top3_accuracy,
            'predictions': y_pred,
            'probabilities': y_proba
        }
    
    def predict(self, symptoms_list, top_k=3):
        """
        Predict diseases from a list of symptoms.
        
        Args:
            symptoms_list (list): List of symptom names
            top_k (int): Number of top predictions to return
            
        Returns:
            list: List of tuples (disease_name, confidence_score)
        """
        # Create symptom vector
        symptom_vector = np.zeros(len(self.all_symptoms))
        
        # Mark present symptoms
        for symptom in symptoms_list:
            # Normalize symptom (lowercase, strip spaces)
            symptom = symptom.strip().lower()
            
            # Try to match symptom (case-insensitive)
            for known_symptom, idx in self.symptom_to_idx.items():
                if known_symptom.lower() == symptom or symptom in known_symptom.lower():
                    symptom_vector[idx] = 1
                    break
        
        # Check if any symptoms were recognized
        if symptom_vector.sum() == 0:
            return []
        
        # Reshape for prediction
        X = symptom_vector.reshape(1, -1)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(X)[0]
        
        # Get top k predictions
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        # Format results
        results = []
        for idx in top_indices:
            disease_name = self.idx_to_disease[idx]
            confidence = probabilities[idx]
            results.append((disease_name, confidence))
        
        return results
    
    def predict_with_details(self, symptoms_list, top_k=5):
        """
        Predict diseases with detailed information including confidence and risk.
        
        Args:
            symptoms_list (list): List of symptom names
            top_k (int): Number of top predictions to return
            
        Returns:
            list: List of dictionaries with disease details
        """
        predictions = self.predict(symptoms_list, top_k)
        
        detailed_results = []
        for disease_name, confidence in predictions:
            # Simple risk classification based on disease type and confidence
            risk_level = self._assess_risk_level(disease_name, confidence)
            
            detailed_results.append({
                'disease': disease_name,
                'confidence': confidence * 100,  # Convert to percentage
                'risk': risk_level
            })
        
        return detailed_results
    
    def _assess_risk_level(self, disease_name, confidence):
        """
        Assess risk level based on disease type and prediction confidence.
        
        Args:
            disease_name (str): Name of the disease
            confidence (float): Prediction confidence
            
        Returns:
            str: Risk level (Critical, High, Medium, Low)
        """
        # High-risk diseases (require immediate attention)
        critical_diseases = [
            'Heart attack', 'Paralysis (brain hemorrhage)', 
            'Hepatitis E', 'AIDS'
        ]
        
        high_risk_diseases = [
            'Dengue', 'Malaria', 'Tuberculosis', 'Pneumonia',
            'Typhoid', 'Hepatitis A', 'Hepatitis B', 'Hepatitis C',
            'Hepatitis D', 'Alcoholic hepatitis', 'Jaundice'
        ]
        
        # Critical risk: life-threatening conditions
        if any(critical in disease_name for critical in critical_diseases):
            if confidence > 0.5:
                return 'Critical'
            else:
                return 'High'
        
        # High risk: serious conditions
        if any(high_risk in disease_name for high_risk in high_risk_diseases):
            if confidence > 0.6:
                return 'High'
            else:
                return 'Medium'
        
        # Medium risk: moderate confidence or moderate conditions
        if confidence > 0.5:
            return 'Medium'
        
        # Low risk: low confidence predictions
        return 'Low'
    
    def save_model(self, output_path='models/disease_model.pkl'):
        """
        Save the trained model to disk.
        
        Args:
            output_path (str): Path to save the model
        """
        print(f"\n💾 Saving model to {output_path}...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save model
        with open(output_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"✅ Model saved successfully!")
    
    def load_model(self, model_path='models/disease_model.pkl'):
        """
        Load a trained model from disk.
        
        Args:
            model_path (str): Path to the saved model
        """
        print(f"📂 Loading model from {model_path}...")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"✅ Model loaded successfully!")
    
    def get_feature_importance(self, top_n=20):
        """
        Get the most important symptoms for disease prediction.
        Only works with RandomForest model.
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            list: List of tuples (symptom_name, importance_score)
        """
        if self.model_type != 'random_forest':
            print("⚠️ Feature importance only available for RandomForest model")
            return []
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Create symptom-importance pairs
        idx_to_symptom = {idx: symptom for symptom, idx in self.symptom_to_idx.items()}
        feature_importance = [
            (idx_to_symptom[i], importances[i]) 
            for i in range(len(importances))
        ]
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return feature_importance[:top_n]


# Example usage
if __name__ == "__main__":
    from data_preprocessor import DataPreprocessor
    
    # Load and preprocess data
    print("Starting model training pipeline...\n")
    preprocessor = DataPreprocessor('DiseaseAndSymptoms.csv')
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(save=True)
    
    # Initialize and train model
    predictor = DiseasePredictor(model_type='random_forest')
    predictor.load_mappings()
    predictor.train(X_train, y_train)
    
    # Evaluate model
    results = predictor.evaluate(X_test, y_test)
    
    # Save model
    predictor.save_model()
    
    # Show top important features
    print("\n🔝 Top 10 Most Important Symptoms:")
    top_features = predictor.get_feature_importance(top_n=10)
    for i, (symptom, importance) in enumerate(top_features, 1):
        print(f"   {i}. {symptom}: {importance:.4f}")

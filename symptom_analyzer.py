"""
Symptom Analyzer - Main AI Assistant Module
============================================
This is the main module that orchestrates symptom analysis, disease prediction,
and clinical explanation generation.

Author: AI Assistant
Date: 2025-12-18
"""

import re
import pickle
import numpy as np
from disease_model import DiseasePredictor


class SymptomAnalyzer:
    """
    Main AI assistant class for analyzing symptoms and predicting diseases.
    Combines ML predictions with symptom matching for robust diagnosis.
    """
    
    def __init__(self, model_path='models/disease_model.pkl', 
                 mappings_path='models/mappings.pkl',
                 disease_symptom_map_path='models/disease_symptom_map.pkl'):
        """
        Initialize the Symptom Analyzer.
        
        Args:
            model_path (str): Path to trained ML model
            mappings_path (str): Path to symptom/disease mappings
            disease_symptom_map_path (str): Path to disease-symptom mapping
        """
        self.predictor = DiseasePredictor()
        self.symptom_to_idx = None
        self.disease_to_idx = None
        self.all_symptoms = None
        self.disease_symptom_map = None
        
        # Load model and mappings
        try:
            self.predictor.load_model(model_path)
            self.predictor.load_mappings(mappings_path)
            
            # Copy references for easier access
            self.symptom_to_idx = self.predictor.symptom_to_idx
            self.disease_to_idx = self.predictor.disease_to_idx
            self.all_symptoms = self.predictor.all_symptoms
            
            # Load disease-symptom mapping
            with open(disease_symptom_map_path, 'rb') as f:
                self.disease_symptom_map = pickle.load(f)
            
            print("✅ Symptom Analyzer initialized successfully!")
            
        except FileNotFoundError as e:
            print("⚠️ Model files not found. Please train the model first.")
            print(f"   Run: python train.py")
            raise e
    
    def parse_symptoms(self, user_input):
        """
        Parse symptoms from user input (structured or unstructured).
        
        Args:
            user_input (str or list): User input as string or list of symptoms
            
        Returns:
            list: List of parsed symptoms
        """
        if isinstance(user_input, list):
            # Already a list, just clean it
            return [s.strip() for s in user_input if s.strip()]
        
        # Handle string input - split by common delimiters
        user_input = user_input.lower()
        
        # Replace common separators with commas
        user_input = re.sub(r'[,;|/]', ',', user_input)
        user_input = re.sub(r'\sand\s', ',', user_input)
        
        # Remove common phrases
        stopwords = ['i have', 'i am experiencing', 'symptoms:', 
                     'symptom:', 'my symptoms are', 'feeling']
        for phrase in stopwords:
            user_input = user_input.replace(phrase, '')
        
        # Split into symptoms
        symptoms = [s.strip() for s in user_input.split(',') if s.strip()]
        
        return symptoms
    
    def normalize_symptoms(self, symptoms_list):
        """
        Normalize user-provided symptoms to match dataset vocabulary.
        Uses fuzzy matching to handle typos and variations.
        
        Args:
            symptoms_list (list): List of user-provided symptoms
            
        Returns:
            tuple: (matched_symptoms, unmatched_symptoms)
        """
        matched = []
        unmatched = []
        
        for user_symptom in symptoms_list:
            user_symptom = user_symptom.strip().lower().replace(' ', '_')
            
            # Try exact match first
            if user_symptom in [s.lower() for s in self.all_symptoms]:
                # Find the actual symptom with correct casing
                for known_symptom in self.all_symptoms:
                    if known_symptom.lower() == user_symptom:
                        matched.append(known_symptom)
                        break
            else:
                # Try partial matching
                found = False
                for known_symptom in self.all_symptoms:
                    # Check if user symptom is contained in known symptom or vice versa
                    if (user_symptom in known_symptom.lower() or 
                        known_symptom.lower() in user_symptom):
                        matched.append(known_symptom)
                        found = True
                        break
                
                if not found:
                    unmatched.append(user_symptom)
        
        return matched, unmatched
    
    def symptom_matching_prediction(self, symptoms_list, top_k=5):
        """
        Alternative prediction method using symptom overlap scoring.
        Useful as a fallback when ML model has low confidence.
        
        Args:
            symptoms_list (list): List of normalized symptoms
            top_k (int): Number of top predictions to return
            
        Returns:
            list: List of tuples (disease_name, match_score)
        """
        if not symptoms_list:
            return []
        
        # Calculate overlap score for each disease
        disease_scores = []
        
        for disease, disease_symptoms in self.disease_symptom_map.items():
            # Calculate Jaccard similarity
            intersection = len(set(symptoms_list) & set(disease_symptoms))
            union = len(set(symptoms_list) | set(disease_symptoms))
            
            if union > 0:
                jaccard_score = intersection / union
                
                # Bonus for matching many symptoms
                symptom_coverage = intersection / len(symptoms_list) if symptoms_list else 0
                
                # Combined score
                combined_score = (jaccard_score * 0.5) + (symptom_coverage * 0.5)
                
                disease_scores.append((disease, combined_score))
        
        # Sort by score
        disease_scores.sort(key=lambda x: x[1], reverse=True)
        
        return disease_scores[:top_k]
    
    def predict_conditions(self, user_input, top_k=5, use_hybrid=True):
        """
        Main prediction method that combines ML and symptom matching.
        
        Args:
            user_input (str or list): User symptoms
            top_k (int): Number of predictions to return
            use_hybrid (bool): Whether to use hybrid ML + symptom matching
            
        Returns:
            dict: Prediction results with metadata
        """
        # Step 1: Parse symptoms
        parsed_symptoms = self.parse_symptoms(user_input)
        
        # Step 2: Normalize symptoms
        matched_symptoms, unmatched_symptoms = self.normalize_symptoms(parsed_symptoms)
        
        if not matched_symptoms:
            return {
                'error': 'No recognized symptoms found',
                'parsed_symptoms': parsed_symptoms,
                'unmatched_symptoms': unmatched_symptoms,
                'predictions': []
            }
        
        # Step 3: Get ML predictions
        ml_predictions = self.predictor.predict_with_details(matched_symptoms, top_k)
        
        # Step 4: Get symptom matching predictions
        if use_hybrid:
            symptom_predictions = self.symptom_matching_prediction(matched_symptoms, top_k)
            
            # Merge predictions (weighted average)
            merged = self._merge_predictions(ml_predictions, symptom_predictions)
        else:
            merged = ml_predictions
        
        # Step 5: Format results
        results = {
            'parsed_symptoms': parsed_symptoms,
            'matched_symptoms': matched_symptoms,
            'unmatched_symptoms': unmatched_symptoms,
            'predictions': merged,
            'num_symptoms_matched': len(matched_symptoms)
        }
        
        return results
    
    def _merge_predictions(self, ml_preds, symptom_preds, ml_weight=0.7):
        """
        Merge ML and symptom-matching predictions using weighted scoring.
        
        Args:
            ml_preds (list): ML model predictions
            symptom_preds (list): Symptom matching predictions
            ml_weight (float): Weight for ML predictions (0-1)
            
        Returns:
            list: Merged predictions
        """
        symptom_weight = 1.0 - ml_weight
        
        # Create score dictionary
        disease_scores = {}
        
        # Add ML scores
        for pred in ml_preds:
            disease = pred['disease']
            score = pred['confidence'] / 100.0  # Convert back to 0-1
            disease_scores[disease] = {
                'ml_score': score,
                'symptom_score': 0.0,
                'risk': pred['risk']
            }
        
        # Add symptom matching scores
        for disease, score in symptom_preds:
            if disease in disease_scores:
                disease_scores[disease]['symptom_score'] = score
            else:
                # Disease found by symptom matching but not ML
                disease_scores[disease] = {
                    'ml_score': 0.0,
                    'symptom_score': score,
                    'risk': 'Low'  # Default risk for low-confidence predictions
                }
        
        # Calculate combined scores
        merged_predictions = []
        for disease, scores in disease_scores.items():
            combined_score = (scores['ml_score'] * ml_weight + 
                            scores['symptom_score'] * symptom_weight)
            
            merged_predictions.append({
                'disease': disease,
                'confidence': combined_score * 100,  # Convert to percentage
                'risk': scores['risk']
            })
        
        # Sort by combined score
        merged_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return merged_predictions
    
    def classify_risk(self, disease_name, confidence):
        """
        Classify the urgency/risk level of a predicted condition.
        
        Args:
            disease_name (str): Name of the disease
            confidence (float): Prediction confidence (0-100)
            
        Returns:
            str: Risk level with urgency
        """
        return self.predictor._assess_risk_level(disease_name, confidence / 100.0)
    
    def generate_explanation(self, prediction_result, detailed=True):
        """
        Generate a human-readable explanation of the prediction.
        
        Args:
            prediction_result (dict): Result from predict_conditions()
            detailed (bool): Whether to include detailed information
            
        Returns:
            str: Formatted explanation text
        """
        if 'error' in prediction_result:
            return f"❌ {prediction_result['error']}"
        
        # Build explanation
        explanation = []
        explanation.append("=" * 70)
        explanation.append("🏥 AI SYMPTOM ANALYSIS RESULT")
        explanation.append("=" * 70)
        
        # Show recognized symptoms
        matched = prediction_result['matched_symptoms']
        unmatched = prediction_result['unmatched_symptoms']
        
        explanation.append(f"\n✅ Recognized Symptoms ({len(matched)}):")
        for symptom in matched:
            explanation.append(f"   • {symptom.replace('_', ' ').title()}")
        
        if unmatched:
            explanation.append(f"\n⚠️ Unrecognized Symptoms ({len(unmatched)}):")
            for symptom in unmatched:
                explanation.append(f"   • {symptom.replace('_', ' ').title()}")
        
        # Show predictions
        predictions = prediction_result['predictions']
        
        if not predictions:
            explanation.append("\n❌ No disease predictions could be made.")
            return "\n".join(explanation)
        
        explanation.append(f"\n📊 Top Predicted Conditions:")
        explanation.append("-" * 70)
        
        for i, pred in enumerate(predictions[:5], 1):
            disease = pred['disease']
            confidence = pred['confidence']
            risk = pred['risk']
            
            # Risk level emoji
            risk_emoji = {
                'Critical': '🔴',
                'High': '🟠',
                'Medium': '🟡',
                'Low': '🟢'
            }.get(risk, '⚪')
            
            explanation.append(f"\n{i}. {disease}")
            explanation.append(f"   Confidence: {confidence:.1f}%")
            explanation.append(f"   Risk Level: {risk_emoji} {risk}")
            
            if detailed and i <= 3:
                # Add disease-specific symptoms for top 3 predictions
                disease_symptoms = self.disease_symptom_map.get(disease, [])
                common_symptoms = set(matched) & set(disease_symptoms)
                
                if common_symptoms:
                    explanation.append(f"   Matching Symptoms: {', '.join([s.replace('_', ' ') for s in list(common_symptoms)[:5]])}")
        
        # Add disclaimer
        explanation.append("\n" + "=" * 70)
        explanation.append("⚠️ IMPORTANT MEDICAL DISCLAIMER")
        explanation.append("=" * 70)
        explanation.append("This is an AI-powered analysis tool for educational purposes only.")
        explanation.append("It is NOT a substitute for professional medical advice, diagnosis,")
        explanation.append("or treatment. Please consult a qualified healthcare provider for")
        explanation.append("any health concerns.")
        explanation.append("=" * 70)
        
        return "\n".join(explanation)
    
    def analyze(self, user_input, detailed=True):
        """
        Complete end-to-end analysis pipeline.
        
        Args:
            user_input (str or list): User symptoms
            detailed (bool): Whether to generate detailed explanation
            
        Returns:
            str: Complete analysis report
        """
        # Get predictions
        result = self.predict_conditions(user_input)
        
        # Generate explanation
        explanation = self.generate_explanation(result, detailed=detailed)
        
        return explanation


# Example usage
if __name__ == "__main__":
    try:
        # Initialize analyzer
        print("🚀 Initializing AI Symptom Analyzer...\n")
        analyzer = SymptomAnalyzer()
        
        # Test with sample symptoms
        print("\n" + "=" * 70)
        print("🧪 TEST CASE 1: Common Cold Symptoms")
        print("=" * 70)
        
        test_input_1 = "fever, headache, cough, runny nose, sore throat"
        result_1 = analyzer.analyze(test_input_1)
        print(result_1)
        
        print("\n\n" + "=" * 70)
        print("🧪 TEST CASE 2: Dengue-like Symptoms")
        print("=" * 70)
        
        test_input_2 = ["high_fever", "joint_pain", "skin_rash", "headache", "back_pain"]
        result_2 = analyzer.analyze(test_input_2)
        print(result_2)
        
        print("\n\n" + "=" * 70)
        print("🧪 TEST CASE 3: Diabetes Symptoms")
        print("=" * 70)
        
        test_input_3 = "increased thirst, frequent urination, fatigue, blurred vision"
        result_3 = analyzer.analyze(test_input_3)
        print(result_3)
        
    except FileNotFoundError:
        print("\n❌ Model files not found!")
        print("Please run the training script first:")
        print("   python train.py")

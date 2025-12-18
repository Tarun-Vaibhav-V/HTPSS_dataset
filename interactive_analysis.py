"""
Interactive Model Analysis Tool
================================
This tool helps you understand model behavior through interactive testing.

Features:
1. Test with custom symptoms
2. Analyze disease similarity
3. Test with incomplete information
4. Find minimum symptoms needed for accurate prediction
5. Explore symptom importance for specific diseases

Author: AI Assistant
Date: 2025-12-18
"""

import numpy as np
import pandas as pd
import pickle
from disease_model import DiseasePredictor
from symptom_analyzer import SymptomAnalyzer


class InteractiveAnalyzer:
    """Interactive analysis tool for model exploration."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.predictor = DiseasePredictor()
        self.predictor.load_model('models/disease_model.pkl')
        self.predictor.load_mappings('models/mappings.pkl')
        
        # Load disease-symptom mapping
        with open('models/disease_symptom_map.pkl', 'rb') as f:
            self.disease_symptom_map = pickle.load(f)
    
    def show_disease_symptoms(self, disease_name):
        """
        Show all symptoms associated with a disease.
        
        Args:
            disease_name (str): Name of the disease
        """
        if disease_name not in self.disease_symptom_map:
            print(f"❌ Disease '{disease_name}' not found!")
            print(f"\n📋 Available diseases: {', '.join(sorted(self.disease_symptom_map.keys()))[:200]}...")
            return
        
        symptoms = self.disease_symptom_map[disease_name]
        
        print(f"\n🦠 Disease: {disease_name}")
        print(f"📊 Number of symptoms: {len(symptoms)}")
        print(f"\n💊 Associated Symptoms:")
        for i, symptom in enumerate(sorted(symptoms), 1):
            print(f"   {i:2d}. {symptom}")
    
    def test_partial_symptoms(self, disease_name, num_symptoms_range=(1, 5)):
        """
        Test how model performs with partial symptom information.
        
        Args:
            disease_name (str): Disease to test
            num_symptoms_range (tuple): Range of symptoms to test (min, max)
        """
        if disease_name not in self.disease_symptom_map:
            print(f"❌ Disease '{disease_name}' not found!")
            return
        
        all_symptoms = list(self.disease_symptom_map[disease_name])
        
        print(f"\n{'='*80}")
        print(f"🧪 PARTIAL SYMPTOM TESTING: {disease_name}")
        print(f"{'='*80}")
        print(f"Total symptoms for this disease: {len(all_symptoms)}")
        
        min_symp, max_symp = num_symptoms_range
        
        for num in range(min_symp, min(max_symp + 1, len(all_symptoms) + 1)):
            # Take first N symptoms
            test_symptoms = all_symptoms[:num]
            
            # Predict
            predictions = self.predictor.predict(test_symptoms, top_k=5)
            
            print(f"\n📊 Testing with {num} symptom(s): {', '.join(test_symptoms[:3])}" + 
                  (f" ... (+{num-3} more)" if num > 3 else ""))
            print(f"   Top predictions:")
            
            for idx, (pred_disease, confidence) in enumerate(predictions, 1):
                marker = "✅" if pred_disease == disease_name else "❌"
                print(f"   {idx}. {marker} {pred_disease:35s} - {confidence*100:.1f}%")
            
            # Check if correct disease is in top 5
            top_diseases = [d for d, _ in predictions]
            if disease_name in top_diseases:
                rank = top_diseases.index(disease_name) + 1
                print(f"   ✅ Correct disease found at rank {rank}")
            else:
                print(f"   ❌ Correct disease NOT in top 5")
    
    def find_minimum_symptoms(self, disease_name, target_confidence=0.8):
        """
        Find minimum number of symptoms needed for confident prediction.
        
        Args:
            disease_name (str): Disease to test
            target_confidence (float): Desired confidence level
        """
        if disease_name not in self.disease_symptom_map:
            print(f"❌ Disease '{disease_name}' not found!")
            return
        
        all_symptoms = list(self.disease_symptom_map[disease_name])
        
        print(f"\n{'='*80}")
        print(f"🎯 MINIMUM SYMPTOMS ANALYSIS: {disease_name}")
        print(f"{'='*80}")
        print(f"Target confidence: {target_confidence*100}%")
        print(f"Total available symptoms: {len(all_symptoms)}")
        
        for num in range(1, len(all_symptoms) + 1):
            test_symptoms = all_symptoms[:num]
            predictions = self.predictor.predict(test_symptoms, top_k=5)
            
            # Check if target disease is predicted with enough confidence
            for pred_disease, confidence in predictions:
                if pred_disease == disease_name:
                    if confidence >= target_confidence:
                        print(f"\n✅ FOUND! Minimum symptoms needed: {num}")
                        print(f"   Confidence: {confidence*100:.1f}%")
                        print(f"   Symptoms: {', '.join(test_symptoms)}")
                        return num
                    else:
                        print(f"   {num} symptoms → Confidence: {confidence*100:.1f}% (too low)")
                    break
            else:
                print(f"   {num} symptoms → Disease not in top 5")
        
        print(f"\n⚠️ Unable to reach {target_confidence*100}% confidence with available symptoms")
    
    def compare_similar_diseases(self, disease1, disease2):
        """
        Compare symptoms of two diseases to see similarities.
        
        Args:
            disease1 (str): First disease
            disease2 (str): Second disease
        """
        if disease1 not in self.disease_symptom_map or disease2 not in self.disease_symptom_map:
            print(f"❌ One or both diseases not found!")
            return
        
        symptoms1 = set(self.disease_symptom_map[disease1])
        symptoms2 = set(self.disease_symptom_map[disease2])
        
        common = symptoms1 & symptoms2
        unique1 = symptoms1 - symptoms2
        unique2 = symptoms2 - symptoms1
        
        print(f"\n{'='*80}")
        print(f"🔍 DISEASE COMPARISON")
        print(f"{'='*80}")
        
        print(f"\n📊 {disease1}")
        print(f"   Total symptoms: {len(symptoms1)}")
        
        print(f"\n📊 {disease2}")
        print(f"   Total symptoms: {len(symptoms2)}")
        
        print(f"\n🤝 Common Symptoms ({len(common)}):")
        for symptom in sorted(common):
            print(f"   • {symptom}")
        
        print(f"\n🔵 Unique to {disease1} ({len(unique1)}):")
        for symptom in sorted(unique1):
            print(f"   • {symptom}")
        
        print(f"\n🔴 Unique to {disease2} ({len(unique2)}):")
        for symptom in sorted(unique2):
            print(f"   • {symptom}")
        
        similarity = len(common) / len(symptoms1 | symptoms2) * 100
        print(f"\n📈 Similarity Score: {similarity:.1f}%")
        
        if similarity > 70:
            print("   ⚠️ VERY SIMILAR - High risk of confusion!")
        elif similarity > 40:
            print("   ⚡ MODERATELY SIMILAR - Some overlap")
        else:
            print("   ✅ DISTINCT - Low risk of confusion")
    
    def test_custom_symptoms(self, symptoms_list):
        """
        Test model with custom symptom list.
        
        Args:
            symptoms_list (list): List of symptom names
        """
        print(f"\n{'='*80}")
        print(f"🩺 CUSTOM SYMPTOM TESTING")
        print(f"{'='*80}")
        print(f"Input symptoms: {', '.join(symptoms_list)}")
        
        predictions = self.predictor.predict(symptoms_list, top_k=10)
        
        print(f"\n📊 Top 10 Predictions:")
        for idx, (disease, confidence) in enumerate(predictions, 1):
            bar_length = int(confidence * 50)
            bar = '█' * bar_length
            print(f"{idx:2d}. {disease:35s} {confidence*100:5.1f}% {bar}")
    
    def show_all_diseases(self):
        """Show all available diseases."""
        print(f"\n{'='*80}")
        print(f"📋 ALL AVAILABLE DISEASES ({len(self.disease_symptom_map)})")
        print(f"{'='*80}\n")
        
        for idx, disease in enumerate(sorted(self.disease_symptom_map.keys()), 1):
            num_symptoms = len(self.disease_symptom_map[disease])
            print(f"{idx:2d}. {disease:45s} ({num_symptoms} symptoms)")
    
    def interactive_menu(self):
        """Run interactive menu."""
        while True:
            print("\n" + "="*80)
            print("🔬 INTERACTIVE MODEL ANALYZER")
            print("="*80)
            print("\nChoose an option:")
            print("1. View disease symptoms")
            print("2. Test with partial symptoms")
            print("3. Find minimum symptoms needed")
            print("4. Compare two diseases")
            print("5. Test custom symptoms")
            print("6. Show all diseases")
            print("7. Quick analysis examples")
            print("0. Exit")
            
            choice = input("\nEnter your choice (0-7): ").strip()
            
            if choice == "0":
                print("\n👋 Goodbye!")
                break
            
            elif choice == "1":
                disease = input("Enter disease name: ").strip()
                self.show_disease_symptoms(disease)
            
            elif choice == "2":
                disease = input("Enter disease name: ").strip()
                self.test_partial_symptoms(disease, (1, 10))
            
            elif choice == "3":
                disease = input("Enter disease name: ").strip()
                conf = float(input("Enter target confidence (0.0-1.0, default 0.8): ").strip() or "0.8")
                self.find_minimum_symptoms(disease, conf)
            
            elif choice == "4":
                disease1 = input("Enter first disease: ").strip()
                disease2 = input("Enter second disease: ").strip()
                self.compare_similar_diseases(disease1, disease2)
            
            elif choice == "5":
                symptoms = input("Enter symptoms (comma-separated): ").strip()
                symptoms_list = [s.strip() for s in symptoms.split(',')]
                self.test_custom_symptoms(symptoms_list)
            
            elif choice == "6":
                self.show_all_diseases()
            
            elif choice == "7":
                self.run_quick_examples()
            
            else:
                print("❌ Invalid choice!")
    
    def run_quick_examples(self):
        """Run some quick analysis examples."""
        print("\n" + "="*80)
        print("🎯 QUICK ANALYSIS EXAMPLES")
        print("="*80)
        
        # Example 1: Test Dengue with partial symptoms
        print("\n📌 Example 1: Dengue with partial symptoms")
        self.test_partial_symptoms("Dengue", (1, 5))
        
        # Example 2: Compare similar diseases
        print("\n📌 Example 2: Comparing Malaria and Dengue")
        self.compare_similar_diseases("Malaria", "Dengue")
        
        # Example 3: Test with common symptoms
        print("\n📌 Example 3: Testing with generic symptoms")
        self.test_custom_symptoms(["fever", "headache", "fatigue"])


def main():
    """Main execution."""
    print("\n🔬 " + "="*76 + " 🔬")
    print("   INTERACTIVE MODEL ANALYSIS TOOL")
    print("🔬 " + "="*76 + " 🔬\n")
    
    analyzer = InteractiveAnalyzer()
    
    # Ask user what they want to do
    print("What would you like to do?")
    print("1. Run interactive menu")
    print("2. Run quick examples")
    print("3. Analyze specific disease")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        analyzer.interactive_menu()
    elif choice == "2":
        analyzer.run_quick_examples()
    elif choice == "3":
        disease = input("Enter disease name: ").strip()
        analyzer.show_disease_symptoms(disease)
        print("\n" + "-"*80)
        analyzer.test_partial_symptoms(disease, (1, 8))
        print("\n" + "-"*80)
        analyzer.find_minimum_symptoms(disease, 0.8)
    else:
        print("\n👋 Running default quick examples...")
        analyzer.run_quick_examples()


if __name__ == "__main__":
    main()

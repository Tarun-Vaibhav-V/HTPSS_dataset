"""
Data Preprocessor Module
========================
This module handles loading and preprocessing the disease-symptom datasets
for training the AI Symptom Analysis Assistant.

Author: AI Assistant
Date: 2025-12-18
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import os


class DataPreprocessor:
    """
    Handles data loading, cleaning, and transformation for disease-symptom datasets.
    """
    
    def __init__(self, dataset_path='DiseaseAndSymptoms.csv'):
        """
        Initialize the preprocessor with dataset path.
        
        Args:
            dataset_path (str): Path to the primary disease-symptom CSV file
        """
        self.dataset_path = dataset_path
        self.df = None
        self.diseases = []
        self.all_symptoms = set()
        self.symptom_to_idx = {}
        self.disease_to_idx = {}
        self.idx_to_disease = {}
        
    def load_data(self):
        """
        Load the dataset from CSV file and perform initial inspection.
        
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        print(f"📂 Loading dataset from: {self.dataset_path}")
        self.df = pd.read_csv(self.dataset_path)
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   - Total rows: {len(self.df)}")
        print(f"   - Columns: {list(self.df.columns)}")
        
        return self.df
    
    def extract_symptoms(self):
        """
        Extract all unique symptoms from the dataset and create vocabulary.
        Symptoms are spread across columns Symptom_1 to Symptom_17.
        
        Returns:
            set: Set of all unique symptoms
        """
        print("\n🔍 Extracting unique symptoms...")
        
        # Get all symptom columns (Symptom_1 through Symptom_17)
        symptom_cols = [col for col in self.df.columns if col.startswith('Symptom_')]
        
        # Extract all unique symptoms
        for col in symptom_cols:
            symptoms = self.df[col].dropna().str.strip()
            self.all_symptoms.update(symptoms)
        
        # Remove empty strings if any
        self.all_symptoms.discard('')
        
        # Create symptom vocabulary mapping
        self.symptom_to_idx = {symptom: idx for idx, symptom in enumerate(sorted(self.all_symptoms))}
        
        print(f"✅ Found {len(self.all_symptoms)} unique symptoms")
        print(f"   Sample symptoms: {list(sorted(self.all_symptoms))[:5]}")
        
        return self.all_symptoms
    
    def extract_diseases(self):
        """
        Extract all unique diseases from the dataset.
        
        Returns:
            list: List of all unique diseases
        """
        print("\n🦠 Extracting unique diseases...")
        
        self.diseases = sorted(self.df['Disease'].unique())
        
        # Create disease mapping
        self.disease_to_idx = {disease: idx for idx, disease in enumerate(self.diseases)}
        self.idx_to_disease = {idx: disease for disease, idx in self.disease_to_idx.items()}
        
        print(f"✅ Found {len(self.diseases)} unique diseases")
        print(f"   Sample diseases: {self.diseases[:5]}")
        
        return self.diseases
    
    def create_symptom_matrix(self):
        """
        Create a binary symptom-disease matrix for ML training.
        Each row represents a unique disease-symptom combination.
        
        Returns:
            tuple: (X, y) where X is symptom matrix and y is disease labels
        """
        print("\n🔢 Creating symptom-disease matrix...")
        
        # Initialize lists to store data
        X_data = []  # Symptom vectors
        y_data = []  # Disease labels
        
        # Get symptom columns
        symptom_cols = [col for col in self.df.columns if col.startswith('Symptom_')]
        
        # Process each row in the dataset
        for idx, row in self.df.iterrows():
            disease = row['Disease']
            
            # Create binary symptom vector
            symptom_vector = np.zeros(len(self.all_symptoms))
            
            # Mark symptoms as present (1) in the vector
            for col in symptom_cols:
                symptom = str(row[col]).strip()
                if symptom and symptom in self.symptom_to_idx:
                    symptom_vector[self.symptom_to_idx[symptom]] = 1
            
            X_data.append(symptom_vector)
            y_data.append(self.disease_to_idx[disease])
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        print(f"✅ Matrix created successfully!")
        print(f"   - Shape: {X.shape} (rows, symptoms)")
        print(f"   - Number of classes (diseases): {len(np.unique(y))}")
        
        return X, y
    
    def create_disease_symptom_mapping(self):
        """
        Create a mapping of each disease to its common symptoms.
        Useful for symptom matching algorithms.
        
        Returns:
            dict: Dictionary mapping disease name to list of symptoms
        """
        print("\n📋 Creating disease-symptom mapping...")
        
        disease_symptoms = {}
        symptom_cols = [col for col in self.df.columns if col.startswith('Symptom_')]
        
        for disease in self.diseases:
            # Get all rows for this disease
            disease_rows = self.df[self.df['Disease'] == disease]
            
            # Collect all symptoms for this disease
            symptoms = set()
            for _, row in disease_rows.iterrows():
                for col in symptom_cols:
                    symptom = str(row[col]).strip()
                    if symptom and symptom != 'nan':
                        symptoms.add(symptom)
            
            disease_symptoms[disease] = sorted(list(symptoms))
        
        print(f"✅ Mapping created for {len(disease_symptoms)} diseases")
        
        return disease_symptoms
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Labels
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print(f"\n✂️ Splitting data (test_size={test_size})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"✅ Data split completed!")
        print(f"   - Training samples: {len(X_train)}")
        print(f"   - Testing samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessed_data(self, output_dir='models'):
        """
        Save preprocessed data and mappings to disk for later use.
        
        Args:
            output_dir (str): Directory to save the files
        """
        print(f"\n💾 Saving preprocessed data to {output_dir}/...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save mappings
        mappings = {
            'symptom_to_idx': self.symptom_to_idx,
            'disease_to_idx': self.disease_to_idx,
            'idx_to_disease': self.idx_to_disease,
            'all_symptoms': list(self.all_symptoms),
            'diseases': self.diseases
        }
        
        with open(f'{output_dir}/mappings.pkl', 'wb') as f:
            pickle.dump(mappings, f)
        
        # Save disease-symptom mapping
        disease_symptom_map = self.create_disease_symptom_mapping()
        with open(f'{output_dir}/disease_symptom_map.pkl', 'wb') as f:
            pickle.dump(disease_symptom_map, f)
        
        print(f"✅ Saved successfully!")
        print(f"   - {output_dir}/mappings.pkl")
        print(f"   - {output_dir}/disease_symptom_map.pkl")
    
    def preprocess_pipeline(self, save=True):
        """
        Run the complete preprocessing pipeline.
        
        Args:
            save (bool): Whether to save preprocessed data to disk
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("=" * 60)
        print("🚀 Starting Data Preprocessing Pipeline")
        print("=" * 60)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Extract symptoms and diseases
        self.extract_symptoms()
        self.extract_diseases()
        
        # Step 3: Create feature matrix
        X, y = self.create_symptom_matrix()
        
        # Step 4: Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Step 5: Save preprocessed data
        if save:
            self.save_preprocessed_data()
        
        print("\n" + "=" * 60)
        print("✅ Preprocessing Pipeline Completed Successfully!")
        print("=" * 60)
        
        return X_train, X_test, y_train, y_test


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor('DiseaseAndSymptoms.csv')
    
    # Run preprocessing pipeline
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(save=True)
    
    print(f"\n📊 Final Statistics:")
    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Testing samples: {len(X_test)}")
    print(f"   - Features (symptoms): {X_train.shape[1]}")
    print(f"   - Classes (diseases): {len(np.unique(y_train))}")

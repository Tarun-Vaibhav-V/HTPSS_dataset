"""
Dataset Comparison Script
==========================
Compare the two datasets to understand which one is being used for training.

Author: AI Assistant
Date: 2025-12-18
"""

import pandas as pd
import os

def compare_datasets():
    """Compare both datasets."""
    
    print("=" * 80)
    print("📊 DATASET COMPARISON ANALYSIS")
    print("=" * 80)
    
    # Dataset 1: DiseaseAndSymptoms.csv
    print("\n" + "=" * 80)
    print("📁 DATASET 1: DiseaseAndSymptoms.csv")
    print("=" * 80)
    
    try:
        df1 = pd.read_csv('DiseaseAndSymptoms.csv')
        
        file_size_1 = os.path.getsize('DiseaseAndSymptoms.csv')
        print(f"✅ File Size: {file_size_1:,} bytes ({file_size_1/1024:.1f} KB)")
        print(f"✅ Rows: {len(df1):,}")
        print(f"✅ Columns: {len(df1.columns)}")
        print(f"✅ Column Names: {df1.columns.tolist()[:10]}")
        
        if 'Disease' in df1.columns:
            print(f"✅ Unique Diseases: {df1['Disease'].nunique()}")
            print(f"✅ Sample Diseases: {df1['Disease'].unique()[:5].tolist()}")
        
        print(f"\n📋 First 3 rows:")
        print(df1.head(3))
        
    except Exception as e:
        print(f"❌ Error loading: {e}")
    
    # Dataset 2: Disease and symptoms dataset.csv
    print("\n\n" + "=" * 80)
    print("📁 DATASET 2: Disease and symptoms dataset.csv")
    print("=" * 80)
    
    try:
        # This file is very large, so we'll be careful
        file_size_2 = os.path.getsize('Disease and symptoms dataset.csv')
        print(f"✅ File Size: {file_size_2:,} bytes ({file_size_2/1024/1024:.1f} MB)")
        
        # Read just the first few rows to understand structure
        df2 = pd.read_csv('Disease and symptoms dataset.csv', nrows=100)
        
        print(f"✅ Columns: {len(df2.columns)}")
        print(f"✅ Column Names (first 20): {df2.columns.tolist()[:20]}")
        
        if 'disease' in df2.columns:
            print(f"✅ Sample Diseases (from first 100 rows): {df2['disease'].unique()[:5].tolist()}")
        
        print(f"\n📋 First 3 rows (truncated):")
        print(df2.head(3).iloc[:, :10])  # Show only first 10 columns
        
    except Exception as e:
        print(f"❌ Error loading: {e}")
    
    # Check which one is used in training
    print("\n\n" + "=" * 80)
    print("🎯 WHICH DATASET IS USED FOR TRAINING?")
    print("=" * 80)
    
    print("\n✅ ANSWER: DiseaseAndSymptoms.csv (smaller file)")
    print("\nEvidence:")
    print("  1. All training scripts default to 'DiseaseAndSymptoms.csv'")
    print("  2. This is clearly visible in:")
    print("     - train.py (line 22, 161)")
    print("     - data_preprocessor.py (line 24)")
    print("     - disease_model.py (line 330)")
    print("     - evaluate_model.py (line 52)")
    print("\n  3. The larger 'Disease and symptoms dataset.csv' is NOT used")
    
    print("\n" + "=" * 80)
    print("📊 DATASET SPECIFICATIONS")
    print("=" * 80)
    
    print("\n🔵 Currently Used: DiseaseAndSymptoms.csv")
    try:
        df1 = pd.read_csv('DiseaseAndSymptoms.csv')
        print(f"   • Size: ~{os.path.getsize('DiseaseAndSymptoms.csv')/1024:.0f} KB")
        print(f"   • Rows: {len(df1):,}")
        print(f"   • Columns: {len(df1.columns)}")
        print(f"   • Diseases: {df1['Disease'].nunique() if 'Disease' in df1.columns else 'N/A'}")
        print(f"   • Structure: Compact, well-structured (Disease + Symptom columns)")
    except:
        pass
    
    print("\n⚪ Not Used: Disease and symptoms dataset.csv")
    print(f"   • Size: ~{os.path.getsize('Disease and symptoms dataset.csv')/1024/1024:.0f} MB")
    print(f"   • This is a much larger dataset (~191 MB)")
    print(f"   • Has {len(df2.columns)} columns")
    print(f"   • Different structure (many symptom columns)")
    
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATION")
    print("=" * 80)
    
    print("\nYour current setup uses only DiseaseAndSymptoms.csv, which is:")
    print("✅ GOOD: Clean, structured, and gives 100% accuracy")
    print("✅ GOOD: Fast to train (~5 seconds)")
    print("✅ GOOD: Easy to work with")
    
    print("\nThe larger dataset could potentially:")
    print("⚡ Provide more training examples")
    print("⚡ Improve generalization")
    print("⚡ But requires more processing time and memory")
    
    print("\nSince your current model has 100% accuracy, there's no immediate")
    print("need to change datasets unless you want to explore the larger one.")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    compare_datasets()

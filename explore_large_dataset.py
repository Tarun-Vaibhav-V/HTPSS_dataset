"""
Large Dataset Exploration
=========================
Step 1: Understand the structure and quality of the 191 MB dataset.

Author: AI Assistant
Date: 2025-12-18
"""

import pandas as pd
import numpy as np
import os

def explore_large_dataset():
    """Explore the large dataset structure and quality."""
    
    print("="*80)
    print("🔍 PHASE 1: EXPLORING LARGE DATASET")
    print("="*80)
    
    filename = 'Disease and symptoms dataset.csv'
    
    # File size
    file_size = os.path.getsize(filename)
    print(f"\n📁 File: {filename}")
    print(f"   Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
    
    # Read first chunk to understand structure
    print("\n🔍 Reading first 1000 rows to understand structure...")
    df_sample = pd.read_csv(filename, nrows=1000)
    
    print(f"\n📊 DATASET STRUCTURE:")
    print(f"   Columns: {len(df_sample.columns)}")
    print(f"   Sample rows examined: {len(df_sample)}")
    
    # Show column names
    print(f"\n📋 First 20 column names:")
    for i, col in enumerate(df_sample.columns[:20], 1):
        print(f"   {i:2d}. {col}")
    
    if len(df_sample.columns) > 20:
        print(f"   ... and {len(df_sample.columns) - 20} more columns")
    
    # Check data types
    print(f"\n🔢 Data Types:")
    print(df_sample.dtypes.value_counts())
    
    # Check for disease column
    disease_col = None
    for col in ['disease', 'Disease', 'diseases', 'Diseases']:
        if col in df_sample.columns:
            disease_col = col
            break
    
    if disease_col:
        print(f"\n🦠 DISEASE INFORMATION:")
        print(f"   Disease column: '{disease_col}'")
        print(f"   Unique diseases (in sample): {df_sample[disease_col].nunique()}")
        print(f"\n   Sample diseases:")
        for i, disease in enumerate(df_sample[disease_col].unique()[:10], 1):
            count = (df_sample[disease_col] == disease).sum()
            print(f"   {i:2d}. {disease} ({count} samples in first 1000)")
    
    # Check symptom columns (should be binary 0/1)
    symptom_cols = [col for col in df_sample.columns if col != disease_col]
    print(f"\n💊 SYMPTOM COLUMNS:")
    print(f"   Number of symptom columns: {len(symptom_cols)}")
    
    # Check if binary
    if len(symptom_cols) > 0:
        sample_col = symptom_cols[0]
        unique_vals = df_sample[sample_col].unique()
        print(f"   Sample column '{sample_col}' unique values: {unique_vals}")
        
        if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            print(f"   ✅ Confirmed: Binary format (0/1)")
        else:
            print(f"   ⚠️ Warning: Not pure binary! Values: {unique_vals}")
    
    # Now read FULL dataset
    print(f"\n📖 Reading FULL dataset (this may take 30-60 seconds)...")
    df_full = pd.read_csv(filename)
    
    print(f"\n📊 FULL DATASET STATISTICS:")
    print(f"   Total rows: {len(df_full):,}")
    print(f"   Total columns: {len(df_full.columns)}")
    print(f"   Memory usage: {df_full.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    if disease_col:
        print(f"\n🦠 DISEASE DISTRIBUTION:")
        print(f"   Unique diseases: {df_full[disease_col].nunique()}")
        
        disease_counts = df_full[disease_col].value_counts()
        print(f"\n   Top 10 most common diseases:")
        for i, (disease, count) in enumerate(disease_counts.head(10).items(), 1):
            percentage = count / len(df_full) * 100
            print(f"   {i:2d}. {disease:40s} {count:6,} samples ({percentage:5.2f}%)")
        
        print(f"\n   Bottom 10 least common diseases:")
        for i, (disease, count) in enumerate(disease_counts.tail(10).items(), 1):
            percentage = count / len(df_full) * 100
            print(f"   {i:2d}. {disease:40s} {count:6,} samples ({percentage:5.2f}%)")
        
        # Check balance
        max_count = disease_counts.max()
        min_count = disease_counts.min()
        imbalance_ratio = max_count / min_count
        
        print(f"\n   📊 Balance Analysis:")
        print(f"      Max samples (most common): {max_count:,}")
        print(f"      Min samples (least common): {min_count:,}")
        print(f"      Imbalance ratio: {imbalance_ratio:.2f}x")
        
        if imbalance_ratio > 10:
            print(f"      ⚠️ WARNING: Highly imbalanced! Need to address this.")
        elif imbalance_ratio > 3:
            print(f"      ⚡ Moderately imbalanced. Consider stratified sampling.")
        else:
            print(f"      ✅ Well balanced dataset.")
    
    # Check missing values
    print(f"\n🔍 DATA QUALITY CHECK:")
    missing_total = df_full.isnull().sum().sum()
    print(f"   Missing values: {missing_total:,}")
    
    if missing_total > 0:
        print(f"   ⚠️ Columns with missing values:")
        missing_cols = df_full.isnull().sum()
        missing_cols = missing_cols[missing_cols > 0]
        for col, count in missing_cols.head(10).items():
            percentage = count / len(df_full) * 100
            print(f"      {col}: {count} ({percentage:.2f}%)")
    else:
        print(f"   ✅ No missing values!")
    
    # Symptom statistics
    print(f"\n💊 SYMPTOM STATISTICS:")
    symptom_cols = [col for col in df_full.columns if col != disease_col]
    
    # Calculate how many symptoms per sample (average)
    if len(symptom_cols) > 0:
        symptoms_per_sample = df_full[symptom_cols].sum(axis=1)
        print(f"   Symptoms per sample:")
        print(f"      Mean: {symptoms_per_sample.mean():.2f}")
        print(f"      Median: {symptoms_per_sample.median():.0f}")
        print(f"      Min: {symptoms_per_sample.min():.0f}")
        print(f"      Max: {symptoms_per_sample.max():.0f}")
        
        # Most common symptoms
        symptom_frequencies = df_full[symptom_cols].sum().sort_values(ascending=False)
        print(f"\n   Top 15 most common symptoms:")
        for i, (symptom, count) in enumerate(symptom_frequencies.head(15).items(), 1):
            percentage = count / len(df_full) * 100
            print(f"   {i:2d}. {symptom:40s} {count:6,} ({percentage:5.1f}%)")
    
    # Save summary
    print(f"\n💾 Saving exploration summary...")
    
    summary = {
        'total_samples': len(df_full),
        'total_features': len(df_full.columns),
        'disease_column': disease_col,
        'num_diseases': df_full[disease_col].nunique() if disease_col else 0,
        'imbalance_ratio': imbalance_ratio if disease_col else 0,
        'missing_values': missing_total,
        'avg_symptoms_per_sample': symptoms_per_sample.mean() if len(symptom_cols) > 0 else 0
    }
    
    with open('large_dataset_summary.txt', 'w') as f:
        f.write("LARGE DATASET EXPLORATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n\nDisease Distribution:\n")
        f.write("-"*60 + "\n")
        if disease_col:
            for disease, count in disease_counts.items():
                f.write(f"{disease}: {count}\n")
    
    print(f"   ✅ Saved to: large_dataset_summary.txt")
    
    # Decision point
    print(f"\n" + "="*80)
    print(f"✅ EXPLORATION COMPLETE!")
    print(f"="*80)
    
    print(f"\n💡 KEY FINDINGS:")
    print(f"   1. Total samples: {len(df_full):,}")
    print(f"   2. Unique diseases: {df_full[disease_col].nunique() if disease_col else 'N/A'}")
    print(f"   3. Symptom features: {len(symptom_cols)}")
    print(f"   4. Data quality: {'Good (no missing)' if missing_total == 0 else f'Has {missing_total} missing values'}")
    print(f"   5. Balance: {'Imbalanced' if imbalance_ratio > 3 else 'Balanced'}")
    
    print(f"\n🎯 RECOMMENDATION:")
    if imbalance_ratio > 10:
        print(f"   ⚠️ Dataset is highly imbalanced!")
        print(f"   → Use stratified sampling")
        print(f"   → Consider class weights in model")
    elif len(df_full) > 100000:
        print(f"   💡 Large dataset ({len(df_full):,} samples)")
        print(f"   → Start with 20% sample for quick experiment")
        print(f"   → Then train on full data if promising")
    else:
        print(f"   ✅ Good size, proceed directly with full data")
    
    return df_full, disease_col, symptom_cols


if __name__ == "__main__":
    df, disease_col, symptom_cols = explore_large_dataset()

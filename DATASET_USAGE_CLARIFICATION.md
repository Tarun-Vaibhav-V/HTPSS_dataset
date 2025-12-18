# 📊 Dataset Usage Clarification

## ❓ Your Question:
"Did we train our model on DiseaseAndSymptoms.csv and Disease and symptoms dataset.csv?"

---

## ✅ **ANSWER: NO - Only ONE dataset is used**

Your model was trained **ONLY** on:
- ✅ **DiseaseAndSymptoms.csv** (632 KB)

The other dataset is **NOT used**:
- ❌ **Disease and symptoms dataset.csv** (191 MB) - NOT USED

---

## 📁 Dataset Comparison

### 📌 **Dataset 1: DiseaseAndSymptoms.csv** ✅ **USED**

| Property | Value |
|----------|-------|
| **File Size** | 632 KB (small) |
| **Rows** | 4,920 samples |
| **Columns** | 18 columns |
| **Structure** | Disease + 17 Symptom columns |
| **Format** | Clean, structured CSV |
| **Status** | ✅ **Currently being used for training** |

**Example Structure:**
```
Disease,Symptom_1,Symptom_2,Symptom_3,...Symptom_17
Fungal infection,itching,skin_rash,nodal_skin_eruptions,...
AIDS,muscle_wasting,patches_in_throat,high_fever,...
```

**Key Features:**
- ✅ 41 unique diseases
- ✅ 131 unique symptoms
- ✅ Clean, well-labeled data
- ✅ Each disease has ~120 samples
- ✅ Fast to load and process (~5 seconds training)

---

### 📌 **Dataset 2: Disease and symptoms dataset.csv** ❌ **NOT USED**

| Property | Value |
|----------|-------|
| **File Size** | 191 MB (300x larger!) |
| **Rows** | Unknown (very large) |
| **Columns** | 378 columns |
| **Structure** | Disease + 377 binary symptom columns (0/1) |
| **Format** | Wide format with many columns |
| **Status** | ❌ **NOT used in training** |

**Example Structure:**
```
disease,anxiety_and_nervousness,depression,shortness_of_breath,...(375 more columns)
panic disorder,0,0,1,1,0,1,1,0,...
```

**Key Features:**
- ❓ Different diseases (possibly more)
- ❓ 377 binary symptom columns (0=absent, 1=present)
- ❌ NOT referenced in any training script
- ❌ Larger file = longer processing time

---

## 🔍 Evidence - Which Dataset is Used?

All code files reference **only** `DiseaseAndSymptoms.csv`:

### **1. train.py (Main Training Script)**
```python
# Line 22
def train_model(data_path='DiseaseAndSymptoms.csv', ...):

# Line 161
default='DiseaseAndSymptoms.csv',
```

### **2. data_preprocessor.py**
```python
# Line 24
def __init__(self, dataset_path='DiseaseAndSymptoms.csv'):
```

### **3. disease_model.py**
```python
# Line 330
preprocessor = DataPreprocessor('DiseaseAndSymptoms.csv')
```

### **4. evaluate_model.py**
```python
# Line 52
preprocessor = DataPreprocessor('DiseaseAndSymptoms.csv')
```

**Conclusion:** Every script defaults to `DiseaseAndSymptoms.csv`

---

## 📊 Current Model Training Details

Your model was trained with:

```
Dataset: DiseaseAndSymptoms.csv
├── Total Samples: 4,920
├── Training Samples: 3,936 (80%)
├── Testing Samples: 984 (20%)
├── Features (Symptoms): 131
├── Classes (Diseases): 41
└── Performance: 100% accuracy
```

---

## 🤔 Why Only One Dataset?

The smaller **DiseaseAndSymptoms.csv** was chosen because:

1. ✅ **Perfectly suited for the task**
   - Contains exactly what's needed
   - Well-structured and clean
   - 41 diseases covering major conditions

2. ✅ **Excellent results**
   - Achieved 100% accuracy
   - Fast training time (~5 seconds)
   - No need for larger dataset

3. ✅ **Easy to work with**
   - Standard CSV format
   - Clear symptom-disease mapping
   - Manageable file size

---

## 🎯 What About the Larger Dataset?

The **Disease and symptoms dataset.csv** (191 MB) is:

**Pros:**
- ⚡ More data (potentially better for generalization)
- ⚡ More symptom coverage (377 vs 131 symptoms)
- ⚡ Possibly more diseases

**Cons:**
- ⚠️ Much larger (300x bigger file)
- ⚠️ Slower processing
- ⚠️ Different format (binary 0/1)
- ⚠️ May require different preprocessing
- ⚠️ Not needed since current model is already perfect

**Current Status:** Available but unused

---

## 💡 Should You Use Both Datasets?

### **Option 1: Keep Current Setup (Recommended)** ✅

**Reasons:**
- Current model has 100% accuracy
- Fast training and inference
- Clean, manageable data
- Works perfectly for your use case

**Verdict:** If it ain't broke, don't fix it!

---

### **Option 2: Experiment with Larger Dataset** 🔬

**When to consider:**
- Want to test with more symptoms
- Need to validate on different data
- Research purposes
- Want to compare model performance

**How to do it:**
```bash
# Train with the larger dataset
python train.py --data "Disease and symptoms dataset.csv"
```

**Note:** You'll probably need to modify the preprocessing code since the structure is different (binary columns vs symptom names).

---

## 🎓 Summary Table

| Aspect | DiseaseAndSymptoms.csv | Disease and symptoms dataset.csv |
|--------|------------------------|-----------------------------------|
| **Used for Training** | ✅ YES | ❌ NO |
| **File Size** | 632 KB | 191 MB |
| **Format** | Symptom names | Binary 0/1 |
| **Columns** | 18 | 378 |
| **Processing Speed** | Fast | Slower |
| **Current Accuracy** | 100% | Not tested |
| **Code References** | Everywhere | Nowhere |

---

## 🚀 Quick Commands

### Check which dataset is being used:
```bash
# View train.py to see default dataset
grep -n "DiseaseAndSymptoms.csv" *.py
```

### Train with specific dataset:
```bash
# With default dataset (current)
python train.py

# With larger dataset (experimental)
python train.py --data "Disease and symptoms dataset.csv"
```

---

## 🎯 Final Answer

**Your model uses ONLY:**
- ✅ **DiseaseAndSymptoms.csv** (632 KB, 4,920 samples, 41 diseases, 131 symptoms)

**NOT used:**
- ❌ **Disease and symptoms dataset.csv** (191 MB, unused)

**Why it matters:**
- You have a second, larger dataset available
- It's currently not being used
- Your current model is already perfect (100% accuracy)
- You could experiment with it if you want more data

---

## 📝 To Verify Yourself:

1. **Check file sizes:**
   ```bash
   ls -lh *.csv
   ```

2. **Check code references:**
   ```bash
   grep "DiseaseAndSymptoms.csv" *.py
   grep "Disease and symptoms dataset.csv" *.py
   ```

3. **Check training output:**
   - Look at `evaluation_results.txt`
   - Shows: "Loading dataset from: DiseaseAndSymptoms.csv"

---

**Created:** 2025-12-18  
**Question:** Which datasets are used for training?  
**Answer:** Only DiseaseAndSymptoms.csv (632 KB)  
**Status:** Confirmed ✅

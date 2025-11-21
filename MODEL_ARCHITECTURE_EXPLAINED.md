# Credit Scoring Model Architecture - Detailed Explanation

## 🎯 Simple Overview

Think of it like a **3-stage decision-making process**:

1. **Layer 1 (Base Model)**: A general model that gives initial risk assessment for ALL applicants
2. **Layer 2 (Segment Models)**: Two specialized models that focus on different risk groups
3. **Layer 3 (Meta Model)**: A final model that combines all predictions to make the best decision

---

## 📊 How One Applicant Flows Through the System

Let's follow **one applicant (ID: CONT_001)** through all layers:

### Input: Applicant Data
```
MUQAVILE: CONT_001
AGE: 42
INCOME: 5000
CC_O_3MWPS_EVER: 0
CL_A_CWPS_90D: 1
... (100+ more features)
```

---

## Layer 1: Base Model (Foundation)

**What it does:** Analyzes ALL applicants using the same model

**Model Type:** Logistic Regression with WOE Binning

**Process:**
```
Raw Features (100+)
    ↓
Feature Elimination (removes bad features)
    ↓
WOE Binning (converts to risk buckets)
    ↓
Logistic Regression Model
    ↓
OUTPUT: Base Probability
```

**Output for CONT_001:**
```python
Base Probability = 0.015  # 1.5% chance of default

# Convert to score (using odds_at_ref=50 for segmentation)
Base Score = 210
```

**Key Point:** This probability is used to decide which segment models to apply!

---

## Layer 2: Segment Models (Specialization)

### Step 2A: Segmentation Decision

The **Base Score** determines which specialized models will analyze this applicant:

```python
# Segmentation thresholds
if Base_Score < 180:
    → Apply "Good" segment model (low risk group)

if Base_Score >= 200:
    → Apply "Not-Good" segment model (high risk group)

if 180 <= Base_Score < 200:
    → Apply BOTH models (borderline cases)
```

**For CONT_001 (score=210):**
- ✅ Gets scored by BOTH segment models (borderline case)

---

### Step 2B: Good Segment Model

**Trained on:** Applicants with Base_Score < 180 (low-risk population)

**Model Type:** LightGBM (tree-based, captures complex patterns)

**Why LGBM for Good segment?**
- Good applicants have subtle differences
- Need complex model to find patterns among similar applicants
- Tree models work better when default rate is low

**Process:**
```
Raw Features (100+)
    ↓
Feature Elimination
    ↓
Scaling (StandardScaler)
    ↓
Imputation (fill missing values)
    ↓
Target Encoding (categorical features)
    ↓
LightGBM Model
    ↓
OUTPUT: Good Segment Probability
```

**Output for CONT_001:**
```python
Good_Segment_Probability = 0.009  # 0.9% chance of default
```

---

### Step 2C: Not-Good Segment Model

**Trained on:** Applicants with Base_Score >= 200 (high-risk population)

**Model Type:** Logistic Regression

**Why Logistic Regression for Not-Good segment?**
- High-risk applicants already have clear signals
- Don't need complex models (might overfit)
- Linear model provides stability

**Process:**
```
Raw Features (100+)
    ↓
Feature Elimination
    ↓
Scaling (StandardScaler)
    ↓
Imputation (fill missing values)
    ↓
Target Encoding (categorical features)
    ↓
Logistic Regression Model
    ↓
OUTPUT: Not-Good Segment Probability
```

**Output for CONT_001:**
```python
NotGood_Segment_Probability = 0.022  # 2.2% chance of default
```

---

## Layer 3: Meta Model (Final Decision)

**What it does:** Combines all predictions intelligently

**Model Type:** Logistic Regression (3 features only)

**Input Features:**
```python
# For CONT_001, we now have 3 predictions:
Feature_1 = Base_Probability         = 0.015  # From Layer 1
Feature_2 = Good_Probability          = 0.009  # From Layer 2a
Feature_3 = NotGood_Probability       = 0.022  # From Layer 2b
```

**Process:**
```
3 Probabilities (from all previous layers)
    ↓
Logistic Regression Model
    ↓
OUTPUT: Final Probability
```

**How Meta Model Works:**

The meta model learns **weights** for each prediction during training:

```python
# Example weights learned during training
weight_base = 0.3
weight_good = 0.5
weight_notgood = 0.2

# Meta model formula (simplified):
Final_Probability = sigmoid(
    weight_base * Base_Probability +
    weight_good * Good_Probability +
    weight_notgood * NotGood_Probability
)
```

**Output for CONT_001:**
```python
Final_Probability = 0.012  # 1.2% chance of default
```

**Key Insight:** The meta model learned to:
- Trust the Good model more (weight=0.5) when applicant seems good
- Use Base model (weight=0.3) as foundation
- Use NotGood model less (weight=0.2) when applicant isn't clearly bad

---

## Final Step: Convert Probability to Score

```python
# Using odds_at_ref=100 for final scoring
Final_Probability = 0.012

# Convert to odds
odds = (1 / 0.012) - 1 = 82.33

# Convert to score
score = ((log(82.33) - log(100)) / log(2)) * 20 + 200
Final_Score = 195
```

**Apply Policy Adjustment:**
```python
# Cap at 250 and apply 5% discount
Adjusted_Score = min(195, 250, 195 * 0.95)
Adjusted_Score = min(195, 250, 185.25)
Adjusted_Score = 185.25
```

---

## 📈 Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICANT: CONT_001                       │
│  AGE:42, INCOME:5000, CC_O_3MWPS_EVER:0, ... (100+ features)│
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │      LAYER 1: BASE MODEL           │
        │  (Logistic Regression + WOE)       │
        │                                    │
        │  Processes: 100+ features          │
        │  → WOE binning                     │
        │  → Feature elimination             │
        │  → Logistic Regression             │
        └────────────────┬───────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │   OUTPUT: Base_Probability = 0.015 │
        │          Base_Score = 210          │
        └────────────────┬───────────────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
┌─────────────────┐            ┌─────────────────┐
│ Score < 180?    │            │ Score >= 200?   │
│ YES → Apply     │            │ YES → Apply     │
│ Good Model      │            │ Not-Good Model  │
└────────┬────────┘            └────────┬────────┘
         │                              │
         ▼                              ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│ LAYER 2a: GOOD MODEL    │   │ LAYER 2b: NOT-GOOD MODEL│
│ (LightGBM)              │   │ (Logistic Regression)   │
│                         │   │                         │
│ Processes: 100+ features│   │ Processes: 100+ features│
│ → Scaling               │   │ → Scaling               │
│ → Imputation            │   │ → Imputation            │
│ → Target Encoding       │   │ → Target Encoding       │
│ → LightGBM              │   │ → Logistic Regression   │
└────────┬────────────────┘   └────────┬────────────────┘
         │                              │
         ▼                              ▼
┌──────────────────────┐      ┌──────────────────────┐
│ Good_Prob = 0.009    │      │ NotGood_Prob = 0.022 │
└──────────┬───────────┘      └───────────┬──────────┘
           │                              │
           └──────────────┬───────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │     LAYER 3: META MODEL             │
        │   (Logistic Regression)             │
        │                                     │
        │   INPUT FEATURES (only 3):          │
        │   - Base_Prob      = 0.015          │
        │   - Good_Prob      = 0.009          │
        │   - NotGood_Prob   = 0.022          │
        │                                     │
        │   Combines with learned weights     │
        └────────────────┬────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │  FINAL OUTPUT:                     │
        │  Final_Probability = 0.012         │
        │  Final_Score = 195                 │
        │  Adjusted_Score = 185.25           │
        └────────────────────────────────────┘
```

---

## 🔍 Why This Architecture?

### Problem with Single Model
```
Single Model: All applicants → One Model → Prediction
```
❌ Can't specialize for different risk levels
❌ Treats good and bad applicants the same way
❌ Misses subtle patterns in specific groups

### Our Solution: 3-Layer Ensemble
```
Layer 1: Initial assessment
Layer 2: Specialized analysis for each risk group
Layer 3: Intelligent combination
```
✅ Specialization for different populations
✅ Better accuracy through ensemble
✅ Meta model learns which layer to trust

---

## 📝 Training vs Scoring

### During Training:

**What gets trained:**
1. **Layer 1:** Base model learns from all 85,955 training samples
2. **Layer 2a:** Good model learns from ~74,000 good segment samples
3. **Layer 2b:** Not-Good model learns from ~24,000 not-good segment samples
4. **Layer 3:** Meta model learns from predictions of all above models

**Output:** 4 saved models (.pkl files)

### During Scoring:

**For each new applicant:**
1. **Apply Layer 1** → Get base probability
2. **Check segmentation** → Determine which Layer 2 models to apply
3. **Apply Layer 2** → Get segment probabilities
4. **Apply Layer 3** → Combine all probabilities → Final prediction

---

## 🎓 Real Example with Numbers

Let's score 3 different applicants:

### Applicant A: Excellent Credit
```
Layer 1 (Base):          Prob = 0.005  →  Score = 220
  ↓
Layer 2 (Good Model):    Prob = 0.003  (very low risk)
Layer 2 (Not-Good Model): Prob = 0.010  (not applied, score too good)
  ↓
Layer 3 (Meta):          Final_Prob = 0.004
  ↓
Final Score: 225 → After policy: 213
```

### Applicant B: Medium Credit (Our CONT_001)
```
Layer 1 (Base):          Prob = 0.015  →  Score = 210
  ↓
Layer 2 (Good Model):    Prob = 0.009  (borderline)
Layer 2 (Not-Good Model): Prob = 0.022  (borderline)
  ↓
Layer 3 (Meta):          Final_Prob = 0.012  (meta chose between 0.009 and 0.022)
  ↓
Final Score: 195 → After policy: 185
```

### Applicant C: Poor Credit
```
Layer 1 (Base):          Prob = 0.045  →  Score = 165
  ↓
Layer 2 (Good Model):    Prob = 0.020  (not applied, score too low)
Layer 2 (Not-Good Model): Prob = 0.055  (high risk confirmed)
  ↓
Layer 3 (Meta):          Final_Prob = 0.050
  ↓
Final Score: 160 → After policy: 152
```

---

## 🔑 Key Takeaways

1. **Layer 1** gives everyone a base score (foundation)
2. **Layer 2** applies specialized models based on Layer 1 score
3. **Layer 3** intelligently combines all predictions
4. **Meta model** learns which layer to trust more for each type of applicant

**The magic:** Meta model automatically learns that:
- For clearly good applicants → trust Good model more
- For clearly bad applicants → trust Not-Good model more
- For borderline applicants → carefully balance all three

This is why it performs better than any single model!

---

## 📊 Code Example: Tracking All Outputs

Here's how to see all intermediate outputs:

```python
import pickle
import pandas as pd

# Load all models
with open('Models/base_model_training.pkl', 'rb') as f:
    base_model = pickle.load(f)
with open('Models/good_model_training.pkl', 'rb') as f:
    good_model = pickle.load(f)
with open('Models/not_good_model_training.pkl', 'rb') as f:
    notgood_model = pickle.load(f)
with open('Models/meta_model_training.pkl', 'rb') as f:
    meta_model = pickle.load(f)

# Load one applicant
applicant = pd.read_csv('Data/sample_score_data.csv').iloc[0:1]
applicant = applicant.set_index('MUQAVILE')
X = applicant.drop(columns=['TARGET'], errors='ignore')

print("="*60)
print("APPLICANT:", X.index[0])
print("="*60)

# Layer 1: Base Model
print("\nLAYER 1: BASE MODEL")
base_pred = base_model.test(X)
base_prob = base_pred['some_key'][0][0]  # Adjust key based on your output
print(f"  Base Probability: {base_prob:.6f}")

# Layer 2: Segment Models
print("\nLAYER 2: SEGMENT MODELS")
good_pred = good_model.test(X)
good_prob = good_pred['some_key'][0][0]
print(f"  Good Model Probability: {good_prob:.6f}")

notgood_pred = notgood_model.test(X)
notgood_prob = notgood_pred['some_key'][0][0]
print(f"  Not-Good Model Probability: {notgood_prob:.6f}")

# Layer 3: Meta Model
print("\nLAYER 3: META MODEL")
meta_input = pd.DataFrame({
    'Base': [base_prob],
    'Good': [good_prob],
    'Not_Good': [notgood_prob]
})
meta_pred = meta_model.test(meta_input)
final_prob = meta_pred['some_key'][0][0]
print(f"  Final Probability: {final_prob:.6f}")

# Convert to score
import numpy as np
odds = (1 / final_prob) - 1
score = ((np.log(odds) - np.log(100)) / np.log(2)) * 20 + 200
print(f"\n  Final Score: {score:.2f}")

adjusted = min(score, 250, score * 0.95)
print(f"  Adjusted Score: {adjusted:.2f}")
```

---

For questions, contact the QNBAnalytics ML Team.

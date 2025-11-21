# Visual Data Flow - How Predictions Combine

## 🎯 Simple Answer to Your Question

**Where do all model outputs go?**
→ All model outputs become **input features** for the next layer!

**Where does the final prediction come from?**
→ The **Meta Model (Layer 3)** produces the final prediction by combining all previous outputs!

---

## 📊 Data Flow Table

| Stage | Input | Model | Output | Goes To |
|-------|-------|-------|--------|---------|
| **Layer 1** | 100+ features | Base LR + WOE | `base_prob` | Layer 3 + Segmentation |
| **Layer 2a** | 100+ features | Good LGBM | `good_prob` | Layer 3 |
| **Layer 2b** | 100+ features | Not-Good LR | `notgood_prob` | Layer 3 |
| **Layer 3** | 3 probs (base, good, notgood) | Meta LR | `final_prob` | **FINAL OUTPUT** |

---

## 🔄 Complete Data Flow with One Applicant

```
APPLICANT: CONT_001
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INPUT DATA (100+ features):
┌─────────────────────────────────────────────────────────┐
│ AGE: 42                                                 │
│ INCOME: 5000                                            │
│ CC_O_3MWPS_EVER: 0                                      │
│ CL_A_CWPS_90D: 1                                        │
│ HL_O_EVERWPS_365DP: 0                                   │
│ ... (95+ more features)                                 │
└─────────────────────────────────────────────────────────┘
                          │
                          │ (All 100+ features)
                          ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER 1: BASE MODEL (Logistic Regression + WOE Binning)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model Input:  100+ features
Model Type:   Logistic Regression with WOE
Training:     Trained on ALL 85,955 applicants

              ┌─────────────────┐
              │  BASE MODEL     │
  100+        │                 │
features ──→  │  WOE + LR       │  ──→  Probability
              │                 │
              └─────────────────┘

OUTPUT: base_prob = 0.015
        base_score = 210

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SPLIT TO BOTH PATHS (score 210 is in range 180-200):

Path A: Score < 180? No, but close → Still apply Good model
Path B: Score >= 200? Yes → Apply Not-Good model

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER 2a: GOOD SEGMENT MODEL (LightGBM)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model Input:  100+ features (SAME as Layer 1 input)
Model Type:   LightGBM
Training:     Trained ONLY on ~74,000 "good" segment applicants

              ┌─────────────────┐
              │  GOOD MODEL     │
  100+        │                 │
features ──→  │  LGBM           │  ──→  Probability
              │  (specialized)  │
              └─────────────────┘

OUTPUT: good_prob = 0.009

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER 2b: NOT-GOOD SEGMENT MODEL (Logistic Regression)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model Input:  100+ features (SAME as Layer 1 input)
Model Type:   Logistic Regression
Training:     Trained ONLY on ~24,000 "not-good" segment applicants

              ┌─────────────────┐
              │ NOT-GOOD MODEL  │
  100+        │                 │
features ──→  │  LR             │  ──→  Probability
              │  (specialized)  │
              └─────────────────┘

OUTPUT: notgood_prob = 0.022

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER 3: META MODEL (Final Combination)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model Input:  ONLY 3 features (probabilities from previous layers!)
              ┌──────────────────────┐
              │ base_prob    = 0.015 │
              │ good_prob    = 0.009 │
              │ notgood_prob = 0.022 │
              └──────────────────────┘

Model Type:   Logistic Regression (only 3 input features)
Training:     Trained on predictions from Layer 1 & 2

              ┌─────────────────────────┐
              │    META MODEL           │
   3          │                         │
predictions ─→│  LR (3 features)        │ ─→ Final Probability
              │  Learns weights:        │
              │  w1=0.3, w2=0.5, w3=0.2 │
              └─────────────────────────┘

OUTPUT: final_prob = 0.012

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL TRANSFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Probability to Score:
  final_prob = 0.012
  odds = (1/0.012) - 1 = 82.33
  raw_score = ((log(82.33) - log(100)) / log(2)) * 20 + 200
  raw_score = 195

Policy Adjustment:
  adjusted_score = min(195, 250, 195 * 0.95)
  adjusted_score = 185.25

FINAL OUTPUT:
┌─────────────────────────────────────────────────────────┐
│ MUQAVILE:     CONT_001                                  │
│ PROBABILITY:  0.012  (1.2% chance of default)           │
│ RAW_SCORE:    195                                       │
│ FINAL_SCORE:  185.25                                    │
│ RISK_LEVEL:   Fair                                      │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 Key Points

### 1. Layers 1 & 2 See the SAME Input
```
Original Features (100+)
         │
    ┌────┼────┬─────────┐
    │    │    │         │
    ▼    ▼    ▼         ▼
  Layer  Layer Layer   Layer
    1     2a    2b      3
```

- **Layer 1**: Sees all 100+ original features
- **Layer 2a**: Sees all 100+ original features (NOT Layer 1 output)
- **Layer 2b**: Sees all 100+ original features (NOT Layer 1 output)
- **Layer 3**: Sees ONLY 3 features (the probabilities from Layers 1, 2a, 2b)

### 2. Layer 3 is Different
```
Layer 1, 2a, 2b Outputs (3 probabilities)
                    │
                    ▼
               Layer 3 (Meta)
                    │
                    ▼
            Final Prediction
```

Layer 3 does NOT see original features, only the predictions!

---

## 📝 Training: What Gets Saved

After training, you have **4 separate models** saved:

```
Models/
├── base_model_training.pkl          ← Layer 1
│   Contains: WOE binning + LR model
│
├── good_model_training.pkl          ← Layer 2a
│   Contains: Scaler + Imputer + Encoder + LGBM model
│
├── not_good_model_training.pkl      ← Layer 2b
│   Contains: Scaler + Imputer + Encoder + LR model
│
└── meta_model_training.pkl          ← Layer 3
    Contains: LR model (only 3 features)
```

---

## 🔄 Scoring: How Models Are Applied

When scoring a new applicant, here's the exact sequence:

```python
# 1. Apply Layer 1 (Base Model)
base_prob = base_model.predict(X)  # X has 100+ features
# Output: base_prob = 0.015

# 2. Apply Layer 2a (Good Model)
good_prob = good_model.predict(X)  # Same X with 100+ features
# Output: good_prob = 0.009

# 3. Apply Layer 2b (Not-Good Model)
notgood_prob = notgood_model.predict(X)  # Same X with 100+ features
# Output: notgood_prob = 0.022

# 4. Create meta features (ONLY 3!)
X_meta = pd.DataFrame({
    'Base': [base_prob],
    'Good': [good_prob],
    'Not_Good': [notgood_prob]
})

# 5. Apply Layer 3 (Meta Model)
final_prob = meta_model.predict(X_meta)  # Only 3 features!
# Output: final_prob = 0.012

# 6. Convert to score
final_score = probability_to_score(final_prob)
# Output: final_score = 185.25
```

---

## 🧮 How Meta Model Combines Predictions

The meta model learns **coefficients** during training:

```python
# Simplified formula (actual is more complex with sigmoid)
final_prob = (
    coef_base    * base_prob +      # e.g., 0.3 * 0.015 = 0.0045
    coef_good    * good_prob +      # e.g., 0.5 * 0.009 = 0.0045
    coef_notgood * notgood_prob     # e.g., 0.2 * 0.022 = 0.0044
) + intercept

# Result: final_prob ≈ 0.012
```

**What the meta model learned:**
- When applicant is clearly **good** (low base_prob):
  - Trust good_prob more (higher weight)

- When applicant is clearly **bad** (high base_prob):
  - Trust notgood_prob more (higher weight)

- When applicant is **borderline**:
  - Balance all three predictions

---

## 📊 Comparison: All Outputs for 3 Applicants

| Applicant | Layer 1 (Base) | Layer 2a (Good) | Layer 2b (NotGood) | Layer 3 (Final) | Score |
|-----------|----------------|-----------------|--------------------|--------------------|-------|
| **A (Excellent)** | 0.005 | 0.003 ✓ | 0.010 | **0.004** ← Trusts Good | 225 |
| **B (Medium)** | 0.015 | 0.009 | 0.022 | **0.012** ← Balances all | 195 |
| **C (Poor)** | 0.045 | 0.020 | 0.055 ✓ | **0.050** ← Trusts NotGood | 160 |

✓ = Which model the meta model trusts most

---

## 🎓 Why This Works Better Than Single Model

### Single Model Problem:
```
All applicants → One Model → Prediction

Good applicant:    Features → Model → 0.005  ← Model tries its best
Medium applicant:  Features → Model → 0.015  ← Model tries its best
Bad applicant:     Features → Model → 0.045  ← Model tries its best
```
❌ Same model must handle all cases
❌ Can't specialize

### 3-Layer Solution:
```
Good applicant:    Features → Layer 1 (0.005) + Layer 2a (0.003) → Meta → 0.004
                                                  ↑ Specialist!

Medium applicant:  Features → Layer 1 (0.015) + Both 2a & 2b → Meta → 0.012
                                                  ↑ Both specialists!

Bad applicant:     Features → Layer 1 (0.045) + Layer 2b (0.055) → Meta → 0.050
                                                  ↑ Specialist!
```
✅ Specialized models for each segment
✅ Meta model intelligently combines

---

## 💡 Summary: Answer to Your Question

**Q: Where do all model outputs go?**

**A:**
1. **Layer 1 output** (base_prob) → Goes to **Layer 3** as Feature #1
2. **Layer 2a output** (good_prob) → Goes to **Layer 3** as Feature #2
3. **Layer 2b output** (notgood_prob) → Goes to **Layer 3** as Feature #3
4. **Layer 3 output** (final_prob) → This IS the final prediction!

**Q: Where does the final model get its output from?**

**A:** The **Meta Model (Layer 3)** creates the final output by:
- Taking 3 inputs: [base_prob, good_prob, notgood_prob]
- Applying learned weights to each
- Producing one final probability
- This becomes your final credit score

**All previous layers feed into Layer 3, and Layer 3 produces the final answer!**

---

For questions, contact the QNBAnalytics ML Team.

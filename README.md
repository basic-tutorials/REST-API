# Bank of Baku Credit Scoring System

A production-ready 3-layer hierarchical ensemble system for credit risk assessment, implementing advanced machine learning techniques with interpretability and stability.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Design Rationale](#design-rationale)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Technical Specifications](#technical-specifications)
- [Model Performance](#model-performance)

---

## 🎯 Overview

This credit scoring system predicts the probability of loan default and converts it to an interpretable credit score (similar to FICO). The system uses a sophisticated 3-layer stacking ensemble approach that combines multiple models to achieve both high accuracy and business interpretability.

### Key Features

- **3-Layer Hierarchical Architecture**: Combines base model, segment-specific models, and meta-learner
- **Risk Segmentation**: Specialized models for different risk profiles
- **WOE Binning**: Monotonic transformations for interpretability
- **Policy Adjustments**: Built-in business rules for risk management
- **Production Ready**: Complete training and scoring pipelines with database integration

### Credit Score Transformation

The system converts default probabilities to credit scores using the standard log-odds transformation:

```
score = ((log(odds) - log(odds_at_ref)) / log(2)) × PDO + REF
```

**Default Parameters:**
- Reference Score (REF): 200
- Odds at Reference: 100:1 (1% default rate)
- Points to Double Odds (PDO): 20

**Example:**
- Score 180 = 2% default probability (50:1 odds)
- Score 200 = 1% default probability (100:1 odds)
- Score 220 = 0.5% default probability (200:1 odds)

---

## 🏗️ Architecture

### High-Level Pipeline Flow

```mermaid
flowchart TD
    A[Raw Applicant Data] --> B[Data Preprocessing]
    B --> C[Layer 1: Base Model]
    C --> D{Risk Segmentation}
    D -->|Low Risk| E[Layer 2a: Good Segment Model]
    D -->|High Risk| F[Layer 2b: Not-Good Segment Model]
    E --> G[Layer 3: Meta Model]
    F --> G
    C --> G
    G --> H[Probability to Score]
    H --> I[Policy Adjustment]
    I --> J[Final Credit Score]

    style C fill:#e1f5ff
    style E fill:#d4edda
    style F fill:#f8d7da
    style G fill:#fff3cd
    style J fill:#d1ecf1
```

### Layer 1: Base Model (Foundation)

```mermaid
flowchart LR
    A[All Applicants] --> B[Feature Elimination]
    B --> C[WOE Binning]
    C --> D[Logistic Regression]
    D --> E[Base Probability]
    E --> F[Risk Score]

    F -->|Score < 180| G[Good Segment]
    F -->|Score >= 200| H[Not-Good Segment]
    F -->|180 ≤ Score < 200| I[Overlap Region]

    style D fill:#e1f5ff
    style G fill:#d4edda
    style H fill:#f8d7da
    style I fill:#fff3cd
```

**Purpose:**
- Provides initial risk assessment for all applicants
- Creates foundation for intelligent segmentation
- Ensures interpretability through WOE binning

**Key Components:**
- **Feature Elimination Pipeline:**
  - Drop features with >99% missing values
  - Remove constant/near-constant features
  - Eliminate low-information features (Gini < 0.05)
  - Remove correlated features (keep higher Gini)

- **WOE (Weight of Evidence) Binning:**
  - Transforms continuous variables into monotonic bins
  - Handles missing values automatically
  - Reduces impact of outliers
  - Makes model interpretable for regulators

- **Model:** Logistic Regression with hyperparameter optimization

### Layer 2: Segment Models (Specialization)

```mermaid
flowchart TD
    subgraph "Good Segment (Low Risk)"
        A1[Good Applicants] --> B1[Feature Elimination]
        B1 --> C1[Scaling]
        C1 --> D1[Imputation]
        D1 --> E1[Target Encoding]
        E1 --> F1[LightGBM Model]
        F1 --> G1[Good Probability]
    end

    subgraph "Not-Good Segment (High Risk)"
        A2[Not-Good Applicants] --> B2[Feature Elimination]
        B2 --> C2[Scaling]
        C2 --> D2[Imputation]
        D2 --> E2[Target Encoding]
        E2 --> F2[Logistic Regression]
        F2 --> G2[Not-Good Probability]
    end

    style F1 fill:#28a745
    style F2 fill:#dc3545
```

**Purpose:**
- Specialized predictions for different risk profiles
- Captures segment-specific patterns
- Improves accuracy through focused modeling

**Why Different Models?**

| Segment | Model | Rationale |
|---------|-------|-----------|
| **Good** (Low Risk) | **LightGBM** | • Captures complex non-linear patterns<br>• Better for rare events (low default rate)<br>• Handles feature interactions automatically |
| **Not-Good** (High Risk) | **Logistic Regression** | • Provides stability for high-risk segment<br>• Interpretable coefficients<br>• Prevents overfitting on noisy data |

**Key Differences from Base:**
- ❌ No WOE binning (tree models handle raw features better)
- ✅ StandardScaler normalization
- ✅ Null imputation (mean for numeric, constant for categorical)
- ✅ Target encoding for categorical variables

### Layer 3: Meta Model (Ensemble)

```mermaid
flowchart LR
    A[Base Probability] --> D[Meta Logistic Regression]
    B[Good Segment Probability] --> D
    C[Not-Good Segment Probability] --> D
    D --> E[Final Probability]
    E --> F[Credit Score]

    style D fill:#fff3cd
    style F fill:#d1ecf1
```

**Purpose:**
- Learns optimal combination of all previous predictions
- Produces calibrated final probabilities
- Acts as intelligent weighted average

**Why Logistic Regression?**
- Simple, interpretable weights for each layer
- No risk of overfitting (only 3 input features)
- Naturally produces calibrated probabilities
- Fast training and inference

**Meta Features:**
1. Base model probability (Layer 1)
2. Good segment probability (Layer 2a)
3. Not-Good segment probability (Layer 2b)

### Complete Training Flow

```mermaid
sequenceDiagram
    participant DB as Database
    participant L1 as Layer 1 (Base)
    participant SEG as Segmentation
    participant L2 as Layer 2 (Segments)
    participant L3 as Layer 3 (Meta)
    participant OUT as Output

    DB->>L1: Load training data
    L1->>L1: WOE binning + LR training
    L1->>SEG: Generate base probabilities
    SEG->>SEG: Split by risk thresholds
    SEG->>L2: Good segment → LGBM
    SEG->>L2: Not-Good segment → LR
    L2->>L2: Train segment models
    L2->>L3: Generate segment probabilities
    L1->>L3: Pass base probabilities
    L3->>L3: Train meta LR
    L3->>OUT: Save all models
    L3->>OUT: Export training scores
```

### Scoring Flow

```mermaid
sequenceDiagram
    participant APP as New Applicant
    participant L1 as Layer 1
    participant L2 as Layer 2
    participant L3 as Layer 3
    participant POL as Policy Engine
    participant OUT as Final Score

    APP->>L1: Raw features
    L1->>L1: Apply WOE binning
    L1->>L1: LR prediction
    L1->>L2: Base probability

    APP->>L2: Raw features (Good model)
    APP->>L2: Raw features (Not-Good model)
    L2->>L2: Apply transformations
    L2->>L3: Good & Not-Good probabilities

    L1->>L3: Base probability
    L3->>L3: Meta LR prediction
    L3->>POL: Final probability

    POL->>POL: Probability → Score
    POL->>POL: Apply cap (250)
    POL->>POL: Apply discount (×0.95)
    POL->>OUT: Final credit score
```

---

## 🤔 Design Rationale

### Why 3-Layer Hierarchical Architecture?

#### Traditional Approach Problems:
- **Single Model:** Cannot capture diverse risk patterns across segments
- **Simple Ensemble:** Treats all applicants equally, ignoring risk profiles
- **Manual Segmentation:** Rigid rules, doesn't learn optimal splits

#### Our Solution Benefits:

1. **Layer 1 (Base Model with WOE):**
   - ✅ Regulatory compliance (interpretable binning)
   - ✅ Handles missing values and outliers
   - ✅ Monotonic relationships (risk increases = score decreases)
   - ✅ Provides intelligent segmentation basis

2. **Layer 2 (Segment Specialization):**
   - ✅ Good segment: LGBM captures subtle patterns in low-risk population
   - ✅ Not-Good segment: LR provides stability for high-risk population
   - ✅ Each model optimized for its segment's characteristics
   - ✅ Better performance than one-size-fits-all

3. **Layer 3 (Meta Learning):**
   - ✅ Learns which layer to trust more for each applicant
   - ✅ Calibrated probabilities (better than simple averaging)
   - ✅ Leverages strengths of all models
   - ✅ Robust to individual model weaknesses

### Why These Segmentation Thresholds?

**Good Segment:** Score < 180 (probability < 0.038)
**Not-Good Segment:** Score ≥ 200 (probability ≥ 0.019)

```mermaid
graph LR
    A[Very Low Risk<br/>Score < 180] --> B[Overlap Zone<br/>180-200]
    B --> C[High Risk<br/>Score >= 200]

    style A fill:#d4edda
    style B fill:#fff3cd
    style C fill:#f8d7da
```

**Rationale:**
- **Overlap Region (180-200):** Applicants appear in both segments
  - Gives both models a chance to contribute
  - Meta model learns which to trust in borderline cases
- **Thresholds from Business Rules:** Align with historical approval policies
- **Uses odds_at_ref=50 for segmentation:** Matches original notebook behavior

### Why WOE Binning in Layer 1 Only?

**Layer 1 (Uses WOE):**
- ✅ Interpretability required for regulatory compliance
- ✅ Linear model (LR) benefits from transformed features
- ✅ Foundation layer needs stability and explainability

**Layer 2 (No WOE):**
- ✅ Tree-based models (LGBM) naturally handle non-linearity
- ✅ Can discover optimal splits without pre-binning
- ✅ Better performance with raw features
- ✅ Target encoding preserves information better for trees

### Why Policy Adjustments?

Business rules applied after modeling:

```python
final_score = min(raw_score, 250, raw_score × 0.95)
```

**Rationale:**
1. **Hard Cap (250):**
   - Prevents overconfidence in model predictions
   - Limits exposure even for "perfect" applicants
   - Accounts for unknown unknowns

2. **Conservative Discount (5%):**
   - Safety margin for model uncertainty
   - Adjusts for economic cycle changes
   - Provides buffer for data drift

### Why Dual odds_at_ref Values?

**Configuration:**
- `ODDS_AT_REFERENCE_SEGMENTATION = 50` (for Layer 2 splits)
- `ODDS_AT_REFERENCE = 100` (for final scoring)

**Rationale:**
- Matches original notebook behavior exactly
- Segmentation uses different scale than final scoring
- Allows independent tuning of segmentation vs. final scores
- Historical business reasons (legacy system compatibility)

---

## 🚀 Installation

### Prerequisites

- Python 3.7+
- Oracle Database access (for data loading)
- 8GB+ RAM recommended for training

### Install Dependencies

```bash
pip install numpy pandas scikit-learn lightgbm xgboost \
    category_encoders feature_engine joblib matplotlib \
    seaborn shap optbinning imbalanced-learn sqlalchemy \
    cx_Oracle openpyxl
```

### Directory Structure

```
scoring_model/
├── credit_scoring_pipeline.py          # Main pipeline script
├── BOB_Scorecard_Unified_Pipeline.ipynb  # Jupyter notebook version
├── QNBAnalytics_ML/                    # Custom ML library
│   ├── binning.py
│   ├── elimination.py
│   ├── skills_api.py
│   └── ...
├── Data/                               # SQL queries and credentials
│   ├── train_data_sql_training.txt
│   ├── test_data_sql_training.txt
│   ├── test_data_sql_scoring_policy_adjustment.txt
│   ├── user                            # DB username
│   └── pass                            # DB password
├── Models/                             # Saved model artifacts
│   ├── base_model_training.pkl
│   ├── good_model_training.pkl
│   ├── not_good_model_training.pkl
│   └── meta_model_training.pkl
└── Output/                             # Scoring results
    ├── TRAINING_SCORES_training.xlsx
    └── SCORING_RESULTS_training.xlsx
```

---

## 📖 Usage

### Command-Line Interface

```bash
# Test configuration (no training/scoring)
python credit_scoring_pipeline.py --mode test

# Train all 3 layers from database
python credit_scoring_pipeline.py --mode train

# Score new applicants (requires trained models)
python credit_scoring_pipeline.py --mode score

# Score without policy adjustments
python credit_scoring_pipeline.py --mode score --no-policy

# Train and score in one run
python credit_scoring_pipeline.py --mode both
```

### Python API

```python
from credit_scoring_pipeline import CreditScoringPipeline

# Initialize pipeline
pipeline = CreditScoringPipeline()

# Training workflow
pipeline.load_training_data()  # Load from database
results = pipeline.train()      # Train all layers
print(results.head())

# Scoring workflow
pipeline.load_models()          # Load saved models
scores = pipeline.score()       # Score new applicants
print(scores.head())

# Custom scoring (with your own data)
import pandas as pd
x_custom = pd.read_csv('applicants.csv')
scores = pipeline.score(x_data=x_custom, apply_policy=True)
```

### Jupyter Notebook

Open `BOB_Scorecard_Unified_Pipeline.ipynb` for interactive analysis:

```python
# Part 1: Training
version = "training"
ref = 200
odds_at_ref_segmentation = 50
odds_at_ref_scoring = 100
points_to_double = 20

# Run training cells...

# Part 2: Scoring
# Run scoring cells...
```

---

## ⚙️ Configuration

### Key Parameters

Edit `Config` class in `credit_scoring_pipeline.py`:

```python
class Config:
    # Score transformation
    REFERENCE_SCORE = 200              # Anchor score
    ODDS_AT_REFERENCE = 100            # Final scoring odds
    ODDS_AT_REFERENCE_SEGMENTATION = 50  # Segmentation odds
    POINTS_TO_DOUBLE = 20              # PDO

    # Risk thresholds
    GOOD_SCORE_THRESHOLD = 180         # Good segment upper bound
    NOT_GOOD_SCORE_THRESHOLD = 200     # Not-good segment lower bound

    # Policy rules
    POLICY_CONSTANT = 250              # Maximum score cap
    POLICY_MULTIPLIER = 0.95           # Conservative discount

    # Model selection
    GOOD_MODEL = 'LGBM'                # Model for good segment
    NOT_GOOD_MODEL = 'Logistic Regression'  # Model for not-good

    # Feature exclusion
    COLS_TO_DROP = [...]               # Features to exclude
```

### Database Configuration

**Security Note:** Current implementation uses plaintext credential files. For production:
- Use environment variables
- Implement secret management (AWS Secrets Manager, Azure Key Vault)
- Use encrypted configuration files

```python
# Current (not recommended for production):
username = pd.read_table('Data/user', header=None)[0][0]
password = pd.read_table('Data/pass', header=None)[0][0]

# Recommended for production:
import os
username = os.environ['DB_USERNAME']
password = os.environ['DB_PASSWORD']
```

---

## 🔧 Technical Specifications

### Model Algorithms

| Layer | Model Type | Optimization | Key Parameters |
|-------|-----------|--------------|----------------|
| **Layer 1** | Logistic Regression | RandomizedSearchCV | C, penalty, solver |
| **Layer 2a** (Good) | LightGBM | RandomizedSearchCV | learning_rate, max_depth, n_estimators |
| **Layer 2b** (Not-Good) | Logistic Regression | RandomizedSearchCV | C, penalty, solver |
| **Layer 3** | Logistic Regression | RandomizedSearchCV | C, penalty, solver |

### Feature Engineering Pipeline

**Layer 1 (Base):**
```
Raw Features → Null Elimination → Constant Elimination →
Low Gini Elimination → Correlation Elimination → WOE Binning →
Logistic Regression
```

**Layer 2 (Segments):**
```
Raw Features → Null Elimination → Constant Elimination →
Low Gini Elimination → Correlation Elimination → Scaling →
Imputation → Target Encoding → Model Training
```

**Layer 3 (Meta):**
```
[Base_Prob, Good_Prob, NotGood_Prob] → Logistic Regression
```

### Feature Elimination Criteria

| Eliminator | Threshold | Rationale |
|------------|-----------|-----------|
| Null Features | >99% missing | No predictive value |
| Constant Features | Zero variance | Cannot separate classes |
| Low Gini Features | Gini < 0.05 | Weak discriminatory power |
| Correlated Features | High correlation | Reduce multicollinearity |

### Performance Metrics

The pipeline tracks multiple metrics:
- **Gini Coefficient:** Discriminatory power
- **AUC-ROC:** Classification performance
- **KS Statistic:** Separation between classes
- **Calibration:** Predicted vs. actual default rates

### Model Artifacts

Saved using `pickle` protocol:
- `base_model_training.pkl`: ~5-20 MB (includes WOE transformers)
- `good_model_training.pkl`: ~10-50 MB (LGBM with trees)
- `not_good_model_training.pkl`: ~5-20 MB (LR with coefficients)
- `meta_model_training.pkl`: ~1-5 MB (simple LR with 3 features)

---

## 📊 Model Performance

### Expected Performance Ranges

| Metric | Layer 1 (Base) | Layer 2 (Segments) | Layer 3 (Meta) |
|--------|----------------|-------------------|----------------|
| **Gini** | 0.40-0.50 | 0.45-0.60 | 0.50-0.65 |
| **AUC** | 0.70-0.75 | 0.73-0.80 | 0.75-0.82 |
| **KS** | 0.35-0.45 | 0.40-0.55 | 0.45-0.60 |

### Score Distribution Guidelines

Typical distribution of final scores:

| Score Range | Risk Level | Expected % | Typical Default Rate |
|-------------|-----------|-----------|---------------------|
| 220-250 | Excellent | 15-20% | <0.5% |
| 200-220 | Good | 25-30% | 0.5-1.5% |
| 180-200 | Fair | 25-30% | 1.5-3% |
| 160-180 | Poor | 15-20% | 3-6% |
| <160 | High Risk | 5-10% | >6% |

---

## 🔍 Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError: feature_engine.variable_manipulation`
```bash
# Solution: Update to compatible version
pip install feature_engine>=1.4.0
```

**Issue:** `ORA-01843: not a valid month` (database error)
```python
# Solution: Check date formats in SQL queries
# Ensure consistent TO_DATE usage
```

**Issue:** Models not found when scoring
```bash
# Solution: Train models first or check paths
python credit_scoring_pipeline.py --mode train
```

**Issue:** Memory error during training
```python
# Solution: Reduce training data size or increase RAM
# Or process in batches
```

---

## 📚 References

### Credit Scoring Methodology
- Anderson, R. (2007). *The Credit Scoring Toolkit*
- Thomas, L., Edelman, D., & Crook, J. (2002). *Credit Scoring and Its Applications*

### Machine Learning Techniques
- Wolpert, D. (1992). "Stacked Generalization"
- Breiman, L. (1996). "Stacking Regressions"
- Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"

### WOE Binning
- Naeem Siddiqi (2006). *Credit Risk Scorecards*
- Optimal Binning Library: https://gnpalencia.org/optbinning/

---

## 👥 Authors

**QNBAnalytics ML Team**
- Bank of Baku
- Version: 0.3.2
- Last Updated: 2024

---

## 📄 License

Internal use only - Bank of Baku proprietary software.

---

## 🚦 Quick Start

```bash
# 1. Clone and setup
cd scoring_model
pip install -r requirements.txt

# 2. Configure database credentials
echo "your_username" > Data/user
echo "your_password" > Data/pass

# 3. Test configuration
python credit_scoring_pipeline.py --mode test

# 4. Train models
python credit_scoring_pipeline.py --mode train

# 5. Score new applicants
python credit_scoring_pipeline.py --mode score

# 6. View results
open Output/SCORING_RESULTS_training.xlsx
```

---

For questions or support, contact the QNBAnalytics ML Team.

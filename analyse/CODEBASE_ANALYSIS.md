# QNBAnalytics ML - Codebase Analysis

> **Bank of Baku Credit Scoring System**
> Version: 0.3.2
> Type: Credit Risk Assessment & Scoring Model

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Technologies & Dependencies](#technologies--dependencies)
5. [Core Modules](#core-modules)
6. [Data Flow & Pipeline](#data-flow--pipeline)
7. [Trained Models](#trained-models)
8. [Configuration](#configuration)
9. [Entry Points](#entry-points)
10. [Performance Metrics](#performance-metrics)
11. [Key Features](#key-features)
12. [Database & Data Sources](#database--data-sources)
13. [Architecture Decisions](#architecture-decisions)

---

## Project Overview

QNBAnalytics ML is a sophisticated **credit risk scoring system** designed for Bank of Baku. The system implements a hierarchical multi-layer machine learning approach for:

- Credit risk assessment
- Applicant scoring
- Policy adjustment
- Risk segmentation

The core innovation is a **3-layer stacked ensemble architecture** that combines base models with segment-specific models to achieve superior discrimination power.

---

## Architecture

### 3-Layer Stacked Ensemble

```
┌─────────────────────────────────────────────────────────────┐
│                      LAYER 3: META MODEL                    │
│              Logistic Regression (Stacking)                 │
│         Combines all predictions → Final Score              │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
┌───────┴───────┐                         ┌────────┴────────┐
│   LAYER 2A    │                         │    LAYER 2B     │
│  Good Segment │                         │ Not-Good Segment│
│   (LightGBM)  │                         │      (LR)       │
│  Score >= 180 │                         │   Score <= 200  │
└───────┬───────┘                         └────────┬────────┘
        │                                          │
        └─────────────────────┬────────────────────┘
                              │
┌─────────────────────────────┴─────────────────────────────┐
│                    LAYER 1: BASE MODEL                     │
│               Logistic Regression (Full Data)              │
│                WOE Binning + Feature Selection             │
└────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────┴─────────────────────────────┐
│                      INPUT DATA                            │
│            Preprocessed Applicant Features                 │
└────────────────────────────────────────────────────────────┘
```

### Layer Descriptions

| Layer | Model | Input | Output | Purpose |
|-------|-------|-------|--------|---------|
| **Layer 1** | Logistic Regression | All applicants | Base probability | Initial risk assessment |
| **Layer 2A** | LightGBM | Good segment (prob < 0.18) | Refined probability | Better discrimination for low-risk |
| **Layer 2B** | Logistic Regression | Not-good segment (prob >= 0.25) | Refined probability | Better discrimination for high-risk |
| **Layer 3** | Logistic Regression | Stacked probabilities | Final score | Optimal combination |

---

## Project Structure

```
scoring_model/
│
├── QNBAnalytics_ML/                    # Core Python Package
│   ├── __init__.py                     # Package initialization
│   ├── skills_api.py                   # Main API & pipeline orchestration
│   ├── classifier.py                   # Model training implementations
│   ├── skills_library.py               # Utility classes for pipeline steps
│   ├── elimination.py                  # Feature selection/elimination
│   ├── preprocessing.py                # Data preprocessing utilities
│   ├── binning.py                      # WOE optimal binning
│   ├── feature_importances.py          # SHAP-based feature importance
│   ├── data.py                         # Data loading & manipulation
│   ├── kpi.py                          # Classification metrics
│   ├── pipelines.py                    # Model persistence
│   ├── helpers.py                      # Logging configuration
│   ├── default_grids.py                # Default hyperparameters
│   └── param_grid_best.py              # Optimized search grids
│
├── Data/                               # Data & Configuration
│   ├── requirements.txt                # Python dependencies
│   ├── sorted_indices.xlsx             # Index ordering for reproducibility
│   └── [training data files]
│
├── Models/                             # Trained Model Artifacts
│   ├── base_model_training.pkl         # Layer 1 model (40 MB)
│   ├── good_model_training.pkl         # Layer 2A model (30 MB)
│   ├── not_good_model_training.pkl     # Layer 2B model (20 MB)
│   ├── meta_model_training.pkl         # Layer 3 model (5 MB)
│   └── binning.pkl                     # WOE binning transformer
│
├── scripts/                            # SQL Scripts
│   ├── DATAMART.sql                    # Main data mart creation
│   ├── target_calculation.sql          # Target variable computation
│   └── test_data_sql_scoring_policy_adjustment.sql
│
├── docs/                               # Documentation
│   ├── Bank of Baku - ML Training.pdf
│   └── Bank of Baku Data Mart Documentation.xlsx
│
├── BOB_Scorecard_Training.ipynb        # Training Entry Point
├── BOB_Scorecard_Scoring_Policy_Adjustment.ipynb  # Scoring Entry Point
│
└── .git/                               # Version Control
```

---

## Technologies & Dependencies

### Machine Learning & Statistics

| Package | Version | Purpose |
|---------|---------|---------|
| LightGBM | 3.3.2 | Gradient boosting (Layer 2A) |
| XGBoost | 1.6.0 | Extreme gradient boosting |
| scikit-learn | 1.1.3 | ML toolkit, Logistic Regression |
| statsmodels | 0.13.2 | Statistical modeling |
| imbalanced-learn | - | Handling imbalanced data |

### Feature Engineering

| Package | Version | Purpose |
|---------|---------|---------|
| optbinning | 0.14.1 | Optimal binning for WOE |
| feature_engine | 1.3.0 | Feature selection & elimination |
| category_encoders | 2.4.0 | Target encoding |

### Model Interpretation

| Package | Version | Purpose |
|---------|---------|---------|
| SHAP | 0.40.0 | Model explainability |
| scikit-optimize | 0.9.0 | Hyperparameter optimization |

### Data & Utilities

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 1.3.5 | Data manipulation |
| numpy | 1.21.6 | Numerical computing |
| SQLAlchemy | 1.4.35 | Database ORM |
| cx-Oracle | 8.3.0 | Oracle connectivity |
| joblib | 1.1.0 | Model serialization |

### Visualization

| Package | Version | Purpose |
|---------|---------|---------|
| matplotlib | 3.5.1 | Static plots |
| seaborn | 0.11.2 | Statistical visualization |
| plotly | 5.10.0 | Interactive charts |

---

## Core Modules

### skills_api.py (53 KB)
**Main API and Pipeline Orchestration**

```python
# Key Classes
ClassifierPipeline    # Main pipeline orchestration
LogisticRegression    # LR model wrapper
RandomForest          # RF model wrapper
XGBoost               # XGB model wrapper
LGBM                  # LightGBM model wrapper
```

Primary interface for:
- Pipeline construction
- Model training
- Cross-validation
- Prediction generation

---

### classifier.py (47 KB)
**Model Training Implementations**

```python
# Key Classes
LogisticRegression    # sklearn LogisticRegression wrapper
RandomForestClassifier # sklearn RF wrapper
XGBClassifier         # XGBoost wrapper
LGBMClassifier        # LightGBM wrapper
```

Features:
- Hyperparameter search (RandomizedSearchCV)
- Cross-validation evaluation
- Model fitting and prediction

---

### skills_library.py (25 KB)
**Utility Classes for Pipeline Steps**

```python
# Key Classes
data_explore          # Automatic variable classification
DropCols              # Column dropping utility
# Various feature elimination helpers
```

Handles:
- Variable type detection (numeric/categorical)
- Data exploration statistics
- Pipeline step utilities

---

### elimination.py (19 KB)
**Feature Selection & Elimination**

```python
# Key Classes
DropConstantFeatures           # Remove constant columns
DropLowGiniFeatures            # Remove low-predictive features
DropCorrelatedLowerGiniFeatures # Remove correlated features
```

Elimination criteria:
- Null threshold: 99% missing
- Gini threshold: 0.05
- Correlation-based (keep higher Gini)

---

### preprocessing.py (8.5 KB)
**Data Preprocessing**

```python
# Key Functions
conversions()    # Data type conversions
impute_null()    # Null value imputation
encode()         # Categorical encoding
scale()          # Feature scaling
```

---

### binning.py (4 KB)
**Optimal Binning for WOE**

```python
# Key Class
opt_binning      # WOE binning using optbinning library
```

Features:
- Monotonic constraints
- Automatic bin optimization
- WOE transformation

---

### kpi.py (3.5 KB)
**Classification Metrics**

```python
# Key Functions
roc_auc()           # ROC-AUC score
gini()              # Gini coefficient
ks_statistic()      # KS statistic
lift_table()        # Lift analysis
confusion_matrix()  # Classification matrix
```

---

### feature_importances.py (6 KB)
**SHAP-based Feature Importance**

```python
# Key Functions
get_feature_importances()   # Calculate SHAP values
plot_feature_importances()  # Visualize importance
```

---

### pipelines.py (5.5 KB)
**Model Persistence**

```python
# Key Class
pipe_manager    # Save/load/register models
```

Handles:
- Pickle serialization
- Model registration
- Pipeline loading

---

## Data Flow & Pipeline

### Training Pipeline

```
┌─────────────────┐
│  1. Data Load   │  Oracle DB → pandas DataFrame
└────────┬────────┘
         ▼
┌─────────────────┐
│  2. Explore     │  Variable classification (numeric/categorical)
└────────┬────────┘
         ▼
┌─────────────────┐
│  3. Eliminate   │  Drop nulls → constants → low Gini → correlated
└────────┬────────┘
         ▼
┌─────────────────┐
│  4. WOE Bin     │  Optimal binning with monotonic constraints
└────────┬────────┘
         ▼
┌─────────────────┐
│  5. Scale       │  StandardScaler on numeric features
└────────┬────────┘
         ▼
┌─────────────────┐
│  6. Encode      │  Target encoding for categoricals
└────────┬────────┘
         ▼
┌─────────────────┐
│  7. Layer 1     │  Base Logistic Regression
└────────┬────────┘
         ▼
┌─────────────────┐
│  8. Segment     │  Split by Layer 1 predictions
└────────┬────────┘
         ▼
┌─────────────────┐
│  9. Layer 2     │  Train segment-specific models
└────────┬────────┘
         ▼
┌─────────────────┐
│  10. Layer 3    │  Meta model on stacked predictions
└────────┬────────┘
         ▼
┌─────────────────┐
│  11. Save       │  Pickle models to /Models/
└─────────────────┘
```

### Scoring Pipeline

```
┌─────────────────┐
│  1. Load Models │  Load 4 pickled pipelines
└────────┬────────┘
         ▼
┌─────────────────┐
│  2. Layer 1     │  Base model prediction
└────────┬────────┘
         ▼
┌─────────────────┐
│  3. Layer 2     │  Segment model predictions
└────────┬────────┘
         ▼
┌─────────────────┐
│  4. Layer 3     │  Meta model final prediction
└────────┬────────┘
         ▼
┌─────────────────┐
│  5. Transform   │  Probability → Credit Score
└────────┬────────┘
         ▼
┌─────────────────┐
│  6. Adjust      │  Apply policy rules
└─────────────────┘
```

### Credit Score Transformation

```python
# Probability to Odds
odds = (1 / probability) - 1

# Odds to Score
score = ((log(odds) - log(odds_at_reference)) / log(2)) * points_to_double + reference_score

# Policy Adjustment
final_score = min(score, policy_constant, score * policy_multiplier)
```

---

## Trained Models

| Model File | Size | Layer | Algorithm | Description |
|------------|------|-------|-----------|-------------|
| `base_model_training.pkl` | 40 MB | 1 | Logistic Regression | Base model on full dataset |
| `good_model_training.pkl` | 30 MB | 2A | LightGBM | Good segment (low risk) |
| `not_good_model_training.pkl` | 20 MB | 2B | Logistic Regression | Not-good segment (high risk) |
| `meta_model_training.pkl` | 5 MB | 3 | Logistic Regression | Stacking meta model |
| `binning.pkl` | 566 KB | - | OptBinning | WOE transformer |

**Total Model Size:** ~95 MB

---

## Configuration

### Hyperparameter Grids

Located in `QNBAnalytics_ML/default_grids.py`:

```python
# Base Model (Logistic Regression)
base_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    ...
}

# Good Segment (LightGBM)
good_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    ...
}

# Not-Good Segment (Logistic Regression)
not_good_grid = { ... }

# Meta Model (Logistic Regression)
meta_grid = { ... }
```

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `MLFLOW_LOGGING` | Enable MLflow tracking | `False` |

---

## Entry Points

### Training

**File:** `BOB_Scorecard_Training.ipynb`

```python
# Import
from QNBAnalytics_ML.skills_api import ClassifierPipeline

# Initialize pipelines
Pipeline_Base = ClassifierPipeline(...)
Pipeline_Good = ClassifierPipeline(...)
Pipeline_Not_Good = ClassifierPipeline(...)
Pipeline_Meta = ClassifierPipeline(...)

# Train sequentially
Pipeline_Base.fit(X_train, y_train)
Pipeline_Good.fit(X_good, y_good)
Pipeline_Not_Good.fit(X_not_good, y_not_good)
Pipeline_Meta.fit(X_meta, y_train)

# Save models
pipe_manager.save(Pipeline_Base, 'Models/base_model_training.pkl')
```

### Scoring

**File:** `BOB_Scorecard_Scoring_Policy_Adjustment.ipynb`

```python
# Load models
base_model = pipe_manager.load('Models/base_model_training.pkl')
good_model = pipe_manager.load('Models/good_model_training.pkl')
not_good_model = pipe_manager.load('Models/not_good_model_training.pkl')
meta_model = pipe_manager.load('Models/meta_model_training.pkl')

# Sequential prediction
base_prob = base_model.predict_proba(X)
good_prob = good_model.predict_proba(X_good)
not_good_prob = not_good_model.predict_proba(X_not_good)
final_prob = meta_model.predict_proba(X_meta)

# Transform to credit score
score = probability_to_score(final_prob)
```

---

## Performance Metrics

### Target Metrics

| Metric | Target Range | Description |
|--------|--------------|-------------|
| **Gini Coefficient** | 0.56 - 0.61 | Primary discrimination metric |
| **ROC-AUC** | 0.78 - 0.80 | Model discrimination ability |
| **KS Statistic** | 0.40 - 0.50 | Maximum separation |

### Tracked Metrics

```python
# Per-model metrics
- Train Gini
- CV Gini (cross-validation)
- Validation Gini
- Test Gini

# Analysis metrics
- Lift tables (decile analysis)
- Gains charts
- Confusion matrix
- Feature importance (SHAP)
```

### Overfitting Monitoring

```
If (Train Gini - Test Gini) > 0.05:
    → Possible overfitting, review regularization
```

---

## Key Features

### 1. Variable Injection Decorator
Dynamically manages feature columns during pipeline transformation:
```python
@inject_variables
def transform(self, X):
    # Automatically tracks column changes
    ...
```

### 2. MLflow Integration
Optional experiment tracking:
```python
if os.environ.get('MLFLOW_LOGGING'):
    mlflow.log_metric('gini', gini_score)
    mlflow.log_params(best_params)
```

### 3. Comprehensive Logging
```python
from QNBAnalytics_ML.helpers import getLogger
logger = getLogger(__name__)
logger.info("Training completed")
```

### 4. Policy Adjustment Logic
Business rule-based score adjustment:
```python
final_score = min(
    calculated_score,
    policy_constant,
    calculated_score * policy_multiplier
)
```

### 5. WOE Binning
Credit industry-standard feature transformation:
- Monotonic constraints
- Automatic optimal bins
- Weight of Evidence calculation

---

## Database & Data Sources

### Connection

```python
from sqlalchemy import create_engine

# Oracle connection
engine = create_engine(
    'oracle+cx_oracle://user:pass@host:port/service'
)

# Chunked loading for memory efficiency
for chunk in pd.read_sql(query, engine, chunksize=25000):
    process(chunk)
```

### SQL Scripts

| Script | Size | Purpose |
|--------|------|---------|
| `DATAMART.sql` | 630 KB | Main data mart creation |
| `target_calculation.sql` | - | Target variable computation |
| `test_data_sql_scoring_policy_adjustment.sql` | - | Scoring data extraction |

---

## Architecture Decisions

### Design Patterns

| Pattern | Implementation | Rationale |
|---------|----------------|-----------|
| **Pipeline Pattern** | sklearn fit/transform | Consistent API, reproducibility |
| **Composition** | Skills library | Flexible component assembly |
| **Decorator** | Variable injection | Dynamic column tracking |
| **Factory** | Classifier creation | Uniform model instantiation |

### Technical Decisions

1. **Pickle Serialization**
   - Fast save/load
   - Preserves full pipeline state
   - sklearn compatibility

2. **WOE Binning**
   - Industry standard for credit
   - Handles non-linear relationships
   - Monotonic constraints for interpretability

3. **Target Encoding**
   - Handles high-cardinality categoricals
   - Better than one-hot for tree models

4. **Hierarchical Ensemble**
   - Segment-specific optimization
   - Better discrimination in tails
   - Interpretable layer contributions

5. **Cross-Validation**
   - Built into all training
   - Monitors overfitting
   - Robust hyperparameter selection

### Code Quality

- Clear module separation
- Comprehensive docstrings
- Type hints (partial)
- Logging throughout
- Error handling

---

## Quick Reference

### Common Operations

```python
# Load a model
from QNBAnalytics_ML.pipelines import pipe_manager
model = pipe_manager.load('Models/base_model_training.pkl')

# Make predictions
probabilities = model.predict_proba(X)[:, 1]

# Get feature importance
from QNBAnalytics_ML.feature_importances import get_feature_importances
importance = get_feature_importances(model, X)

# Calculate metrics
from QNBAnalytics_ML.kpi import gini, roc_auc
print(f"Gini: {gini(y_true, y_pred)}")
print(f"AUC: {roc_auc(y_true, y_pred)}")
```

### File Locations

| Need | Location |
|------|----------|
| Train models | `BOB_Scorecard_Training.ipynb` |
| Score applicants | `BOB_Scorecard_Scoring_Policy_Adjustment.ipynb` |
| Modify hyperparameters | `QNBAnalytics_ML/default_grids.py` |
| Add preprocessing steps | `QNBAnalytics_ML/preprocessing.py` |
| Change elimination rules | `QNBAnalytics_ML/elimination.py` |
| Update metrics | `QNBAnalytics_ML/kpi.py` |

---

## Version History

| Version | Changes |
|---------|---------|
| 0.3.2 | Current version |
| - | Initial development |

---

## Contact & Support

For questions about this codebase, refer to:
- `docs/Bank of Baku - ML Training.pdf`
- `docs/Bank of Baku Data Mart Documentation.xlsx`

---

*Documentation generated: 2025-11-18*

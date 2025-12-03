# Changelog

All notable changes to the Bank of Baku Credit Scoring Model project.

## [2.0.0] - 2025-11-25

### Added
- **New Features for Training**: Added 5 new predictive features from updated data
  - `MAAS` (salary) - numeric feature, expected top 5 importance
  - `WORKGROUP` (employment type) - categorical feature
  - `QRUP` (organization type) - categorical feature
  - `SAHƏLƏR` (business sector) - categorical feature
  - `SEKTOR` (specific sector) - categorical feature

- **WOE Binning Implementation**: Replaced optbinning with category_encoders.WOEEncoder
  - File: `QNBAnalytics_ML/binning.py`
  - Uses `category_encoders.WOEEncoder` for WOE encoding
  - Compatible API with previous optbinning implementation
  - Works with Python 3.8.16 (no C compiler needed)
  - Includes regularization parameters: randomized=True, sigma=0.05, regularization=1.0

- **Documentation**: Comprehensive retraining guides
  - `docs/QUICK_START.md` - Quick reference for retraining
  - `docs/RETRAINING_GUIDE.md` - Detailed step-by-step guide
  - `docs/CATEGORICAL_ENCODING_GUIDE.md` - Deep dive into TargetEncoder
  - `docs/README_RETRAINING.md` - Overview and summary
  - `docs/explore_new_features.py` - Data exploration script

### Changed
- **Binning Library**: Migrated from `optbinning>=0.17.0` to `category_encoders.woe.WOEEncoder`
  - Reason: optbinning had compatibility issues with Python 3.8.16 on Windows (requires C compiler)
  - Solution: Use category_encoders which is already in dependencies for TargetEncoder

- **Data Processing**: Updated feature list
  - Added 7 ID columns to drop list: ID, FIN, MKR_ID, MKR_DATE, IDS, SCR_CONSTANT, SCR_MULTIPLIER
  - Removed these from training pipeline to avoid data leakage

- **Pipeline Order**: Reordered preprocessing steps in `BOB_Scorecard_Training.ipynb`
  - Old order: binning → imputation → encoding
  - New order: imputation → encoding → binning
  - Reason: Categorical encoding must happen before WOE binning

### Removed
- **Deprecated Binning Files**:
  - `QNBAnalytics_ML/binning_custom.py` - Custom sklearn implementation (superseded)
  - `QNBAnalytics_ML/binning_new.py` - Intermediate version (not needed)
  - `QNBAnalytics_ML/binning_woe.py` - Source file (now copied to binning.py)

### Fixed
- **Python 3.8.16 Compatibility**: Resolved scipy/optbinning installation errors
- **Missing Value Handling**: Enabled null imputation for new MAAS feature
- **Categorical Encoding**: Enabled TargetEncoder for new categorical features

### Technical Details

**Requirements Update**:
```python
# Old (removed):
# optbinning>=0.17.0

# New (using):
category_encoders>=2.3.0  # Already installed for TargetEncoder
```

**Binning Implementation**:
```python
from category_encoders.woe import WOEEncoder

self.woe_encoder = WOEEncoder(
    cols=all_variables,
    randomized=True,      # Add regularization
    sigma=0.05,           # Small noise for stability
    regularization=1.0,   # Prevent overfitting
    random_state=42
)
```

**Expected Performance**:
- Base Model Gini: 0.563 → 0.58-0.61 (+0.02 to +0.05)
- Meta Model Gini: 0.575 → 0.59-0.62 (+0.02 to +0.05)
- Training time: 13 min → 15-20 min (+2-7 min)

### Backup Files
- `QNBAnalytics_ML/binning_OLD_BACKUP.py` - Original optbinning implementation (preserved for reference)

### Migration Notes

**For Users Retraining the Model**:
1. Update `cols_to_drop` in notebook Cell 2 (add 7 ID columns)
2. Enable `apply_null_imputation = True` and `apply_categorical_encoding = True` in Cell 10
3. Reorder pipeline in Cell 11: imputation → encoding → binning
4. Backup existing models: `mkdir Models/backup && cp Models/*.pkl Models/backup/`
5. Delete old binning: `rm Models/binning.pkl`
6. Run notebook: `jupyter notebook BOB_Scorecard_Training.ipynb`

**Verification**:
- Check encoding logs: "Encoding started..." → "Encoding completed."
- Verify new features in `Output/BINNING_TABLE_BASE_training.xlsx`
- Confirm MAAS in top 10 of `Output/FEATURE_IMPORTANCES_BASE_training.xlsx`
- Validate Gini improvement: 0.563 → 0.58+

---

## [1.0.0] - Previous Version

### Initial Implementation
- 3-layer hierarchical ensemble architecture
- Layer 1: Logistic Regression with WOE binning (optbinning)
- Layer 2: LGBM + Logistic Regression segment models
- Layer 3: Meta model combining predictions
- Original feature set: 2,738 credit bureau features
- Base Gini: 0.563 (test set)
- Training time: ~13 minutes

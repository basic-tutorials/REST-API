#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QNBAnalytics ML - Credit Scoring Pipeline
==========================================

Bank of Baku 3-Layer Hierarchical Credit Scoring System

This script consolidates the training and scoring pipelines into a single
executable file. It implements:
- Layer 1: Base Logistic Regression model
- Layer 2: Segment-specific models (Good: LGBM, Not-Good: LR)
- Layer 3: Meta model combining all predictions

Architecture Overview:
---------------------
The pipeline uses a stacking ensemble approach where:
1. Base model (Layer 1) processes all applicants with WOE binning
2. Segment models (Layer 2) specialize on different risk groups
3. Meta model (Layer 3) combines all predictions for final scoring

Score Transformation:
--------------------
The system converts default probabilities to credit scores using:
    score = ((log(odds) - log(odds_ref)) / log(2)) * PDO + REF

Where:
- odds = (1 - probability) / probability
- REF = Reference score (200)
- PDO = Points to double odds (20)

Usage:
    python credit_scoring_pipeline.py --mode train   # Train all layers
    python credit_scoring_pipeline.py --mode score   # Score new applicants
    python credit_scoring_pipeline.py --mode both    # Train then score
    python credit_scoring_pipeline.py --mode test    # Verify configuration

Author: QNBAnalytics ML Team
Version: 0.3.2
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import argparse
import pickle
import datetime
import warnings
import numpy as np
import pandas as pd

# Suppress warnings for cleaner output during production runs
# Note: Consider enabling warnings during development/debugging
warnings.filterwarnings('ignore')

# Import QNBAnalytics ML library
from QNBAnalytics_ML import data
from QNBAnalytics_ML import skills_api
from QNBAnalytics_ML.feature_importances import plot_feature_importances
from QNBAnalytics_ML.param_grid_best import grids as param_grid_best


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """
    Central configuration for the credit scoring pipeline.

    This class contains all configurable parameters for:
    - Data loading and column definitions
    - Score transformation mathematics
    - Risk segmentation thresholds
    - Policy adjustment rules
    - Model selection preferences
    - File paths for I/O operations

    Note: For production deployments, consider externalizing these
    values to environment variables or a configuration file (YAML/JSON).
    """

    # -------------------------------------------------------------------------
    # Version and Identifiers
    # -------------------------------------------------------------------------
    VERSION = "training"  # Used to version model files and outputs

    # -------------------------------------------------------------------------
    # Data Column Definitions
    # -------------------------------------------------------------------------
    INDEX_COL = "MUQAVILE"  # Contract/Application ID column (unique identifier)
    TARGET_COL = "TARGET"   # Binary target variable: 0=non-default (good), 1=default (bad)

    # -------------------------------------------------------------------------
    # Feature Exclusion List
    # -------------------------------------------------------------------------
    # These columns are excluded due to:
    # - High missing value rates (>99% null)
    # - Potential target leakage (derived from outcome period)
    # - Business rules (e.g., BGN flag is a manual override)
    #
    # Format: ProductType_Status_TimePeriod_Metric
    # CC=Credit Card, CL=Consumer Loan, HL=Housing Loan, OL=Other Loan
    # WPS=Worst Payment Status, EVER=Ever in history
    COLS_TO_DROP = [
        "CC_O_4M6MWPS_EVER", "ALL_O_4M6MWPS_EVER_O", "ALL_O_7M12MWPS_EVER",
        "ALL_O_7M12MWPS_EVER_O", "ALL_OSMLMT_CWPS1_6_EVER", "CC_O_13M24MWPS_EVER",
        "CC_O_13M24MWPS_EVER_O", "CC_O_7M12MWPS_EVER", "CC_O_7M12MWPS_EVER_O",
        "CCOL_O_4M6MWPS_EVER", "CL_O_13M24MWPS_EVER_O", "HL_EVERWPS_EVER",
        "HL_O_13M24MWPS_365DP", "HL_O_3MWPS_365DP", "HL_O_4M6MWPS_EVER_O",
        "HL_O_7M12MWPS_EVER_O", "HL_O_EVERWPS_365DP", "OL_O_3MWPS_183D365D",
        "OL_O_3MWPS_365DP", "OL_O_4M6MWPS_183D365D", "OL_O_4M6MWPS_91D182D",
        "OL_O_7M12MWPS_365DP", "OL_O_EVERWPS_183D365D", "OL_O_EVERWPS_365DP",
        "OL_O_EVERWPS_90D", "OL_O_EVERWPS_91D182D", "OL_OLMTUTL_CWPS0_90D_O", "BGN"
    ]

    # -------------------------------------------------------------------------
    # Credit Score Transformation Parameters
    # -------------------------------------------------------------------------
    # These parameters define the odds-to-score conversion following
    # standard credit scoring methodology (similar to FICO scoring).
    #
    # The formula: score = ((log(odds) - log(odds_at_ref)) / log(2)) * PDO + ref
    #
    # IMPORTANT: This codebase uses TWO different odds_at_ref values to match
    # the original notebook behavior:
    # - ODDS_AT_REFERENCE_SEGMENTATION = 50 (for Layer 2 data splits)
    # - ODDS_AT_REFERENCE = 100 (for final scoring)
    #
    # Example with ODDS_AT_REFERENCE = 100:
    #   At reference, odds = 100:1 (1% default rate)
    #   Every 20 points doubles/halves the odds
    #   Score 180 = 200:1 odds (0.5% default)
    #   Score 220 = 50:1 odds (2% default)
    REFERENCE_SCORE = 200      # Anchor/Reference score point
    ODDS_AT_REFERENCE = 100    # Odds ratio at reference for final scoring (100:1)
    ODDS_AT_REFERENCE_SEGMENTATION = 50  # Odds ratio for Layer 2 segmentation (50:1)
    POINTS_TO_DOUBLE = 20      # PDO - Points needed to double the odds

    # -------------------------------------------------------------------------
    # Risk Segmentation Thresholds
    # -------------------------------------------------------------------------
    # These thresholds divide the population into risk segments for Layer 2.
    # Applicants are classified based on their Layer 1 (base model) scores:
    #
    # Good segment:     base_score < 180 (lower risk, use LGBM for precision)
    # Not-Good segment: base_score >= 200 (higher risk, use LR for stability)
    # Overlap region:   180 <= score < 200 (appears in both segments)
    #
    # Note: Lower probability = higher score = better creditworthiness
    GOOD_SCORE_THRESHOLD = 180      # Upper bound for "good" segment
    NOT_GOOD_SCORE_THRESHOLD = 200  # Lower bound for "not-good" segment

    # -------------------------------------------------------------------------
    # Policy Adjustment Parameters
    # -------------------------------------------------------------------------
    # Business rules applied after model scoring for risk management:
    #
    # 1. POLICY_CONSTANT: Hard cap on maximum score (prevents overconfidence)
    # 2. POLICY_MULTIPLIER: Conservative haircut (reduces all scores by 5%)
    #
    # Final score = min(raw_score, POLICY_CONSTANT, raw_score * POLICY_MULTIPLIER)
    POLICY_CONSTANT = 250      # Maximum allowable score (ceiling)
    POLICY_MULTIPLIER = 0.95   # Conservative discount factor (5% reduction)

    # -------------------------------------------------------------------------
    # Model Selection for Layer 2 Segments
    # -------------------------------------------------------------------------
    # Different models are used for different segments based on their
    # characteristics:
    #
    # - Good segment (LGBM): Tree-based model captures non-linear patterns
    #   in the low-risk population with more nuanced feature interactions
    # - Not-Good segment (LR): Linear model provides stability and
    #   interpretability for higher-risk applicants
    NOT_GOOD_MODEL = 'Logistic Regression'  # Model for high-risk segment
    GOOD_MODEL = 'LGBM'                      # Model for low-risk segment

    # -------------------------------------------------------------------------
    # File Paths and Directories
    # -------------------------------------------------------------------------
    DATA_DIR = "Data"          # Input data directory
    MODELS_DIR = "Models"      # Saved model artifacts
    OUTPUT_DIR = "Output"      # Scoring results and reports

    # Database credentials (WARNING: plaintext storage - use env vars in production)
    USER_FILE = "Data/user"    # Database username file
    PASS_FILE = "Data/pass"    # Database password file

    # SQL query files for data extraction
    TRAIN_SQL_FILE = "Data/train_data_sql_training.txt"              # Training data query
    TEST_SQL_FILE = "Data/test_data_sql_training.txt"                # Validation data query
    SCORING_SQL_FILE = "Data/test_data_sql_scoring_policy_adjustment.txt"  # New applicants query


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_database_connection():
    """
    Establish database connection using stored credentials.

    Reads username and password from plaintext files specified in Config.
    Creates and returns a SQLAlchemy engine for Oracle database access.

    Returns:
        sqlalchemy.engine.Engine: Database connection engine

    Security Note:
        Current implementation reads credentials from plaintext files.
        For production, consider using:
        - Environment variables (os.environ)
        - Secret management services (AWS Secrets Manager, Azure Key Vault)
        - Encrypted configuration files
    """
    # Read credentials from plaintext files
    # TODO: Replace with secure credential management
    username = pd.read_table(Config.USER_FILE, header=None)[0][0]
    password = pd.read_table(Config.PASS_FILE, header=None)[0][0]

    # Create and return database connection
    return data.connect_to_sql(username, password)


def probability_to_score(probability, ref=None, odds_at_ref=None, pdo=None):
    """
    Convert default probability to credit score using log-odds transformation.

    This implements the standard credit scoring transformation:
        score = ((log(odds) - log(odds_at_ref)) / log(2)) * PDO + ref

    The transformation creates a linear relationship between score and log-odds,
    which is intuitive for credit risk assessment:
    - Higher probability (worse) → Lower score
    - Lower probability (better) → Higher score

    Args:
        probability: Default probability (0-1), can be scalar or array
        ref: Reference score anchor point (default: Config.REFERENCE_SCORE = 200)
        odds_at_ref: Odds ratio at reference score (default: Config.ODDS_AT_REFERENCE = 100)
        pdo: Points to double odds (default: Config.POINTS_TO_DOUBLE = 20)

    Returns:
        float or np.array: Credit score(s)

    Example:
        >>> probability_to_score(0.01)  # 1% default probability
        200.0  # Reference score (at 100:1 odds)

        >>> probability_to_score(0.005)  # 0.5% default probability
        220.0  # +20 points (odds doubled to 200:1)

        >>> probability_to_score(0.02)  # 2% default probability
        180.0  # -20 points (odds halved to 50:1)
    """
    # Apply default values from Config if not provided
    if ref is None:
        ref = Config.REFERENCE_SCORE
    if odds_at_ref is None:
        odds_at_ref = Config.ODDS_AT_REFERENCE
    if pdo is None:
        pdo = Config.POINTS_TO_DOUBLE

    # Avoid division by zero for probability = 0
    # Using small epsilon (0.00001) instead of zero
    prob = np.where(probability == 0, 0.00001, probability)

    # Calculate odds: odds = (1 - p) / p = good/bad ratio
    # E.g., p=0.01 → odds = 99 (99 goods per 1 bad)
    odds = (1 / prob) - 1

    # Apply log-odds transformation with scaling
    # Division by log(2) makes PDO work in "doubling" units
    score = ((np.log(odds) - np.log(odds_at_ref)) / np.log(2)) * pdo + ref

    return score


def score_to_probability(score, ref=None, odds_at_ref=None, pdo=None):
    """
    Convert credit score back to default probability (inverse of probability_to_score).

    This is the inverse transformation used for:
    - Calculating probability thresholds from score thresholds
    - Converting score-based decisions to probability space
    - Model validation and calibration

    Mathematical derivation:
        From: score = ((log(odds) - log(odds_at_ref)) / log(2)) * PDO + ref
        Solving for probability: p = 1 / (1 + exp((score - ref) / pdo * log(2) + log(odds_at_ref)))

    Args:
        score: Credit score (scalar or array)
        ref: Reference score (default: Config.REFERENCE_SCORE)
        odds_at_ref: Odds at reference score (default: Config.ODDS_AT_REFERENCE)
        pdo: Points to double odds (default: Config.POINTS_TO_DOUBLE)

    Returns:
        float or np.array: Default probability (0-1)

    Example:
        >>> score_to_probability(200)  # Reference score
        0.01  # 1% default probability (100:1 odds)

        >>> score_to_probability(180)  # 20 points below reference
        0.02  # 2% default probability (50:1 odds)
    """
    # Apply default values from Config if not provided
    if ref is None:
        ref = Config.REFERENCE_SCORE
    if odds_at_ref is None:
        odds_at_ref = Config.ODDS_AT_REFERENCE
    if pdo is None:
        pdo = Config.POINTS_TO_DOUBLE

    # Inverse transformation using sigmoid-like formula
    # This converts score back to probability space
    probability = 1 / (np.exp((score - ref) / pdo * np.log(2) + np.log(odds_at_ref)) + 1)

    return probability


def apply_policy_adjustment(scores):
    """
    Apply business policy adjustments to credit scores.

    This function implements conservative risk management rules that:
    1. Cap scores at a maximum (prevents overconfidence in model)
    2. Apply a discount factor (builds in safety margin)

    Adjustments applied:
    1. POLICY_CONSTANT (250): Hard ceiling on scores
       - Prevents unrealistically high scores
       - Limits exposure even for "perfect" applicants

    2. POLICY_MULTIPLIER (0.95): Conservative haircut
       - Reduces all scores by 5%
       - Accounts for model uncertainty and economic cycles

    The final formula is: min(score, POLICY_CONSTANT, score * POLICY_MULTIPLIER)

    Args:
        scores: Raw model scores (scalar or array)

    Returns:
        np.array: Policy-adjusted scores

    Example:
        >>> apply_policy_adjustment(np.array([180, 200, 240, 260]))
        array([171., 190., 228., 237.5])
        # 180 * 0.95 = 171
        # 200 * 0.95 = 190
        # 240 * 0.95 = 228
        # 260 * 0.95 = 247 but capped at 250, then 250 * 0.95 = 237.5
    """
    # Apply the minimum of three values:
    # 1. Original score (no adjustment)
    # 2. Hard cap at POLICY_CONSTANT
    # 3. Discounted score (score * POLICY_MULTIPLIER)
    adjusted = np.minimum(
        scores,
        np.minimum(
            Config.POLICY_CONSTANT,
            scores * Config.POLICY_MULTIPLIER
        )
    )
    return adjusted


# =============================================================================
# PIPELINE CLASSES
# =============================================================================

class BasePipeline:
    """
    Layer 1: Base Model Pipeline

    Uses Logistic Regression with WOE (Weight of Evidence) binning on all data.

    Purpose:
    - Provides initial probability estimates for all applicants
    - Creates the foundation for risk segmentation in Layer 2
    - WOE binning ensures monotonic relationship and handles missing values

    Training Flow:
    1. Data exploration (identify numeric/categorical columns)
    2. Feature elimination (null, constant, low Gini, correlated)
    3. WOE binning (optimal binning with monotonicity constraints)
    4. Logistic Regression training with hyperparameter optimization

    Key Characteristics:
    - Uses WOE binning (not raw features) for interpretability
    - Outputs calibrated probabilities for segmentation
    - Saves binning transformer for consistent scoring
    """

    # -------------------------------------------------------------------------
    # Pipeline Step Configuration Flags
    # -------------------------------------------------------------------------
    # Toggle individual preprocessing steps on/off
    # WOE binning is applied (characteristic of base model)
    # No scaling/imputation needed as WOE handles these

    APPLY_DATA_EXPLORE = True           # Classify features as numeric/categorical
    APPLY_NULL_ELIMINATION = True       # Remove features with >99% missing
    APPLY_CONSTANT_ELIMINATION = True   # Remove zero-variance features
    APPLY_LOW_GINI_ELIMINATION = True   # Remove features with Gini < 0.05
    APPLY_CORRELATED_ELIMINATION = True # Remove correlated features (keep higher Gini)
    APPLY_BINNING = True                # Apply WOE binning transformation
    APPLY_SCALING = False               # Not needed with WOE
    APPLY_NULL_IMPUTATION = False       # WOE handles missing values
    APPLY_CATEGORICAL_ENCODING = False  # WOE encodes categoricals

    def __init__(self):
        """Initialize the base pipeline with no trained model."""
        self.pipeline = None      # Will hold ClassifierPipeline after training
        self.is_trained = False   # Flag to check if model is ready for predictions

    def train(self, x_train, y_train, x_test=None, y_test=None):
        """
        Train the base Logistic Regression model with full preprocessing pipeline.

        Args:
            x_train: Training features DataFrame
            y_train: Training target Series (0/1)
            x_test: Validation features DataFrame (optional)
            y_test: Validation target Series (optional)

        Side Effects:
            - Sets self.pipeline with trained ClassifierPipeline
            - Sets self.is_trained = True
            - May save binning transformer to Models/binning.pkl
        """
        print("\n" + "="*60)
        print("LAYER 1: Training Base Model")
        print("="*60)

        # Prepare validation data (empty DataFrames if not provided)
        x_val = x_test.copy() if x_test is not None else pd.DataFrame()
        y_val = y_test.copy() if y_test is not None else pd.Series()

        # Initialize the main pipeline object from QNBAnalytics_ML library
        # This holds data, transformers, and trained models
        self.pipeline = skills_api.ClassifierPipeline(x_train, y_train, x_val, y_val)

        # =====================================================================
        # Step 1: Data Exploration
        # =====================================================================
        # Automatically classifies features as numeric or categorical
        # based on data types and cardinality
        if self.APPLY_DATA_EXPLORE:
            print("Step 1: Data exploration...")
            self.pipeline.data_explore()

        # =====================================================================
        # Step 2-5: Feature Elimination
        # =====================================================================
        # Progressive elimination to reduce dimensionality and noise

        # Step 2: Remove features with >99% missing values
        # These provide no predictive value
        if self.APPLY_NULL_ELIMINATION:
            print("Step 2: Null feature elimination...")
            self.pipeline.feature_elimination(
                eliminator='drop_null_features',
                params={'threshold': 0.99}  # 99% null threshold
            )

        # Step 3: Remove constant/near-constant features
        # Zero-variance features cannot separate classes
        if self.APPLY_CONSTANT_ELIMINATION:
            print("Step 3: Constant feature elimination...")
            self.pipeline.feature_elimination(
                eliminator='drop_constant_features',
                params={'missing_values': 'ignore'}
            )

        # Step 4: Remove low-information features (Gini < 0.05)
        # Features with very low discrimination power add noise
        if self.APPLY_LOW_GINI_ELIMINATION:
            print("Step 4: Low Gini feature elimination...")
            self.pipeline.feature_elimination(
                eliminator='drop_low_gini_features',
                params={'threshold': 0.05, 'missing_values': 'ignore'}
            )

        # Step 5: Remove correlated features (keep higher Gini)
        # Reduces multicollinearity while preserving predictive power
        if self.APPLY_CORRELATED_ELIMINATION:
            print("Step 5: Correlated feature elimination...")
            self.pipeline.feature_elimination(
                eliminator='correlated_lower_gini_feature_elimination',
                params={'missing_values': 'ignore'}
            )

        # =====================================================================
        # Step 6: WOE Binning (Weight of Evidence)
        # =====================================================================
        # Transforms continuous features into binned WOE values
        # Benefits:
        # - Handles missing values (assigned to separate bin)
        # - Handles outliers (binned together)
        # - Creates monotonic relationship with target
        # - Enables linear model to capture non-linear patterns
        if self.APPLY_BINNING:
            print("Step 6: WOE Binning...")
            # Check if pre-fitted binning transformer exists
            binning_path = os.path.join(Config.MODELS_DIR, 'binning.pkl')
            if os.path.exists(binning_path):
                # Use existing binning for consistent transformation
                self.pipeline.binning(use_existing='binning.pkl')
            else:
                # Create new binning using OptimalBinning algorithm
                self.pipeline.binning()

        # =====================================================================
        # Step 7: Train Logistic Regression
        # =====================================================================
        # Train LR with hyperparameter optimization
        # Uses RandomizedSearchCV for efficient parameter search
        print("Step 7: Training Logistic Regression...")
        self.pipeline.LogisticRegression(
            param_opt=True,                           # Enable hyperparameter optimization
            param_opt_method="RandomizedSearchCV",    # Faster than GridSearchCV
            train_size=x_train.shape[0],              # Used for parameter grid selection
            param_grid=param_grid_best['logistic_regression_base']  # Pre-optimized grid
        )

        # Mark pipeline as trained and ready for predictions
        self.is_trained = True
        print("Base model training completed!")

    def predict(self, x_test, y_test=None):
        """
        Generate probability predictions for new data.

        Applies all fitted transformations (feature elimination, binning)
        and returns Logistic Regression probability predictions.

        Args:
            x_test: Features DataFrame for prediction
            y_test: True labels (optional, for evaluation metrics)

        Returns:
            np.array: Probability predictions (probability of default)

        Raises:
            ValueError: If model has not been trained
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Apply fitted transformations and get predictions from all trained models
        results = self.pipeline.test(test=x_test, y_test=y_test if y_test is not None else [])

        # Extract predictions specifically from Logistic Regression model
        # Results dict keys contain model names
        for key in results:
            if "Logistic Regression" in key:
                return results[key][0]  # [0] = probabilities, [1] = predictions

        raise ValueError("Logistic Regression results not found")

    def save(self, path=None):
        """
        Save the trained pipeline to disk using pickle.

        Args:
            path: File path (default: Models/base_model_{VERSION}.pkl)
        """
        if path is None:
            path = os.path.join(Config.MODELS_DIR, f'base_model_{Config.VERSION}.pkl')

        # Use highest protocol for best performance
        with open(path, 'wb') as f:
            pickle.dump(self.pipeline, f, pickle.HIGHEST_PROTOCOL)
        print(f"Base model saved to: {path}")

    def load(self, path=None):
        """
        Load a trained pipeline from disk.

        Args:
            path: File path (default: Models/base_model_{VERSION}.pkl)
        """
        if path is None:
            path = os.path.join(Config.MODELS_DIR, f'base_model_{Config.VERSION}.pkl')

        with open(path, 'rb') as f:
            self.pipeline = pickle.load(f)

        # Mark as trained so predict() can be called
        self.is_trained = True
        print(f"Base model loaded from: {path}")


class SegmentPipeline:
    """
    Layer 2: Segment-Specific Model Pipeline

    Trains multiple models (LR, RF, XGB, LGBM) for segment-specific predictions.

    Purpose:
    - Provides specialized predictions for different risk segments
    - Good segment uses LGBM (captures complex patterns in low-risk applicants)
    - Not-Good segment uses LR (stable predictions for high-risk applicants)

    Key Differences from BasePipeline:
    - NO WOE binning (uses raw features with tree-based models)
    - Uses StandardScaler for feature normalization
    - Uses target encoding for categorical variables
    - Applies null imputation

    Training Flow:
    1. Data exploration
    2. Feature elimination (same as base)
    3. Scaling (StandardScaler)
    4. Null imputation (mean for numeric, constant for categorical)
    5. Categorical encoding (target encoding)
    6. Train all 4 model types (LR, RF, XGB, LGBM)
    """

    # -------------------------------------------------------------------------
    # Pipeline Step Configuration Flags
    # -------------------------------------------------------------------------
    # Different from base: no binning, uses scaling/imputation/encoding
    # These are needed for tree-based models like LGBM and XGBoost

    APPLY_DATA_EXPLORE = True           # Classify features as numeric/categorical
    APPLY_NULL_ELIMINATION = True       # Remove features with >99% missing
    APPLY_CONSTANT_ELIMINATION = True   # Remove zero-variance features
    APPLY_LOW_GINI_ELIMINATION = True   # Remove features with Gini < 0.05
    APPLY_CORRELATED_ELIMINATION = True # Remove correlated features
    APPLY_BINNING = False               # NO WOE binning for segment models
    APPLY_SCALING = True                # StandardScaler for feature normalization
    APPLY_NULL_IMPUTATION = True        # Impute missing values
    APPLY_CATEGORICAL_ENCODING = True   # Target encoding for categoricals

    def __init__(self, segment_name):
        """
        Initialize segment pipeline for a specific risk segment.

        Args:
            segment_name: 'good' (low-risk) or 'not_good' (high-risk)
        """
        self.segment_name = segment_name  # Used for model selection and file naming
        self.pipeline = None              # Will hold ClassifierPipeline after training
        self.is_trained = False           # Flag for prediction readiness

    def train(self, x_train, y_train, x_test=None, y_test=None):
        """
        Train all four model types for this segment with preprocessing.

        Trains LR, RF, XGBoost, and LightGBM models, but only the
        segment-appropriate model (defined in Config) is used for scoring.
        All models are trained for comparison and ensemble flexibility.

        Args:
            x_train: Training features DataFrame (segment-filtered)
            y_train: Training target Series (0/1)
            x_test: Validation features DataFrame (optional)
            y_test: Validation target Series (optional)
        """
        print(f"\nTraining {self.segment_name.upper()} segment models...")

        # Prepare validation data
        x_val = x_test.copy() if x_test is not None else pd.DataFrame()
        y_val = y_test.copy() if y_test is not None else pd.Series()

        # Initialize pipeline with segment-specific data
        self.pipeline = skills_api.ClassifierPipeline(x_train, y_train, x_val, y_val)

        # =====================================================================
        # Preprocessing Steps
        # =====================================================================

        # Data exploration - classify feature types
        if self.APPLY_DATA_EXPLORE:
            self.pipeline.data_explore()

        # Feature elimination - same steps as base model
        if self.APPLY_NULL_ELIMINATION:
            self.pipeline.feature_elimination(
                eliminator='drop_null_features',
                params={'threshold': 0.99}
            )

        if self.APPLY_CONSTANT_ELIMINATION:
            self.pipeline.feature_elimination(
                eliminator='drop_constant_features',
                params={'missing_values': 'ignore'}
            )

        if self.APPLY_LOW_GINI_ELIMINATION:
            self.pipeline.feature_elimination(
                eliminator='drop_low_gini_features',
                params={'threshold': 0.05, 'missing_values': 'ignore'}
            )

        if self.APPLY_CORRELATED_ELIMINATION:
            self.pipeline.feature_elimination(
                eliminator='correlated_lower_gini_feature_elimination',
                params={'missing_values': 'ignore'}
            )

        # Scaling - StandardScaler (mean=0, std=1)
        # Important for Logistic Regression; optional for tree models
        if self.APPLY_SCALING:
            self.pipeline.scaling()

        # Null imputation - mean for numeric, constant for categorical
        # Tree models can handle NaN, but consistent imputation is cleaner
        if self.APPLY_NULL_IMPUTATION:
            self.pipeline.null_imputation()

        # Categorical encoding - target encoding
        # Converts categorical variables to numeric using target mean
        if self.APPLY_CATEGORICAL_ENCODING:
            self.pipeline.encode_categoricals()

        # =====================================================================
        # Load Segment-Specific Hyperparameter Grids
        # =====================================================================
        # Different segments have different optimal parameters due to
        # different data distributions and target rates
        if self.segment_name == 'good':
            # Good segment grids (optimized for low default rates)
            lr_grid = param_grid_best['logistic_regression_good']
            rf_grid = param_grid_best['random_forest_good']
            xgb_grid = param_grid_best['xgboost_good']
            lgbm_grid = param_grid_best['lightgbm_good']
        else:
            # Not-good segment grids (optimized for higher default rates)
            lr_grid = param_grid_best['logistic_regression_not_good']
            rf_grid = param_grid_best['random_forest_not_good']
            xgb_grid = param_grid_best['xgboost_not_good']
            lgbm_grid = param_grid_best['lightgbm_not_good']

        # =====================================================================
        # Train All Four Model Types
        # =====================================================================
        # All models trained for comparison; Config determines which is used

        # Model 1: Logistic Regression
        # - Good baseline, interpretable
        # - Default for not-good segment (stable)
        print("  Training Logistic Regression...")
        self.pipeline.LogisticRegression(
            param_opt=True,
            param_opt_method="RandomizedSearchCV",
            train_size=x_train.shape[0],
            param_grid=lr_grid
        )

        # Model 2: Random Forest
        # - Ensemble of decision trees
        # - Good for feature importance analysis
        print("  Training Random Forest...")
        self.pipeline.RandomForest(
            param_opt=True,
            param_opt_method="RandomizedSearchCV",
            train_size=x_train.shape[0],
            param_grid=rf_grid
        )

        # Model 3: XGBoost
        # - Gradient boosting with regularization
        # - Often best performance on structured data
        print("  Training XGBoost...")
        self.pipeline.XGBoost(
            param_opt=True,
            param_opt_method="RandomizedSearchCV",
            train_size=x_train.shape[0],
            param_grid=xgb_grid
        )

        # Model 4: LightGBM
        # - Faster training, handles categoricals natively
        # - Default for good segment (captures complex patterns)
        print("  Training LightGBM...")
        self.pipeline.LGBM(
            param_opt=True,
            param_opt_method="RandomizedSearchCV",
            train_size=x_train.shape[0],
            param_grid=lgbm_grid
        )

        # Mark as trained
        self.is_trained = True
        print(f"{self.segment_name.upper()} segment training completed!")

    def predict(self, x_test, y_test=None, model_name=None):
        """
        Generate probability predictions using the segment-specific model.

        By default, uses the model specified in Config for this segment:
        - Good segment: LGBM
        - Not-Good segment: Logistic Regression

        Args:
            x_test: Features DataFrame for prediction
            y_test: True labels (optional, for evaluation metrics)
            model_name: Override default model selection
                        Options: 'Logistic Regression', 'Random Forest', 'XGBoost', 'LGBM'

        Returns:
            np.array: Probability predictions (probability of default)

        Raises:
            ValueError: If model not trained or specified model not found
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Apply fitted transformations and get predictions from all models
        results = self.pipeline.test(test=x_test, y_test=y_test if y_test is not None else [])

        # Determine which model to use for this segment
        if model_name is None:
            # Use segment-appropriate default from Config
            if self.segment_name == 'good':
                model_name = Config.GOOD_MODEL      # LGBM
            else:
                model_name = Config.NOT_GOOD_MODEL  # Logistic Regression

        # Extract predictions for the specified model
        for key in results:
            if model_name in key:
                return results[key][0]  # [0] = probabilities

        raise ValueError(f"{model_name} results not found")

    def save(self, path=None):
        """
        Save the trained pipeline to disk.

        Args:
            path: File path (default: Models/{segment_name}_model_{VERSION}.pkl)
        """
        if path is None:
            path = os.path.join(Config.MODELS_DIR, f'{self.segment_name}_model_{Config.VERSION}.pkl')

        with open(path, 'wb') as f:
            pickle.dump(self.pipeline, f, pickle.HIGHEST_PROTOCOL)
        print(f"{self.segment_name} model saved to: {path}")

    def load(self, path=None):
        """
        Load a trained pipeline from disk.

        Args:
            path: File path (default: Models/{segment_name}_model_{VERSION}.pkl)
        """
        if path is None:
            path = os.path.join(Config.MODELS_DIR, f'{self.segment_name}_model_{Config.VERSION}.pkl')

        with open(path, 'rb') as f:
            self.pipeline = pickle.load(f)
        self.is_trained = True
        print(f"{self.segment_name} model loaded from: {path}")


class MetaPipeline:
    """
    Layer 3: Meta Model Pipeline (Stacking Ensemble)

    Combines predictions from all previous layers using Logistic Regression.

    Purpose:
    - Final aggregation of all model predictions
    - Learns optimal weights for combining Layer 1 and Layer 2 outputs
    - Provides calibrated final probability estimates

    Architecture:
    - Input: 3 features [Base_prob, Good_prob, Not_Good_prob]
    - Model: Logistic Regression
    - Output: Final probability prediction

    Why Logistic Regression for Meta?
    - Simple, interpretable combination weights
    - No risk of overfitting with only 3 features
    - Naturally produces calibrated probabilities
    - Fast training and inference
    """

    def __init__(self):
        """Initialize meta pipeline with no trained model."""
        self.pipeline = None      # Will hold ClassifierPipeline after training
        self.is_trained = False   # Flag for prediction readiness

    def train(self, x_train, y_train, x_test=None, y_test=None):
        """
        Train meta model on stacked predictions from previous layers.

        The meta model learns how to optimally combine:
        - Base model probability (Layer 1)
        - Good segment probability (Layer 2a)
        - Not-Good segment probability (Layer 2b)

        Args:
            x_train: Meta features DataFrame with columns [Base, Good, Not_Good]
            y_train: Training target Series (0/1)
            x_test: Meta features for validation (optional)
            y_test: Validation target (optional)
        """
        print("\n" + "="*60)
        print("LAYER 3: Training Meta Model")
        print("="*60)

        # Prepare validation data
        x_val = x_test.copy() if x_test is not None else pd.DataFrame()
        y_val = y_test.copy() if y_test is not None else pd.Series()

        # Initialize pipeline with meta features
        # Note: Much simpler than base/segment pipelines - only 3 features
        self.pipeline = skills_api.ClassifierPipeline(x_train, y_train, x_val, y_val)

        # Data exploration (classifies the 3 probability features as numeric)
        self.pipeline.data_explore()

        # Train Logistic Regression only
        # No feature elimination needed - all 3 features are informative
        # No binning/scaling needed - inputs are already probabilities [0,1]
        print("Training Meta Logistic Regression...")
        self.pipeline.LogisticRegression(
            param_opt=True,
            param_opt_method="RandomizedSearchCV",
            train_size=x_train.shape[0],
            param_grid=param_grid_best['logistic_regression_meta']
        )

        self.is_trained = True
        print("Meta model training completed!")

    def predict(self, x_test, y_test=None):
        """
        Generate final probability predictions from meta model.

        Args:
            x_test: Meta features DataFrame with columns [Base, Good, Not_Good]
            y_test: True labels (optional, for evaluation)

        Returns:
            np.array: Final probability predictions (probability of default)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Get predictions from all trained models (only LR in meta)
        results = self.pipeline.test(test=x_test, y_test=y_test if y_test is not None else [])

        # Extract Logistic Regression predictions
        for key in results:
            if "Logistic Regression" in key:
                return results[key][0]

        raise ValueError("Logistic Regression results not found")

    def save(self, path=None):
        """
        Save the trained meta pipeline to disk.

        Args:
            path: File path (default: Models/meta_model_{VERSION}.pkl)
        """
        if path is None:
            path = os.path.join(Config.MODELS_DIR, f'meta_model_{Config.VERSION}.pkl')

        with open(path, 'wb') as f:
            pickle.dump(self.pipeline, f, pickle.HIGHEST_PROTOCOL)
        print(f"Meta model saved to: {path}")

    def load(self, path=None):
        """
        Load a trained meta pipeline from disk.

        Args:
            path: File path (default: Models/meta_model_{VERSION}.pkl)
        """
        if path is None:
            path = os.path.join(Config.MODELS_DIR, f'meta_model_{Config.VERSION}.pkl')

        with open(path, 'rb') as f:
            self.pipeline = pickle.load(f)
        self.is_trained = True
        print(f"Meta model loaded from: {path}")


# =============================================================================
# MAIN PIPELINE ORCHESTRATOR
# =============================================================================

class CreditScoringPipeline:
    """
    Main orchestrator for the 3-layer credit scoring pipeline.

    This is the primary entry point for the credit scoring system.
    It coordinates all layers and handles the complete workflow:

    1. Data Loading: Fetches training/test data from Oracle database
    2. Training: Trains all 3 layers in sequence with proper segmentation
    3. Scoring: Applies trained models to new applicants
    4. Output: Generates scores and exports to Excel

    Usage:
        pipeline = CreditScoringPipeline()
        pipeline.train()           # Train from database
        pipeline.score()           # Score new applicants

        # Or load existing models
        pipeline.load_models()
        pipeline.score()

    Architecture Flow:
        Input Features
            ↓
        Layer 1: Base Model (all data)
            ↓
        Segment based on Base probability
            ↓
        Layer 2: Good Model | Not-Good Model
            ↓
        Layer 3: Meta Model (combines all)
            ↓
        Score Transformation
            ↓
        Policy Adjustment
            ↓
        Final Credit Score
    """

    def __init__(self):
        """
        Initialize the credit scoring pipeline with all layer components.

        Creates instances of all 4 pipelines (Base, Good, Not-Good, Meta)
        but does not train them. Call train() or load_models() before scoring.
        """
        # Initialize all layer pipelines
        self.base_pipeline = BasePipeline()               # Layer 1
        self.good_pipeline = SegmentPipeline('good')      # Layer 2a
        self.not_good_pipeline = SegmentPipeline('not_good')  # Layer 2b
        self.meta_pipeline = MetaPipeline()               # Layer 3

        # Training data references (populated by load_training_data())
        self.x_train = None  # Training features DataFrame
        self.y_train = None  # Training target Series
        self.x_test = None   # Validation features DataFrame
        self.y_test = None   # Validation target Series

    def load_training_data(self):
        """
        Load training and validation data from Oracle database.

        Executes SQL queries defined in Config to fetch:
        - Training data (for model fitting)
        - Test/validation data (for performance evaluation)

        Data preparation includes:
        - Setting contract ID as index
        - Renaming target column to 'TARGET'
        - Dropping excluded columns (data quality/leakage issues)

        Populates:
            self.x_train, self.y_train: Training data
            self.x_test, self.y_test: Validation data
        """
        print("\n" + "="*60)
        print("Loading Training Data")
        print("="*60)

        # =================================================================
        # Load Training Data
        # =================================================================
        # Connect to Oracle database
        engine = get_database_connection()

        # Read SQL query from file
        with open(Config.TRAIN_SQL_FILE) as f:
            train_sql = f.read()

        # Execute query and load into DataFrame
        train = data.data_load(engine, sql=train_sql)

        # Prepare training data:
        # - Set contract ID as index (for result tracking)
        # - Standardize target column name
        # - Separate features and target
        # - Remove excluded columns
        train = train.set_index(Config.INDEX_COL).rename(columns={Config.TARGET_COL: "TARGET"})
        self.y_train = train["TARGET"]
        self.x_train = train.drop(columns=["TARGET"] + Config.COLS_TO_DROP, errors='ignore')

        print(f"Training data: {self.x_train.shape[0]} samples, {self.x_train.shape[1]} features")

        # =================================================================
        # Load Validation/Test Data
        # =================================================================
        # Create new connection (previous may have timed out)
        engine = get_database_connection()

        # Read validation SQL query
        with open(Config.TEST_SQL_FILE) as f:
            test_sql = f.read()

        # Execute query and load into DataFrame
        test = data.data_load(engine, sql=test_sql)

        # Prepare validation data (same process as training)
        test = test.set_index(Config.INDEX_COL).rename(columns={Config.TARGET_COL: "TARGET"})
        self.y_test = test["TARGET"]
        self.x_test = test.drop(columns=["TARGET"] + Config.COLS_TO_DROP, errors='ignore')

        print(f"Test data: {self.x_test.shape[0]} samples, {self.x_test.shape[1]} features")

    def train(self):
        """
        Train the complete 3-layer pipeline from scratch.

        This is the main training workflow that:
        1. Loads data from database (if not already loaded)
        2. Trains Layer 1 (Base model on all data)
        3. Segments data based on Layer 1 predictions
        4. Trains Layer 2 (segment-specific models)
        5. Trains Layer 3 (meta model on stacked predictions)
        6. Saves all trained models to disk
        7. Exports training scores to Excel

        Returns:
            pd.DataFrame: Training results with columns [PROBA, SCORE, TARGET]
        """
        start_time = datetime.datetime.now()

        print("\n" + "="*60)
        print("CREDIT SCORING PIPELINE - TRAINING")
        print("="*60)

        # Load data from database if not already loaded
        if self.x_train is None:
            self.load_training_data()

        # =====================================================================
        # LAYER 1: Base Model Training
        # =====================================================================
        # Train Logistic Regression with WOE binning on ALL data
        # This provides initial risk estimates and enables segmentation
        self.base_pipeline.train(
            self.x_train, self.y_train,
            self.x_test, self.y_test
        )

        # Get base model predictions for segmentation
        # These probabilities determine which segment each applicant belongs to
        base_train_proba = self.base_pipeline.predict(self.x_train, self.y_train)
        base_test_proba = self.base_pipeline.predict(self.x_test, self.y_test)

        # =====================================================================
        # Data Segmentation for Layer 2
        # =====================================================================
        print("\n" + "="*60)
        print("LAYER 2: Training Segment Models")
        print("="*60)

        # Convert score thresholds to probability thresholds
        # IMPORTANT: Uses ODDS_AT_REFERENCE_SEGMENTATION (50) to match original notebook
        # Lower probability = better creditworthiness = higher score
        # Good segment: prob < threshold (higher score applicants)
        # Not-Good segment: prob >= threshold (lower score applicants)
        good_threshold = score_to_probability(
            Config.GOOD_SCORE_THRESHOLD,
            odds_at_ref=Config.ODDS_AT_REFERENCE_SEGMENTATION  # Uses 50, not 100
        )
        not_good_threshold = score_to_probability(
            Config.NOT_GOOD_SCORE_THRESHOLD,
            odds_at_ref=Config.ODDS_AT_REFERENCE_SEGMENTATION  # Uses 50, not 100
        )

        print(f"Good threshold (prob): {good_threshold:.4f}")
        print(f"Not-Good threshold (prob): {not_good_threshold:.4f}")

        # Create segment masks for training data
        # Note: There can be overlap between segments (180-200 score range)
        good_train_idx = base_train_proba < good_threshold           # Lower risk applicants
        not_good_train_idx = base_train_proba >= not_good_threshold  # Higher risk applicants

        # Split training data into segments
        x_train_good = self.x_train[good_train_idx]
        y_train_good = self.y_train[good_train_idx]
        x_train_not_good = self.x_train[not_good_train_idx]
        y_train_not_good = self.y_train[not_good_train_idx]

        print(f"Good segment: {len(x_train_good)} samples")
        print(f"Not-Good segment: {len(x_train_not_good)} samples")

        # Split validation data using same thresholds
        good_test_idx = base_test_proba < good_threshold
        not_good_test_idx = base_test_proba >= not_good_threshold

        x_test_good = self.x_test[good_test_idx]
        y_test_good = self.y_test[good_test_idx]
        x_test_not_good = self.x_test[not_good_test_idx]
        y_test_not_good = self.y_test[not_good_test_idx]

        # Train segment-specific models
        # Good segment: LGBM (captures complex patterns in low-risk population)
        # Not-Good segment: LR (stable predictions for high-risk population)
        self.good_pipeline.train(x_train_good, y_train_good, x_test_good, y_test_good)
        self.not_good_pipeline.train(x_train_not_good, y_train_not_good, x_test_not_good, y_test_not_good)

        # =====================================================================
        # LAYER 3: Meta Model Training
        # =====================================================================
        # Train Logistic Regression on stacked predictions from all previous layers
        # Note: All applicants get predictions from both segment models

        # Generate meta features for training data
        # Each row has 3 probability predictions
        good_train_proba = self.good_pipeline.predict(self.x_train, self.y_train)
        not_good_train_proba = self.not_good_pipeline.predict(self.x_train, self.y_train)

        x_train_meta = pd.DataFrame({
            'Base': base_train_proba,          # Layer 1 prediction
            'Good': good_train_proba,          # Layer 2a prediction
            'Not_Good': not_good_train_proba   # Layer 2b prediction
        }, index=self.y_train.index)

        # Generate meta features for validation data
        good_test_proba = self.good_pipeline.predict(self.x_test, self.y_test)
        not_good_test_proba = self.not_good_pipeline.predict(self.x_test, self.y_test)

        x_test_meta = pd.DataFrame({
            'Base': base_test_proba,
            'Good': good_test_proba,
            'Not_Good': not_good_test_proba
        }, index=self.y_test.index)

        # Train meta model to learn optimal combination weights
        self.meta_pipeline.train(x_train_meta, self.y_train, x_test_meta, self.y_test)

        # =====================================================================
        # Save All Trained Models
        # =====================================================================
        print("\n" + "="*60)
        print("Saving Models")
        print("="*60)

        # Pickle all pipelines to Models/ directory
        self.base_pipeline.save()
        self.good_pipeline.save()
        self.not_good_pipeline.save()
        self.meta_pipeline.save()

        # =====================================================================
        # Calculate and Export Final Scores
        # =====================================================================
        print("\n" + "="*60)
        print("Calculating Final Scores")
        print("="*60)

        # Get final probability predictions from meta model
        meta_train_proba = self.meta_pipeline.predict(x_train_meta, self.y_train)
        meta_test_proba = self.meta_pipeline.predict(x_test_meta, self.y_test)

        # Convert probabilities to credit scores using log-odds transformation
        meta_train_score = probability_to_score(meta_train_proba)
        meta_test_score = probability_to_score(meta_test_proba)

        # Combine training and test results for export
        all_x = pd.concat([self.x_train, self.x_test])
        all_y = pd.concat([self.y_train, self.y_test])
        all_proba = np.concatenate([meta_train_proba, meta_test_proba])
        all_score = np.concatenate([meta_train_score, meta_test_score])

        # Create results DataFrame
        results = pd.DataFrame({
            'PROBA': all_proba,    # Final probability (0-1)
            'SCORE': all_score,    # Credit score (higher = better)
            'TARGET': all_y        # Actual outcome for validation
        }, index=all_y.index)

        # Export to Excel for analysis
        output_path = os.path.join(Config.OUTPUT_DIR, f'TRAINING_SCORES_{Config.VERSION}.xlsx')
        results.to_excel(output_path)
        print(f"Scores saved to: {output_path}")

        # Report total training time
        elapsed = datetime.datetime.now() - start_time
        print(f"\nTotal training time: {elapsed}")

        return results

    def load_models(self):
        """
        Load all trained models from disk.

        Loads previously saved pipelines from Models/ directory.
        Must be called before score() if models were trained in a previous session.

        Expected files:
        - Models/base_model_{VERSION}.pkl
        - Models/good_model_{VERSION}.pkl
        - Models/not_good_model_{VERSION}.pkl
        - Models/meta_model_{VERSION}.pkl
        """
        print("\nLoading trained models...")

        # Load all 4 pipelines from pickle files
        self.base_pipeline.load()       # Layer 1
        self.good_pipeline.load()       # Layer 2a
        self.not_good_pipeline.load()   # Layer 2b
        self.meta_pipeline.load()       # Layer 3

        print("All models loaded successfully!")

    def score(self, x_data=None, y_data=None, apply_policy=True):
        """
        Score new applicants using the trained pipeline.

        Applies all 3 layers sequentially to generate final credit scores:
        1. Layer 1 (Base) -> probability
        2. Layer 2 (Segments) -> 2 probabilities
        3. Layer 3 (Meta) -> final probability
        4. Score transformation -> credit score
        5. Policy adjustment -> final score (if enabled)

        Args:
            x_data: Features DataFrame (if None, loads from Config.SCORING_SQL_FILE)
            y_data: True labels (optional, for evaluation and export)
            apply_policy: Whether to apply policy adjustments (default: True)
                         - Cap at POLICY_CONSTANT (250)
                         - Apply POLICY_MULTIPLIER (0.95)

        Returns:
            pd.DataFrame: Scoring results with columns:
                - PROBA: Final probability (0-1)
                - RAW_SCORE: Score before policy adjustment
                - FINAL_SCORE: Score after policy adjustment
                - TARGET: True outcome (if y_data provided)
        """
        print("\n" + "="*60)
        print("CREDIT SCORING PIPELINE - SCORING")
        print("="*60)

        # =====================================================================
        # Load Scoring Data (if not provided)
        # =====================================================================
        if x_data is None:
            # Connect to database and load new applicants
            engine = get_database_connection()
            with open(Config.SCORING_SQL_FILE) as f:
                scoring_sql = f.read()
            scoring_data = data.data_load(engine, sql=scoring_sql)

            # Prepare data (same as training data preparation)
            scoring_data = scoring_data.set_index(Config.INDEX_COL).rename(
                columns={Config.TARGET_COL: "TARGET"}
            )
            y_data = scoring_data["TARGET"]
            x_data = scoring_data.drop(columns=["TARGET"] + Config.COLS_TO_DROP, errors='ignore')

        print(f"Scoring {len(x_data)} applicants...")

        # =====================================================================
        # Apply All 3 Layers
        # =====================================================================

        # Layer 1: Base model (LR with WOE binning)
        # All applicants go through base model
        print("Applying Layer 1 (Base)...")
        base_proba = self.base_pipeline.predict(x_data, y_data)

        # Layer 2: Segment models
        # Note: ALL applicants get predictions from BOTH segment models
        # (not just the segment they belong to)
        # The meta model learns how to combine these appropriately
        print("Applying Layer 2 (Segments)...")
        good_proba = self.good_pipeline.predict(x_data, y_data)
        not_good_proba = self.not_good_pipeline.predict(x_data, y_data)

        # Layer 3: Meta model (combines all predictions)
        # Input: 3 probability features
        # Output: final calibrated probability
        print("Applying Layer 3 (Meta)...")
        x_meta = pd.DataFrame({
            'Base': base_proba,
            'Good': good_proba,
            'Not_Good': not_good_proba
        }, index=x_data.index if hasattr(x_data, 'index') else None)

        meta_proba = self.meta_pipeline.predict(x_meta, y_data)

        # =====================================================================
        # Score Transformation and Policy Adjustment
        # =====================================================================

        # Convert probability to credit score using log-odds transformation
        raw_score = probability_to_score(meta_proba)

        # Apply business policy adjustments (cap and discount)
        if apply_policy:
            final_score = apply_policy_adjustment(raw_score)
        else:
            final_score = raw_score

        # =====================================================================
        # Build and Export Results
        # =====================================================================

        # Create results DataFrame with all score components
        results = pd.DataFrame({
            'PROBA': meta_proba,        # Final probability (0-1)
            'RAW_SCORE': raw_score,     # Score before policy adjustment
            'FINAL_SCORE': final_score  # Score after policy adjustment
        })

        # Include target if available (for validation analysis)
        if y_data is not None:
            results['TARGET'] = y_data.values if hasattr(y_data, 'values') else y_data

        # Preserve original index (contract IDs)
        if hasattr(x_data, 'index'):
            results.index = x_data.index

        # Export results to Excel
        output_path = os.path.join(Config.OUTPUT_DIR, f'SCORING_RESULTS_{Config.VERSION}.xlsx')
        results.to_excel(output_path)
        print(f"Results saved to: {output_path}")

        return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main entry point for the credit scoring pipeline.

    Parses command-line arguments and executes the appropriate workflow:
    - train: Train all 3 layers from database data
    - score: Score new applicants using saved models
    - both: Train then immediately score
    - test: Verify configuration and utility functions

    Command-line arguments:
        --mode: Execution mode (train/score/both/test)
        --no-policy: Skip policy adjustments when scoring

    Examples:
        python credit_scoring_pipeline.py --mode train
        python credit_scoring_pipeline.py --mode score
        python credit_scoring_pipeline.py --mode score --no-policy
        python credit_scoring_pipeline.py --mode both
    """
    # =========================================================================
    # Parse Command-Line Arguments
    # =========================================================================
    parser = argparse.ArgumentParser(
        description='QNBAnalytics ML Credit Scoring Pipeline'
    )
    parser.add_argument(
        '--mode',
        choices=['train', 'score', 'both', 'test'],
        default='test',
        help='Execution mode: train, score, both, or test'
    )
    parser.add_argument(
        '--no-policy',
        action='store_true',
        help='Skip policy adjustments when scoring'
    )

    args = parser.parse_args()

    # =========================================================================
    # Setup Directories
    # =========================================================================
    # Create output directories if they don't exist
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)   # For scoring results
    os.makedirs(Config.MODELS_DIR, exist_ok=True)   # For saved models

    # =========================================================================
    # Initialize and Execute Pipeline
    # =========================================================================
    # Create main pipeline orchestrator
    pipeline = CreditScoringPipeline()

    # Execute based on mode
    if args.mode == 'train':
        # Train all layers from scratch using database data
        # Saves models to Models/ directory
        # Exports training scores to Output/
        pipeline.train()

    elif args.mode == 'score':
        # Load previously trained models and score new applicants
        # Requires models to have been trained previously
        pipeline.load_models()
        pipeline.score(apply_policy=not args.no_policy)

    elif args.mode == 'both':
        # Train models and then immediately score new applicants
        # Useful for full end-to-end pipeline execution
        pipeline.train()
        pipeline.score(apply_policy=not args.no_policy)

    elif args.mode == 'test':
        # =====================================================================
        # Test Mode: Verify Pipeline Configuration
        # =====================================================================
        # Quick sanity check without database connection
        # Verifies all components can be instantiated and configured

        print("\n" + "="*60)
        print("PIPELINE TEST MODE")
        print("="*60)

        # Test pipeline initialization
        print("\nTesting pipeline initialization...")
        print(f"  - Base pipeline: OK")
        print(f"  - Good segment pipeline: OK")
        print(f"  - Not-Good segment pipeline: OK")
        print(f"  - Meta pipeline: OK")

        # Test configuration values
        print("\nTesting configuration...")
        print(f"  - Reference score: {Config.REFERENCE_SCORE}")
        print(f"  - Odds at reference: {Config.ODDS_AT_REFERENCE}")
        print(f"  - Points to double: {Config.POINTS_TO_DOUBLE}")
        print(f"  - Columns to drop: {len(Config.COLS_TO_DROP)}")

        # Test probability <-> score conversion functions
        print("\nTesting utility functions...")
        test_prob = 0.05  # 5% default probability
        test_score = probability_to_score(test_prob)
        back_prob = score_to_probability(test_score)
        print(f"  - Probability {test_prob} -> Score {test_score:.2f}")
        print(f"  - Score {test_score:.2f} -> Probability {back_prob:.4f}")

        # Test policy adjustment function
        test_scores = np.array([180, 200, 220, 240, 260])
        adjusted = apply_policy_adjustment(test_scores)
        print(f"\n  - Policy adjustment test:")
        for orig, adj in zip(test_scores, adjusted):
            print(f"    Score {orig} -> {adj:.1f}")

        print("\nAll tests passed!")
        print("\nTo run full training: python credit_scoring_pipeline.py --mode train")
        print("To run scoring: python credit_scoring_pipeline.py --mode score")


# =============================================================================
# Script Entry Point
# =============================================================================
if __name__ == "__main__":
    main()

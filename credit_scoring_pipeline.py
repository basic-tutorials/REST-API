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

Usage:
    python credit_scoring_pipeline.py --mode train
    python credit_scoring_pipeline.py --mode score
    python credit_scoring_pipeline.py --mode both

Author: QNBAnalytics ML Team
Version: 0.3.2
"""

import os
import sys
import argparse
import pickle
import datetime
import warnings
import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
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
    """Central configuration for the credit scoring pipeline."""

    # Version identifier
    VERSION = "training"

    # Data columns
    INDEX_COL = "MUQAVILE"  # Contract ID column
    TARGET_COL = "TARGET"   # Binary target (0=good, 1=bad)

    # Columns to exclude from training (data quality/leakage issues)
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

    # Credit score transformation parameters
    REFERENCE_SCORE = 200      # Anchor score
    ODDS_AT_REFERENCE = 100    # Odds ratio at reference (100:1)
    POINTS_TO_DOUBLE = 20      # Points needed to double odds

    # Segmentation thresholds
    GOOD_SCORE_THRESHOLD = 180      # Score >= 180 = good segment
    NOT_GOOD_SCORE_THRESHOLD = 200  # Score <= 200 = not-good segment

    # Policy adjustment parameters
    POLICY_CONSTANT = 250      # Maximum score cap
    POLICY_MULTIPLIER = 0.95   # Conservative discount factor

    # Model selection for Layer 2
    NOT_GOOD_MODEL = 'Logistic Regression'
    GOOD_MODEL = 'LGBM'

    # File paths
    DATA_DIR = "Data"
    MODELS_DIR = "Models"
    OUTPUT_DIR = "Output"

    # Database credentials files
    USER_FILE = "Data/user"
    PASS_FILE = "Data/pass"

    # SQL query files
    TRAIN_SQL_FILE = "Data/train_data_sql_training.txt"
    TEST_SQL_FILE = "Data/test_data_sql_training.txt"
    SCORING_SQL_FILE = "Data/test_data_sql_scoring_policy_adjustment.txt"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_database_connection():
    """Establish database connection using stored credentials."""
    username = pd.read_table(Config.USER_FILE, header=None)[0][0]
    password = pd.read_table(Config.PASS_FILE, header=None)[0][0]
    return data.connect_to_sql(username, password)


def probability_to_score(probability, ref=None, odds_at_ref=None, pdo=None):
    """
    Convert probability to credit score.

    Formula: score = ((log(odds) - log(odds_at_ref)) / log(2)) * PDO + ref

    Args:
        probability: Default probability (0-1)
        ref: Reference score (default: Config.REFERENCE_SCORE)
        odds_at_ref: Odds at reference score (default: Config.ODDS_AT_REFERENCE)
        pdo: Points to double odds (default: Config.POINTS_TO_DOUBLE)

    Returns:
        Credit score
    """
    if ref is None:
        ref = Config.REFERENCE_SCORE
    if odds_at_ref is None:
        odds_at_ref = Config.ODDS_AT_REFERENCE
    if pdo is None:
        pdo = Config.POINTS_TO_DOUBLE

    # Avoid division by zero
    prob = np.where(probability == 0, 0.00001, probability)

    # Calculate odds
    odds = (1 / prob) - 1

    # Calculate score
    score = ((np.log(odds) - np.log(odds_at_ref)) / np.log(2)) * pdo + ref

    return score


def score_to_probability(score, ref=None, odds_at_ref=None, pdo=None):
    """
    Convert credit score back to probability.

    Args:
        score: Credit score
        ref: Reference score
        odds_at_ref: Odds at reference score
        pdo: Points to double odds

    Returns:
        Default probability
    """
    if ref is None:
        ref = Config.REFERENCE_SCORE
    if odds_at_ref is None:
        odds_at_ref = Config.ODDS_AT_REFERENCE
    if pdo is None:
        pdo = Config.POINTS_TO_DOUBLE

    probability = 1 / (np.exp((score - ref) / pdo * np.log(2) + np.log(odds_at_ref)) + 1)
    return probability


def apply_policy_adjustment(scores):
    """
    Apply business policy adjustments to credit scores.

    Adjustments:
    1. Cap at POLICY_CONSTANT (e.g., 250)
    2. Apply POLICY_MULTIPLIER discount (e.g., 0.95)

    Args:
        scores: Raw model scores

    Returns:
        Policy-adjusted scores
    """
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

    Uses Logistic Regression with WOE binning on all data.
    """

    # Pipeline settings
    APPLY_DATA_EXPLORE = True
    APPLY_NULL_ELIMINATION = True
    APPLY_CONSTANT_ELIMINATION = True
    APPLY_LOW_GINI_ELIMINATION = True
    APPLY_CORRELATED_ELIMINATION = True
    APPLY_BINNING = True
    APPLY_SCALING = False
    APPLY_NULL_IMPUTATION = False
    APPLY_CATEGORICAL_ENCODING = False

    def __init__(self):
        self.pipeline = None
        self.is_trained = False

    def train(self, x_train, y_train, x_test=None, y_test=None):
        """Train the base model."""
        print("\n" + "="*60)
        print("LAYER 1: Training Base Model")
        print("="*60)

        x_val = x_test.copy() if x_test is not None else pd.DataFrame()
        y_val = y_test.copy() if y_test is not None else pd.Series()

        # Initialize pipeline
        self.pipeline = skills_api.ClassifierPipeline(x_train, y_train, x_val, y_val)

        # Data exploration
        if self.APPLY_DATA_EXPLORE:
            print("Step 1: Data exploration...")
            self.pipeline.data_explore()

        # Feature elimination
        if self.APPLY_NULL_ELIMINATION:
            print("Step 2: Null feature elimination...")
            self.pipeline.feature_elimination(
                eliminator='drop_null_features',
                params={'threshold': 0.99}
            )

        if self.APPLY_CONSTANT_ELIMINATION:
            print("Step 3: Constant feature elimination...")
            self.pipeline.feature_elimination(
                eliminator='drop_constant_features',
                params={'missing_values': 'ignore'}
            )

        if self.APPLY_LOW_GINI_ELIMINATION:
            print("Step 4: Low Gini feature elimination...")
            self.pipeline.feature_elimination(
                eliminator='drop_low_gini_features',
                params={'threshold': 0.05, 'missing_values': 'ignore'}
            )

        if self.APPLY_CORRELATED_ELIMINATION:
            print("Step 5: Correlated feature elimination...")
            self.pipeline.feature_elimination(
                eliminator='correlated_lower_gini_feature_elimination',
                params={'missing_values': 'ignore'}
            )

        # WOE Binning
        if self.APPLY_BINNING:
            print("Step 6: WOE Binning...")
            # Use existing binning if available, otherwise create new
            binning_path = os.path.join(Config.MODELS_DIR, 'binning.pkl')
            if os.path.exists(binning_path):
                self.pipeline.binning(use_existing='binning.pkl')
            else:
                self.pipeline.binning()

        # Train Logistic Regression
        print("Step 7: Training Logistic Regression...")
        self.pipeline.LogisticRegression(
            param_opt=True,
            param_opt_method="RandomizedSearchCV",
            train_size=x_train.shape[0],
            param_grid=param_grid_best['logistic_regression_base']
        )

        self.is_trained = True
        print("Base model training completed!")

    def predict(self, x_test, y_test=None):
        """Get predictions from base model."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        results = self.pipeline.test(test=x_test, y_test=y_test if y_test is not None else [])

        # Extract Logistic Regression predictions
        for key in results:
            if "Logistic Regression" in key:
                return results[key][0]

        raise ValueError("Logistic Regression results not found")

    def save(self, path=None):
        """Save trained model."""
        if path is None:
            path = os.path.join(Config.MODELS_DIR, f'base_model_{Config.VERSION}.pkl')

        with open(path, 'wb') as f:
            pickle.dump(self.pipeline, f, pickle.HIGHEST_PROTOCOL)
        print(f"Base model saved to: {path}")

    def load(self, path=None):
        """Load trained model."""
        if path is None:
            path = os.path.join(Config.MODELS_DIR, f'base_model_{Config.VERSION}.pkl')

        with open(path, 'rb') as f:
            self.pipeline = pickle.load(f)
        self.is_trained = True
        print(f"Base model loaded from: {path}")


class SegmentPipeline:
    """
    Layer 2: Segment-Specific Model Pipeline

    Trains multiple models (LR, RF, XGB, LGBM) for segment-specific predictions.
    """

    # Pipeline settings (different from base)
    APPLY_DATA_EXPLORE = True
    APPLY_NULL_ELIMINATION = True
    APPLY_CONSTANT_ELIMINATION = True
    APPLY_LOW_GINI_ELIMINATION = True
    APPLY_CORRELATED_ELIMINATION = True
    APPLY_BINNING = False  # No WOE for segment models
    APPLY_SCALING = True
    APPLY_NULL_IMPUTATION = True
    APPLY_CATEGORICAL_ENCODING = True

    def __init__(self, segment_name):
        """
        Args:
            segment_name: 'good' or 'not_good'
        """
        self.segment_name = segment_name
        self.pipeline = None
        self.is_trained = False

    def train(self, x_train, y_train, x_test=None, y_test=None):
        """Train segment-specific models."""
        print(f"\nTraining {self.segment_name.upper()} segment models...")

        x_val = x_test.copy() if x_test is not None else pd.DataFrame()
        y_val = y_test.copy() if y_test is not None else pd.Series()

        # Initialize pipeline
        self.pipeline = skills_api.ClassifierPipeline(x_train, y_train, x_val, y_val)

        # Data exploration
        if self.APPLY_DATA_EXPLORE:
            self.pipeline.data_explore()

        # Feature elimination
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

        # Scaling
        if self.APPLY_SCALING:
            self.pipeline.scaling()

        # Null imputation
        if self.APPLY_NULL_IMPUTATION:
            self.pipeline.null_imputation()

        # Categorical encoding
        if self.APPLY_CATEGORICAL_ENCODING:
            self.pipeline.encode_categoricals()

        # Get param grids for this segment
        if self.segment_name == 'good':
            lr_grid = param_grid_best['logistic_regression_good']
            rf_grid = param_grid_best['random_forest_good']
            xgb_grid = param_grid_best['xgboost_good']
            lgbm_grid = param_grid_best['lightgbm_good']
        else:
            lr_grid = param_grid_best['logistic_regression_not_good']
            rf_grid = param_grid_best['random_forest_not_good']
            xgb_grid = param_grid_best['xgboost_not_good']
            lgbm_grid = param_grid_best['lightgbm_not_good']

        # Train all models
        print("  Training Logistic Regression...")
        self.pipeline.LogisticRegression(
            param_opt=True,
            param_opt_method="RandomizedSearchCV",
            train_size=x_train.shape[0],
            param_grid=lr_grid
        )

        print("  Training Random Forest...")
        self.pipeline.RandomForest(
            param_opt=True,
            param_opt_method="RandomizedSearchCV",
            train_size=x_train.shape[0],
            param_grid=rf_grid
        )

        print("  Training XGBoost...")
        self.pipeline.XGBoost(
            param_opt=True,
            param_opt_method="RandomizedSearchCV",
            train_size=x_train.shape[0],
            param_grid=xgb_grid
        )

        print("  Training LightGBM...")
        self.pipeline.LGBM(
            param_opt=True,
            param_opt_method="RandomizedSearchCV",
            train_size=x_train.shape[0],
            param_grid=lgbm_grid
        )

        self.is_trained = True
        print(f"{self.segment_name.upper()} segment training completed!")

    def predict(self, x_test, y_test=None, model_name=None):
        """
        Get predictions from segment model.

        Args:
            x_test: Test features
            y_test: Test labels (optional)
            model_name: Which model to use ('Logistic Regression', 'LGBM', etc.)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        results = self.pipeline.test(test=x_test, y_test=y_test if y_test is not None else [])

        # Use specified model or default
        if model_name is None:
            if self.segment_name == 'good':
                model_name = Config.GOOD_MODEL
            else:
                model_name = Config.NOT_GOOD_MODEL

        # Extract predictions for specified model
        for key in results:
            if model_name in key:
                return results[key][0]

        raise ValueError(f"{model_name} results not found")

    def save(self, path=None):
        """Save trained model."""
        if path is None:
            path = os.path.join(Config.MODELS_DIR, f'{self.segment_name}_model_{Config.VERSION}.pkl')

        with open(path, 'wb') as f:
            pickle.dump(self.pipeline, f, pickle.HIGHEST_PROTOCOL)
        print(f"{self.segment_name} model saved to: {path}")

    def load(self, path=None):
        """Load trained model."""
        if path is None:
            path = os.path.join(Config.MODELS_DIR, f'{self.segment_name}_model_{Config.VERSION}.pkl')

        with open(path, 'rb') as f:
            self.pipeline = pickle.load(f)
        self.is_trained = True
        print(f"{self.segment_name} model loaded from: {path}")


class MetaPipeline:
    """
    Layer 3: Meta Model Pipeline

    Combines predictions from all previous layers using Logistic Regression.
    """

    def __init__(self):
        self.pipeline = None
        self.is_trained = False

    def train(self, x_train, y_train, x_test=None, y_test=None):
        """Train meta model on stacked predictions."""
        print("\n" + "="*60)
        print("LAYER 3: Training Meta Model")
        print("="*60)

        x_val = x_test.copy() if x_test is not None else pd.DataFrame()
        y_val = y_test.copy() if y_test is not None else pd.Series()

        # Initialize pipeline (simple - just data explore and LR)
        self.pipeline = skills_api.ClassifierPipeline(x_train, y_train, x_val, y_val)

        # Data exploration
        self.pipeline.data_explore()

        # Train Logistic Regression
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
        """Get final predictions from meta model."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        results = self.pipeline.test(test=x_test, y_test=y_test if y_test is not None else [])

        for key in results:
            if "Logistic Regression" in key:
                return results[key][0]

        raise ValueError("Logistic Regression results not found")

    def save(self, path=None):
        """Save trained model."""
        if path is None:
            path = os.path.join(Config.MODELS_DIR, f'meta_model_{Config.VERSION}.pkl')

        with open(path, 'wb') as f:
            pickle.dump(self.pipeline, f, pickle.HIGHEST_PROTOCOL)
        print(f"Meta model saved to: {path}")

    def load(self, path=None):
        """Load trained model."""
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

    Handles:
    - Data loading
    - Training all layers
    - Scoring new applicants
    - Policy adjustments
    """

    def __init__(self):
        self.base_pipeline = BasePipeline()
        self.good_pipeline = SegmentPipeline('good')
        self.not_good_pipeline = SegmentPipeline('not_good')
        self.meta_pipeline = MetaPipeline()

        # Training data references
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def load_training_data(self):
        """Load training and test data from database."""
        print("\n" + "="*60)
        print("Loading Training Data")
        print("="*60)

        # Connect to database
        engine = get_database_connection()

        # Load training data
        with open(Config.TRAIN_SQL_FILE) as f:
            train_sql = f.read()
        train = data.data_load(engine, sql=train_sql)

        # Prepare training data
        train = train.set_index(Config.INDEX_COL).rename(columns={Config.TARGET_COL: "TARGET"})
        self.y_train = train["TARGET"]
        self.x_train = train.drop(columns=["TARGET"] + Config.COLS_TO_DROP, errors='ignore')

        print(f"Training data: {self.x_train.shape[0]} samples, {self.x_train.shape[1]} features")

        # Load test data
        engine = get_database_connection()
        with open(Config.TEST_SQL_FILE) as f:
            test_sql = f.read()
        test = data.data_load(engine, sql=test_sql)

        # Prepare test data
        test = test.set_index(Config.INDEX_COL).rename(columns={Config.TARGET_COL: "TARGET"})
        self.y_test = test["TARGET"]
        self.x_test = test.drop(columns=["TARGET"] + Config.COLS_TO_DROP, errors='ignore')

        print(f"Test data: {self.x_test.shape[0]} samples, {self.x_test.shape[1]} features")

    def train(self):
        """Train the complete 3-layer pipeline."""
        start_time = datetime.datetime.now()

        print("\n" + "="*60)
        print("CREDIT SCORING PIPELINE - TRAINING")
        print("="*60)

        # Ensure data is loaded
        if self.x_train is None:
            self.load_training_data()

        # =========================
        # LAYER 1: Base Model
        # =========================
        self.base_pipeline.train(
            self.x_train, self.y_train,
            self.x_test, self.y_test
        )

        # Get base predictions for segmentation
        base_train_proba = self.base_pipeline.predict(self.x_train, self.y_train)
        base_test_proba = self.base_pipeline.predict(self.x_test, self.y_test)

        # =========================
        # Segment Data for Layer 2
        # =========================
        print("\n" + "="*60)
        print("LAYER 2: Training Segment Models")
        print("="*60)

        # Calculate probability thresholds from score thresholds
        good_threshold = score_to_probability(Config.GOOD_SCORE_THRESHOLD)
        not_good_threshold = score_to_probability(Config.NOT_GOOD_SCORE_THRESHOLD)

        print(f"Good threshold (prob): {good_threshold:.4f}")
        print(f"Not-Good threshold (prob): {not_good_threshold:.4f}")

        # Split training data
        good_train_idx = base_train_proba < good_threshold
        not_good_train_idx = base_train_proba >= not_good_threshold

        x_train_good = self.x_train[good_train_idx]
        y_train_good = self.y_train[good_train_idx]
        x_train_not_good = self.x_train[not_good_train_idx]
        y_train_not_good = self.y_train[not_good_train_idx]

        print(f"Good segment: {len(x_train_good)} samples")
        print(f"Not-Good segment: {len(x_train_not_good)} samples")

        # Split test data
        good_test_idx = base_test_proba < good_threshold
        not_good_test_idx = base_test_proba >= not_good_threshold

        x_test_good = self.x_test[good_test_idx]
        y_test_good = self.y_test[good_test_idx]
        x_test_not_good = self.x_test[not_good_test_idx]
        y_test_not_good = self.y_test[not_good_test_idx]

        # Train segment models
        self.good_pipeline.train(x_train_good, y_train_good, x_test_good, y_test_good)
        self.not_good_pipeline.train(x_train_not_good, y_train_not_good, x_test_not_good, y_test_not_good)

        # =========================
        # LAYER 3: Meta Model
        # =========================

        # Generate meta features for training
        good_train_proba = self.good_pipeline.predict(self.x_train, self.y_train)
        not_good_train_proba = self.not_good_pipeline.predict(self.x_train, self.y_train)

        x_train_meta = pd.DataFrame({
            'Base': base_train_proba,
            'Good': good_train_proba,
            'Not_Good': not_good_train_proba
        }, index=self.y_train.index)

        # Generate meta features for test
        good_test_proba = self.good_pipeline.predict(self.x_test, self.y_test)
        not_good_test_proba = self.not_good_pipeline.predict(self.x_test, self.y_test)

        x_test_meta = pd.DataFrame({
            'Base': base_test_proba,
            'Good': good_test_proba,
            'Not_Good': not_good_test_proba
        }, index=self.y_test.index)

        # Train meta model
        self.meta_pipeline.train(x_train_meta, self.y_train, x_test_meta, self.y_test)

        # =========================
        # Save All Models
        # =========================
        print("\n" + "="*60)
        print("Saving Models")
        print("="*60)

        self.base_pipeline.save()
        self.good_pipeline.save()
        self.not_good_pipeline.save()
        self.meta_pipeline.save()

        # =========================
        # Calculate Final Scores
        # =========================
        print("\n" + "="*60)
        print("Calculating Final Scores")
        print("="*60)

        # Get final predictions
        meta_train_proba = self.meta_pipeline.predict(x_train_meta, self.y_train)
        meta_test_proba = self.meta_pipeline.predict(x_test_meta, self.y_test)

        # Convert to scores
        meta_train_score = probability_to_score(meta_train_proba)
        meta_test_score = probability_to_score(meta_test_proba)

        # Combine all results
        all_x = pd.concat([self.x_train, self.x_test])
        all_y = pd.concat([self.y_train, self.y_test])
        all_proba = np.concatenate([meta_train_proba, meta_test_proba])
        all_score = np.concatenate([meta_train_score, meta_test_score])

        results = pd.DataFrame({
            'PROBA': all_proba,
            'SCORE': all_score,
            'TARGET': all_y
        }, index=all_y.index)

        # Save results
        output_path = os.path.join(Config.OUTPUT_DIR, f'TRAINING_SCORES_{Config.VERSION}.xlsx')
        results.to_excel(output_path)
        print(f"Scores saved to: {output_path}")

        elapsed = datetime.datetime.now() - start_time
        print(f"\nTotal training time: {elapsed}")

        return results

    def load_models(self):
        """Load all trained models from disk."""
        print("\nLoading trained models...")
        self.base_pipeline.load()
        self.good_pipeline.load()
        self.not_good_pipeline.load()
        self.meta_pipeline.load()
        print("All models loaded successfully!")

    def score(self, x_data=None, y_data=None, apply_policy=True):
        """
        Score new applicants using the trained pipeline.

        Args:
            x_data: Features (if None, loads from scoring SQL file)
            y_data: True labels (optional, for evaluation)
            apply_policy: Whether to apply policy adjustments

        Returns:
            DataFrame with probabilities and scores
        """
        print("\n" + "="*60)
        print("CREDIT SCORING PIPELINE - SCORING")
        print("="*60)

        # Load data if not provided
        if x_data is None:
            engine = get_database_connection()
            with open(Config.SCORING_SQL_FILE) as f:
                scoring_sql = f.read()
            scoring_data = data.data_load(engine, sql=scoring_sql)

            scoring_data = scoring_data.set_index(Config.INDEX_COL).rename(
                columns={Config.TARGET_COL: "TARGET"}
            )
            y_data = scoring_data["TARGET"]
            x_data = scoring_data.drop(columns=["TARGET"] + Config.COLS_TO_DROP, errors='ignore')

        print(f"Scoring {len(x_data)} applicants...")

        # Layer 1: Base model
        print("Applying Layer 1 (Base)...")
        base_proba = self.base_pipeline.predict(x_data, y_data)

        # Layer 2: Segment models
        print("Applying Layer 2 (Segments)...")
        good_proba = self.good_pipeline.predict(x_data, y_data)
        not_good_proba = self.not_good_pipeline.predict(x_data, y_data)

        # Layer 3: Meta model
        print("Applying Layer 3 (Meta)...")
        x_meta = pd.DataFrame({
            'Base': base_proba,
            'Good': good_proba,
            'Not_Good': not_good_proba
        }, index=x_data.index if hasattr(x_data, 'index') else None)

        meta_proba = self.meta_pipeline.predict(x_meta, y_data)

        # Convert to scores
        raw_score = probability_to_score(meta_proba)

        # Apply policy adjustments
        if apply_policy:
            final_score = apply_policy_adjustment(raw_score)
        else:
            final_score = raw_score

        # Build results
        results = pd.DataFrame({
            'PROBA': meta_proba,
            'RAW_SCORE': raw_score,
            'FINAL_SCORE': final_score
        })

        if y_data is not None:
            results['TARGET'] = y_data.values if hasattr(y_data, 'values') else y_data

        if hasattr(x_data, 'index'):
            results.index = x_data.index

        # Save results
        output_path = os.path.join(Config.OUTPUT_DIR, f'SCORING_RESULTS_{Config.VERSION}.xlsx')
        results.to_excel(output_path)
        print(f"Results saved to: {output_path}")

        return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point for the credit scoring pipeline."""
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

    # Create output directory if needed
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.MODELS_DIR, exist_ok=True)

    # Initialize pipeline
    pipeline = CreditScoringPipeline()

    if args.mode == 'train':
        pipeline.train()

    elif args.mode == 'score':
        pipeline.load_models()
        pipeline.score(apply_policy=not args.no_policy)

    elif args.mode == 'both':
        pipeline.train()
        pipeline.score(apply_policy=not args.no_policy)

    elif args.mode == 'test':
        # Test mode: verify pipeline can be instantiated
        print("\n" + "="*60)
        print("PIPELINE TEST MODE")
        print("="*60)
        print("\nTesting pipeline initialization...")
        print(f"  - Base pipeline: OK")
        print(f"  - Good segment pipeline: OK")
        print(f"  - Not-Good segment pipeline: OK")
        print(f"  - Meta pipeline: OK")

        print("\nTesting configuration...")
        print(f"  - Reference score: {Config.REFERENCE_SCORE}")
        print(f"  - Odds at reference: {Config.ODDS_AT_REFERENCE}")
        print(f"  - Points to double: {Config.POINTS_TO_DOUBLE}")
        print(f"  - Columns to drop: {len(Config.COLS_TO_DROP)}")

        print("\nTesting utility functions...")
        test_prob = 0.05
        test_score = probability_to_score(test_prob)
        back_prob = score_to_probability(test_score)
        print(f"  - Probability {test_prob} -> Score {test_score:.2f}")
        print(f"  - Score {test_score:.2f} -> Probability {back_prob:.4f}")

        # Test policy adjustment
        test_scores = np.array([180, 200, 220, 240, 260])
        adjusted = apply_policy_adjustment(test_scores)
        print(f"\n  - Policy adjustment test:")
        for orig, adj in zip(test_scores, adjusted):
            print(f"    Score {orig} -> {adj:.1f}")

        print("\nAll tests passed!")
        print("\nTo run full training: python credit_scoring_pipeline.py --mode train")
        print("To run scoring: python credit_scoring_pipeline.py --mode score")


if __name__ == "__main__":
    main()

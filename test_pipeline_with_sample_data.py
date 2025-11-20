#!/usr/bin/env python3
"""
Test Credit Scoring Pipeline with Sample Data

This script tests the credit scoring pipeline using synthetic sample data
instead of requiring Oracle database access.

Usage:
    # First, generate sample data
    python generate_sample_data.py

    # Then run this test
    python test_pipeline_with_sample_data.py

This will:
1. Load sample CSV data
2. Train all 3 layers of the pipeline
3. Score new applicants
4. Generate performance metrics
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Import the pipeline
from credit_scoring_pipeline import (
    CreditScoringPipeline,
    Config,
    probability_to_score,
    score_to_probability
)


def load_sample_data():
    """Load sample data from CSV files."""
    print("="*60)
    print("Loading Sample Data")
    print("="*60)

    # Check if sample data exists
    train_path = 'Data/sample_train_data.csv'
    test_path = 'Data/sample_test_data.csv'
    score_path = 'Data/sample_score_data.csv'

    if not all(os.path.exists(p) for p in [train_path, test_path, score_path]):
        print("ERROR: Sample data files not found!")
        print("Please run: python generate_sample_data.py")
        sys.exit(1)

    # Load data
    print(f"Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path)

    print(f"Loading test data from {test_path}...")
    test_df = pd.read_csv(test_path)

    print(f"Loading scoring data from {score_path}...")
    score_df = pd.read_csv(score_path)

    print(f"\nData loaded successfully:")
    print(f"  Training: {train_df.shape}")
    print(f"  Test: {test_df.shape}")
    print(f"  Scoring: {score_df.shape}")

    return train_df, test_df, score_df


def prepare_data(df):
    """Prepare data for pipeline (same format as database loading)."""
    # Set index
    df = df.set_index('MUQAVILE')

    # Separate features and target
    y = df['TARGET']
    X = df.drop(columns=['TARGET'] + Config.COLS_TO_DROP, errors='ignore')

    return X, y


def test_pipeline():
    """Main test function."""
    print("\n" + "="*60)
    print("Bank of Baku Credit Scoring Pipeline - Test with Sample Data")
    print("="*60)
    print()

    # Load sample data
    train_df, test_df, score_df = load_sample_data()

    # Prepare data
    X_train, y_train = prepare_data(train_df)
    X_test, y_test = prepare_data(test_df)
    X_score, y_score = prepare_data(score_df)

    print(f"\nPrepared data:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"  X_score: {X_score.shape}, y_score: {y_score.shape}")

    # Initialize pipeline
    print("\n" + "="*60)
    print("Initializing Pipeline")
    print("="*60)
    pipeline = CreditScoringPipeline()

    # Manually set training data (bypassing database load)
    pipeline.x_train = X_train
    pipeline.y_train = y_train
    pipeline.x_test = X_test
    pipeline.y_test = y_test

    # Test Phase 1: Training
    print("\n" + "="*60)
    print("Phase 1: Training Pipeline")
    print("="*60)
    print("This will train all 3 layers...")
    print()

    try:
        results = pipeline.train()
        print("\n✓ Training completed successfully!")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test Phase 2: Evaluate Training Results
    print("\n" + "="*60)
    print("Phase 2: Training Evaluation")
    print("="*60)

    # Calculate metrics
    auc = roc_auc_score(results['TARGET'], results['PROBA'])
    gini = 2 * auc - 1

    print(f"\nPerformance Metrics:")
    print(f"  AUC:  {auc:.4f}")
    print(f"  Gini: {gini:.4f}")

    # Score distribution by target
    print(f"\nScore Distribution by Target:")
    print(results.groupby('TARGET')['SCORE'].describe())

    # Test Phase 3: Scoring New Applicants
    print("\n" + "="*60)
    print("Phase 3: Scoring New Applicants")
    print("="*60)

    try:
        scoring_results = pipeline.score(x_data=X_score, y_data=y_score, apply_policy=True)
        print("\n✓ Scoring completed successfully!")
    except Exception as e:
        print(f"\n✗ Scoring failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test Phase 4: Evaluate Scoring Results
    print("\n" + "="*60)
    print("Phase 4: Scoring Evaluation")
    print("="*60)

    print(f"\nScoring Results Summary:")
    print(scoring_results[['PROBA', 'RAW_SCORE', 'FINAL_SCORE']].describe())

    # Score distribution by target (if available)
    if 'TARGET' in scoring_results.columns:
        print(f"\nScoring AUC: {roc_auc_score(scoring_results['TARGET'], scoring_results['PROBA']):.4f}")

        # Score ranges
        bins = [0, 160, 180, 200, 220, 300]
        labels = ['High Risk', 'Poor', 'Fair', 'Good', 'Excellent']
        scoring_results['RISK_LEVEL'] = pd.cut(
            scoring_results['FINAL_SCORE'],
            bins=bins,
            labels=labels
        )

        print(f"\nRisk Distribution:")
        print(scoring_results['RISK_LEVEL'].value_counts().sort_index())

        print(f"\nRisk by Target:")
        print(pd.crosstab(scoring_results['RISK_LEVEL'], scoring_results['TARGET'], normalize='columns'))

    # Test Phase 5: Utility Functions
    print("\n" + "="*60)
    print("Phase 5: Testing Utility Functions")
    print("="*60)

    # Test probability to score conversion
    test_probs = [0.005, 0.01, 0.02, 0.04, 0.08]
    print("\nProbability to Score Conversions:")
    print("Probability | Score | Back to Prob")
    print("-" * 40)
    for prob in test_probs:
        score = probability_to_score(prob)
        back_prob = score_to_probability(score)
        print(f"{prob:11.4f} | {score:5.1f} | {back_prob:12.4f}")

    # Test segmentation thresholds
    print(f"\nSegmentation Thresholds:")
    good_prob = score_to_probability(Config.GOOD_SCORE_THRESHOLD,
                                     odds_at_ref=Config.ODDS_AT_REFERENCE_SEGMENTATION)
    not_good_prob = score_to_probability(Config.NOT_GOOD_SCORE_THRESHOLD,
                                         odds_at_ref=Config.ODDS_AT_REFERENCE_SEGMENTATION)
    print(f"  Good threshold (score {Config.GOOD_SCORE_THRESHOLD}): {good_prob:.4f}")
    print(f"  Not-Good threshold (score {Config.NOT_GOOD_SCORE_THRESHOLD}): {not_good_prob:.4f}")

    # Final Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print("✓ All phases completed successfully!")
    print()
    print("Generated Files:")
    print(f"  - Models/base_model_{Config.VERSION}.pkl")
    print(f"  - Models/good_model_{Config.VERSION}.pkl")
    print(f"  - Models/not_good_model_{Config.VERSION}.pkl")
    print(f"  - Models/meta_model_{Config.VERSION}.pkl")
    print(f"  - Output/TRAINING_SCORES_{Config.VERSION}.xlsx")
    print(f"  - Output/SCORING_RESULTS_{Config.VERSION}.xlsx")
    print()
    print("Performance Summary:")
    print(f"  Training Gini: {gini:.4f}")
    print(f"  Training samples: {len(results)}")
    print(f"  Scoring samples: {len(scoring_results)}")
    print()
    print("Next Steps:")
    print("  1. Review Excel files in Output/ directory")
    print("  2. Examine model files in Models/ directory")
    print("  3. Replace sample data with real data for production")
    print()


if __name__ == "__main__":
    test_pipeline()

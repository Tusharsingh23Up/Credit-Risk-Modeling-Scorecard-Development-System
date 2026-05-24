"""
run_scorecard.py
-----------------

This script provides an end‑to‑end demonstration of credit risk modelling
using Weight of Evidence (WoE), logistic regression and scorecard scaling.

When executed as a script it will perform the following steps:

1. Load the sample dataset from the ``data`` directory.
2. Specify which features are continuous and which are categorical.
3. Apply WoE transformations to prepare the modelling data.
4. Split the data into training and testing sets.
5. Fit a logistic regression model on the training data.
6. Evaluate the model on the testing set using accuracy and ROC‑AUC.
7. Generate a points‑based scorecard from the fitted model.
8. Score the full dataset and write the scores to ``reports/scores.csv``.

To run the script from the command line::

    python -m credit_risk_scorecard.src.run_scorecard

The script assumes it is being run from the top level of the repository and
will locate data and output directories relative to that location. Modify
``DATA_PATH`` and ``OUTPUT_PATH`` if you wish to override these defaults.
"""

import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .credit_scorecard import (
    load_data,
    prepare_woe_dataframe,
    train_logistic_regression,
    evaluate_model,
    build_scorecard,
)

import math


# Paths relative to project root
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "credit_risk_data.csv"
OUTPUT_PATH = REPO_ROOT / "reports"
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)


def main() -> None:
    # Step 1: Load data
    print(f"Loading data from {DATA_PATH}")
    df = load_data(str(DATA_PATH))

    # Step 2: Define features
    target = "default"
    continuous_feats = [
        "age",
        "income",
        "loan_amount",
        "loan_term_months",
        "credit_history_length",
        "number_of_dependents",
    ]
    categorical_feats = ["employment_status", "property_owned"]

    # Step 3: Apply WoE transformation
    df_woe, woe_maps, ivs = prepare_woe_dataframe(
        df, target, continuous_feats, categorical_feats, bins=10
    )
    print("Information Value (IV) of features:")
    for feat, iv_val in ivs.items():
        print(f"  {feat}: {iv_val:.4f}")

    # Separate features and target
    X = df_woe.drop(columns=[target])
    y = df_woe[target]

    # Step 4: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Step 5: Fit logistic regression
    model = train_logistic_regression(X_train, y_train)

    # Step 6: Evaluate model
    train_metrics = evaluate_model(model, X_train, y_train)
    test_metrics = evaluate_model(model, X_test, y_test)
    print("Model performance:")
    print(f"  Train Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"  Train AUC:      {train_metrics['auc']:.4f}")
    print(f"  Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Test AUC:       {test_metrics['auc']:.4f}")

    # Step 7: Build scorecard
    scorecard = build_scorecard(
        model,
        woe_maps,
        base_score=600,
        base_odds=50,
        pts_double_odds=20,
    )
    print("\nSample of scorecard (first two features):")
    # Display partial scorecard for brevity
    for i, (feat, mapping) in enumerate(scorecard.items()):
        print(f"{feat}: {list(mapping.items())[:5]}")
        if i >= 2:
            break

    # Step 8: Score the dataset
    # Compute full scores: intercept + sum(coeff * WOE) scaled
    factor = 20.0 / math.log(2)
    offset = 600 - factor * math.log(50)
    logits = model.intercept_[0] + X.dot(model.coef_[0])
    scores = offset + factor * logits
    df_scores = df.copy()
    df_scores['score'] = scores
    # Save scores
    scores_path = OUTPUT_PATH / "scores.csv"
    df_scores.to_csv(scores_path, index=False)
    print(f"\nScores saved to {scores_path}")


if __name__ == "__main__":
    main()
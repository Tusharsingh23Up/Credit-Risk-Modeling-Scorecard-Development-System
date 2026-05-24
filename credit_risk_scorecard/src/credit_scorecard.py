"""
credit_scorecard.py
--------------------

This module implements a simple end‑to‑end credit risk scoring pipeline using
weight of evidence (WoE) transformations, logistic regression and scorecard
generation. It provides reusable functions to prepare data, perform WoE
transformations, train a logistic regression model, evaluate performance, and
convert the fitted model into a point‑based scorecard.

The primary goal of this script is educational: it illustrates the
intermediate steps involved in building a transparent credit scorecard from
raw tabular data. It should not be used as a production solution without
additional validation, regulatory review and domain expertise.

Functions:
    load_data(path):
        Read a CSV dataset into a pandas DataFrame.

    compute_woe_iv(df, feature, target, bins=10):
        Compute Weight of Evidence (WoE) and Information Value (IV) for a
        given feature. For continuous features it performs quantile binning.

    prepare_woe_dataframe(df, target_column, continuous_features,
                          categorical_features, bins=10):
        Apply WoE transformation to all specified features and return a
        transformed DataFrame along with mapping dictionaries and IV values.

    train_logistic_regression(X, y):
        Fit a logistic regression model using scikit‑learn.

    evaluate_model(model, X, y):
        Compute accuracy and ROC‑AUC scores for a fitted model on a dataset.

    build_scorecard(model, woe_maps, base_score=600, base_odds=50,
                    pts_double_odds=20):
        Convert a fitted logistic regression model and WoE mappings into a
        points‑based credit scorecard.
"""

import math
from typing import Dict, Tuple, List, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score


def load_data(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Args:
        path: Path to the CSV file.

    Returns:
        A pandas DataFrame containing the dataset.
    """
    return pd.read_csv(path)


def _bin_continuous(series: pd.Series, bins: int) -> Tuple[pd.Series, List[float]]:
    """Helper to bin a continuous variable into quantiles.

    Args:
        series: Continuous pandas Series.
        bins: Number of quantile bins to create.

    Returns:
        Tuple of (binned_series, bin_edges).
    """
    # Drop duplicates in quantile edges to avoid errors when many duplicates exist
    quantiles = np.linspace(0, 1, bins + 1)
    edges = series.quantile(quantiles).unique()
    # Ensure unique, sorted edges
    edges = np.sort(edges)
    # pd.cut requires unique bin edges
    binned = pd.cut(series, bins=edges, include_lowest=True, duplicates='drop')
    return binned, edges.tolist()


def compute_woe_iv(
    df: pd.DataFrame,
    feature: str,
    target: str,
    bins: int = 10
) -> Tuple[Dict[Union[str, float], float], float, pd.Series]:
    """Compute Weight of Evidence (WoE) and Information Value (IV) for a feature.

    WoE measures how well each category or bin separates good (non‑default) and
    bad (default) observations. IV summarizes the overall predictive power of
    the feature. Higher IV values indicate stronger predictive power.

    For continuous variables the function performs quantile binning. For
    categorical variables the categories are used directly.

    Args:
        df: Input DataFrame containing the feature and target.
        feature: Name of the feature column to transform.
        target: Name of the binary target column (1 for default, 0 for good).
        bins: Number of quantile bins for continuous variables.

    Returns:
        A tuple of (woe_dict, iv, woe_series) where:
            woe_dict: mapping from bin/category to its WoE value.
            iv: Information Value of the feature.
            woe_series: pandas Series of the transformed feature values.
    """
    # Determine if feature is numeric or categorical
    if pd.api.types.is_numeric_dtype(df[feature]):
        # Perform quantile binning
        binned, _ = _bin_continuous(df[feature], bins)
        categories = binned
    else:
        categories = df[feature].astype(str)

    # Compute event (default) and non‑event counts per bin
    grouped = df.groupby(categories)[target]
    event = grouped.sum()
    non_event = grouped.count() - event

    total_event = df[target].sum()
    total_non_event = df[target].count() - total_event

    # Initialize dictionaries
    woe_dict: Dict[Union[str, float], float] = {}
    iv = 0.0

    # Compute WoE and IV per category
    for cat in event.index:
        # Avoid division by zero by adding a small constant
        distr_event = (event[cat] + 0.5) / (total_event + 0.5)
        distr_non_event = (non_event[cat] + 0.5) / (total_non_event + 0.5)
        woe = math.log(distr_non_event / distr_event)
        woe_dict[cat] = woe
        iv += (distr_non_event - distr_event) * woe

    # Create WoE transformed series
    woe_series = categories.map(woe_dict)

    return woe_dict, iv, woe_series


def prepare_woe_dataframe(
    df: pd.DataFrame,
    target_column: str,
    continuous_features: List[str],
    categorical_features: List[str],
    bins: int = 10
) -> Tuple[pd.DataFrame, Dict[str, Dict[Union[str, float], float]], Dict[str, float]]:
    """Transform selected features to their WoE representations.

    This function iterates over each feature, computes its WoE mapping and IV
    value, and applies the mapping to obtain a transformed DataFrame ready for
    logistic regression.

    Args:
        df: Raw DataFrame with both features and target.
        target_column: Name of the binary target column.
        continuous_features: List of continuous feature names.
        categorical_features: List of categorical feature names.
        bins: Number of quantile bins to use for continuous features.

    Returns:
        Tuple of (df_woe, woe_maps, iv_values) where:
            df_woe: DataFrame with WoE transformed features and the target.
            woe_maps: dictionary mapping feature names to WoE dictionaries.
            iv_values: dictionary mapping feature names to their IV values.
    """
    transformed = pd.DataFrame(index=df.index)
    woe_maps: Dict[str, Dict[Union[str, float], float]] = {}
    iv_values: Dict[str, float] = {}

    # Transform continuous features
    for feat in continuous_features:
        woe_map, iv, woe_series = compute_woe_iv(df[[feat, target_column]], feat, target_column, bins=bins)
        transformed[feat] = woe_series
        woe_maps[feat] = woe_map
        iv_values[feat] = iv

    # Transform categorical features
    for feat in categorical_features:
        woe_map, iv, woe_series = compute_woe_iv(df[[feat, target_column]], feat, target_column, bins=bins)
        transformed[feat] = woe_series
        woe_maps[feat] = woe_map
        iv_values[feat] = iv

    # Add target column
    transformed[target_column] = df[target_column]

    return transformed, woe_maps, iv_values


def train_logistic_regression(X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    """Fit a logistic regression model on the given data.

    Args:
        X: DataFrame of independent variables.
        y: Series of binary target labels.

    Returns:
        Fitted LogisticRegression model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def evaluate_model(model: LogisticRegression, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Evaluate a logistic regression model using accuracy and ROC‑AUC.

    Args:
        model: Fitted logistic regression model.
        X: Feature matrix.
        y: True labels.

    Returns:
        Dictionary containing accuracy and AUC scores.
    """
    pred_proba = model.predict_proba(X)[:, 1]
    pred_label = model.predict(X)
    accuracy = accuracy_score(y, pred_label)
    auc = roc_auc_score(y, pred_proba)
    return {"accuracy": accuracy, "auc": auc}


def build_scorecard(
    model: LogisticRegression,
    woe_maps: Dict[str, Dict[Union[str, float], float]],
    base_score: float = 600.0,
    base_odds: float = 50.0,
    pts_double_odds: float = 20.0
) -> Dict[str, Dict[Union[str, float], float]]:
    """Generate a points‑based scorecard from a logistic regression model.

    The scorecard maps each WoE bin/category to a point value such that
    increasing scores correspond to decreasing credit risk. The mapping uses
    common industry conventions of specifying a base score and base odds (the
    odds at the base score) and the number of points that double the odds.

    Args:
        model: Fitted logistic regression model (trained on WoE variables).
        woe_maps: Dictionary mapping each feature to its WoE mapping.
        base_score: Score assigned to an applicant with odds equal to base_odds.
        base_odds: Odds of non‑default at the base score. For example, 50 means
                   50:1 odds of good (non‑default) to bad (default).
        pts_double_odds: Number of scorecard points corresponding to doubling
                   the odds of non‑default.

    Returns:
        scorecard: Nested dictionary mapping each feature to a mapping from
                   bin/category to score points.
    """
    # Extract intercept and coefficients
    intercept = model.intercept_[0]
    coefficients = model.coef_[0]
    feature_names = model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else list(woe_maps.keys())

    # Compute scaling factors
    factor = pts_double_odds / math.log(2)
    offset = base_score - factor * math.log(base_odds)

    # Score associated with intercept
    intercept_points = offset + factor * intercept

    scorecard: Dict[str, Dict[Union[str, float], float]] = {}
    # Compute points for each feature/category
    for coeff, feat in zip(coefficients, feature_names):
        mapping = {}
        for cat, woe_value in woe_maps[feat].items():
            # Negative sign ensures that higher WoE (less risky) yields higher points
            points = -coeff * woe_value * factor
            mapping[cat] = round(points, 2)
        scorecard[feat] = mapping

    # Include intercept contribution under a special key
    scorecard['Intercept'] = {'Points': round(intercept_points, 2)}

    return scorecard
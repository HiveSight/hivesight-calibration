#!/usr/bin/env python3
"""Run calibration analysis on GPT-5 series results.

Trains logistic regression to predict actual GSS responses from:
- LLM predictions
- Demographics (age, income, region, sex, race, education)
- Political attributes (party, ideology, religion, attendance, marital)

Compares raw LLM accuracy vs calibrated accuracy using cross-validation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_results(filepath: Path) -> pd.DataFrame:
    """Load results from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    return pd.DataFrame(data["results"])


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepare features for calibration model.

    Returns:
        X: Feature matrix
        y: Target (actual response as binary: 1=first option, 0=second option)
        feature_names: List of feature names
    """
    # Filter to valid responses only
    valid_df = df[df["llm_parsed"].notna()].copy()

    # Encode categorical variables
    encoders = {}
    categorical_cols = ["region", "sex", "race", "education", "party", "ideology",
                        "religion", "attendance", "marital"]

    for col in categorical_cols:
        if col in valid_df.columns:
            # Fill NaN with "Unknown" for encoding
            valid_df[col] = valid_df[col].fillna("Unknown")
            encoders[col] = LabelEncoder()
            valid_df[f"{col}_enc"] = encoders[col].fit_transform(valid_df[col])

    # Binary: LLM predicted "Favor" (first option)?
    valid_df["llm_favor"] = (valid_df["llm_parsed"] == "Favor").astype(int)

    # Binary: Actual response is "Favor"?
    valid_df["actual_favor"] = (valid_df["actual_response"] == "Favor").astype(int)

    # Build feature matrix
    feature_cols = ["llm_favor", "age", "income"]

    # Add encoded categoricals
    for col in categorical_cols:
        enc_col = f"{col}_enc"
        if enc_col in valid_df.columns:
            feature_cols.append(enc_col)

    X = valid_df[feature_cols].values
    y = valid_df["actual_favor"].values

    return X, y, feature_cols, valid_df


def evaluate_calibration(X: np.ndarray, y: np.ndarray,
                         feature_names: list[str],
                         n_splits: int = 5) -> dict:
    """Evaluate calibration model with cross-validation.

    Returns dictionary with raw accuracy, calibrated accuracy, and improvement.
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Full calibrated model (LLM + all demographics) with regularization
    model_full = LogisticRegression(random_state=42, max_iter=1000, C=0.1)  # Strong regularization
    cv_scores_full = cross_val_score(model_full, X_scaled, y, cv=cv, scoring="accuracy")

    # LLM + key demographics only (party, ideology)
    key_demo_indices = [i for i, name in enumerate(feature_names)
                        if name in ["llm_favor", "party_enc", "ideology_enc"]]
    if len(key_demo_indices) > 1:
        X_key = X_scaled[:, key_demo_indices]
        model_key = LogisticRegression(random_state=42, max_iter=1000, C=0.1)
        cv_scores_key = cross_val_score(model_key, X_key, y, cv=cv, scoring="accuracy")
    else:
        cv_scores_key = np.array([0.0])

    # LLM-only model (just the LLM prediction - Platt scaling)
    X_llm_only = X_scaled[:, 0].reshape(-1, 1)  # Just llm_favor
    model_llm = LogisticRegression(random_state=42, max_iter=1000)
    cv_scores_llm = cross_val_score(model_llm, X_llm_only, y, cv=cv, scoring="accuracy")

    # Raw LLM accuracy (no calibration)
    raw_accuracy = (X[:, 0] == y).mean()

    # Fit full model to get feature importances
    model_full.fit(X_scaled, y)

    # Also compute "oracle" - what if we just flipped the LLM's bias?
    # Since LLM over-predicts Oppose, what accuracy if we just used base rates?
    base_rate_favor = y.mean()
    base_rate_accuracy = max(base_rate_favor, 1 - base_rate_favor)

    return {
        "raw_accuracy": raw_accuracy,
        "llm_calibrated_accuracy": cv_scores_llm.mean(),
        "llm_calibrated_std": cv_scores_llm.std(),
        "key_demo_calibrated_accuracy": cv_scores_key.mean(),
        "key_demo_calibrated_std": cv_scores_key.std(),
        "full_calibrated_accuracy": cv_scores_full.mean(),
        "full_calibrated_std": cv_scores_full.std(),
        "improvement_over_raw": cv_scores_full.mean() - raw_accuracy,
        "base_rate_accuracy": base_rate_accuracy,
        "actual_favor_rate": base_rate_favor,
        "llm_favor_rate": X[:, 0].mean(),
        "feature_importances": dict(zip(feature_names, model_full.coef_[0])),
        "n_samples": len(y),
    }


def main():
    results_dir = Path("results")

    # GPT-5 series files (skip 4o)
    files = {
        "GPT-5-mini": results_dir / "baseline_enhanced_cappun_gpt-5-mini_20251208_142141.json",
        "GPT-5": results_dir / "baseline_enhanced_cappun_gpt-5_20251208_145306.json",
    }

    print("=" * 60)
    print("CALIBRATION ANALYSIS: GPT-5 Series")
    print("=" * 60)
    print()

    all_results = {}

    for model_name, filepath in files.items():
        if not filepath.exists():
            print(f"Skipping {model_name}: file not found")
            continue

        print(f"\n--- {model_name} ---")

        df = load_results(filepath)
        X, y, feature_names, valid_df = prepare_features(df)

        results = evaluate_calibration(X, y, feature_names)
        all_results[model_name] = results

        print(f"Samples: {results['n_samples']}")
        print(f"Actual Favor rate:       {results['actual_favor_rate']:.1%}")
        print(f"LLM Favor rate:          {results['llm_favor_rate']:.1%}")
        print(f"Base rate accuracy:      {results['base_rate_accuracy']:.1%} (always predict majority)")
        print()
        print(f"Raw LLM accuracy:        {results['raw_accuracy']:.1%}")
        print(f"LLM + party/ideology:    {results['key_demo_calibrated_accuracy']:.1%} (+/- {results['key_demo_calibrated_std']*2:.1%})")
        print(f"Full calibrated (5-CV):  {results['full_calibrated_accuracy']:.1%} (+/- {results['full_calibrated_std']*2:.1%})")
        print(f"Improvement over raw:    {results['improvement_over_raw']:+.1%}")

        # Top feature importances
        print("\nTop feature importances:")
        importances = sorted(results['feature_importances'].items(),
                           key=lambda x: abs(x[1]), reverse=True)
        for name, coef in importances[:5]:
            print(f"  {name}: {coef:+.3f}")

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print()
    print(f"{'Model':<15} {'Raw':<10} {'Calibrated':<12} {'Improvement':<12}")
    print("-" * 49)

    for model_name, results in all_results.items():
        print(f"{model_name:<15} {results['raw_accuracy']:.1%}      "
              f"{results['full_calibrated_accuracy']:.1%}        "
              f"{results['improvement_over_raw']:+.1%}")

    print()
    print("Key insight: Calibration uses LLM prediction + demographics to predict")
    print("actual responses. Higher calibrated accuracy means the model learns")
    print("when to trust/override the LLM based on demographic patterns.")


if __name__ == "__main__":
    main()

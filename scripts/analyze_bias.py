#!/usr/bin/env python3
"""Analyze LLM bias patterns and theoretical calibration potential.

Key questions:
1. Where does the LLM disagree with actual responses?
2. Can we identify systematic bias patterns?
3. What's the ceiling for calibration improvement?
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats


def load_results(filepath: Path) -> pd.DataFrame:
    """Load results from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    return pd.DataFrame(data["results"])


def analyze_bias(df: pd.DataFrame, model_name: str):
    """Analyze where LLM predictions differ from actual responses."""
    valid = df[df["llm_parsed"].notna()].copy()

    # Create binary columns
    valid["actual_favor"] = (valid["actual_response"] == "Favor").astype(int)
    valid["llm_favor"] = (valid["llm_parsed"] == "Favor").astype(int)
    valid["correct"] = (valid["actual_response"] == valid["llm_parsed"]).astype(int)

    print(f"\n{'='*60}")
    print(f"BIAS ANALYSIS: {model_name}")
    print(f"{'='*60}")

    # Overall stats
    n = len(valid)
    actual_favor = valid["actual_favor"].mean()
    llm_favor = valid["llm_favor"].mean()
    accuracy = valid["correct"].mean()

    print(f"\nOverall (n={n}):")
    print(f"  Actual favor rate: {actual_favor:.1%}")
    print(f"  LLM favor rate:    {llm_favor:.1%}")
    print(f"  Bias:              {llm_favor - actual_favor:+.1%} (LLM under-predicts Favor)")
    print(f"  Accuracy:          {accuracy:.1%}")

    # Confusion matrix
    print("\nConfusion Matrix:")
    tp = ((valid["actual_favor"] == 1) & (valid["llm_favor"] == 1)).sum()
    tn = ((valid["actual_favor"] == 0) & (valid["llm_favor"] == 0)).sum()
    fp = ((valid["actual_favor"] == 0) & (valid["llm_favor"] == 1)).sum()
    fn = ((valid["actual_favor"] == 1) & (valid["llm_favor"] == 0)).sum()

    print(f"                    Predicted")
    print(f"                    Favor    Oppose")
    print(f"  Actual Favor      {tp:3d}      {fn:3d}")
    print(f"  Actual Oppose     {fp:3d}      {tn:3d}")

    # Key insight: False negatives (LLM says Oppose when actual is Favor)
    print(f"\nError types:")
    print(f"  False negatives (Favor→Oppose): {fn} ({fn/n:.1%})")
    print(f"  False positives (Oppose→Favor): {fp} ({fp/n:.1%})")

    # Analyze by key demographics
    print("\n--- Accuracy by Demographics ---")

    # By ideology
    if "ideology" in valid.columns:
        print("\nBy Ideology:")
        for ideology in sorted(valid["ideology"].dropna().unique()):
            subset = valid[valid["ideology"] == ideology]
            if len(subset) >= 3:
                acc = subset["correct"].mean()
                bias = subset["llm_favor"].mean() - subset["actual_favor"].mean()
                print(f"  {ideology:<25}: {acc:.0%} acc, {bias:+.0%} bias (n={len(subset)})")

    # By party
    if "party" in valid.columns:
        print("\nBy Party (simplified):")
        party_map = {
            "Strong Democrat": "Democrat",
            "Not very strong Democrat": "Democrat",
            "Independent, close to Democrat": "Lean Dem",
            "Independent": "Independent",
            "Independent, close to Republican": "Lean Rep",
            "Not very strong Republican": "Republican",
            "Strong Republican": "Republican",
        }
        valid["party_simple"] = valid["party"].map(party_map)
        for party in ["Democrat", "Lean Dem", "Independent", "Lean Rep", "Republican"]:
            subset = valid[valid["party_simple"] == party]
            if len(subset) >= 3:
                acc = subset["correct"].mean()
                actual = subset["actual_favor"].mean()
                llm = subset["llm_favor"].mean()
                print(f"  {party:<12}: {acc:.0%} acc | Actual {actual:.0%} Favor | LLM {llm:.0%} Favor")

    # Theoretical calibration ceiling
    print("\n--- Calibration Potential ---")

    # If we could perfectly correct the overall bias
    # What if we flipped LLM predictions based on overall bias?
    # The LLM predicts ~30% Favor, actual is ~60% Favor
    # If we flipped all Oppose→Favor, we'd get 100% Favor predictions = 60% accuracy
    # If we kept as-is, we get ~63% accuracy

    # Perfect bias correction per party
    total_correct_after = 0
    for party in valid["party"].dropna().unique():
        subset = valid[valid["party"] == party]
        actual_rate = subset["actual_favor"].mean()
        llm_rate = subset["llm_favor"].mean()

        # Perfect oracle: predict based on actual base rate for this group
        if actual_rate > 0.5:
            # Should predict all Favor
            correct_if_all_favor = subset["actual_favor"].sum()
            total_correct_after += correct_if_all_favor
        else:
            # Should predict all Oppose
            correct_if_all_oppose = (1 - subset["actual_favor"]).sum()
            total_correct_after += correct_if_all_oppose

    oracle_accuracy = total_correct_after / len(valid)
    print(f"  Oracle accuracy (perfect group bias correction): {oracle_accuracy:.1%}")
    print(f"  Current raw accuracy:                            {accuracy:.1%}")
    print(f"  Theoretical max improvement:                     {oracle_accuracy - accuracy:+.1%}")

    return {
        "model": model_name,
        "n": n,
        "accuracy": accuracy,
        "actual_favor_rate": actual_favor,
        "llm_favor_rate": llm_favor,
        "false_negatives": fn,
        "false_positives": fp,
        "oracle_accuracy": oracle_accuracy,
    }


def main():
    results_dir = Path("results")

    files = {
        "GPT-5-mini": results_dir / "baseline_enhanced_cappun_gpt-5-mini_20251208_142141.json",
        "GPT-5": results_dir / "baseline_enhanced_cappun_gpt-5_20251208_145306.json",
    }

    all_results = []
    for model_name, filepath in files.items():
        if filepath.exists():
            df = load_results(filepath)
            results = analyze_bias(df, model_name)
            all_results.append(results)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Calibration Potential")
    print("="*60)
    print(f"\n{'Model':<12} {'Raw Acc':<10} {'Oracle':<10} {'Gap':<10}")
    print("-" * 42)
    for r in all_results:
        print(f"{r['model']:<12} {r['accuracy']:.1%}      {r['oracle_accuracy']:.1%}      {r['oracle_accuracy']-r['accuracy']:+.1%}")

    print("\n'Oracle' = theoretical ceiling if we perfectly knew which groups")
    print("to override. Gap shows how much room for improvement exists.")
    print("\nKey finding: The LLM is systematically biased toward 'Oppose',")
    print("but it's hard to correct because the bias varies by demographic group.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run baseline calibration experiment with enhanced personas.

This script:
1. Loads GSS data for 2022-2024
2. Samples N respondents with valid cappun (death penalty) responses
3. Creates enhanced personas with political identity, religion, marital status
4. Queries GPT-5-mini for each persona
5. Compares LLM responses to actual GSS responses
6. Saves results for calibration analysis
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm.asyncio import tqdm

from hivesight_calibration import GSSLoader, LLMSurvey
from hivesight_calibration.persona import Persona
from hivesight_calibration.data import (
    ATTEND_MAP,
    DEGREE_MAP,
    MARITAL_MAP,
    PARTYID_MAP,
    POLVIEWS_MAP,
    RACE_MAP,
    REGION_MAP,
    RELIG_MAP,
    SEX_MAP,
)


def create_enhanced_persona(row: pd.Series) -> Persona:
    """Create a persona with all available demographic attributes."""
    return Persona(
        age=int(row["age"]),
        income=int(row["realinc"]) if pd.notna(row["realinc"]) else 0,
        region=REGION_MAP.get(int(row["region"]), "Unknown"),
        sex=SEX_MAP.get(int(row["sex"]), "Unknown"),
        race=RACE_MAP.get(int(row["race"]), "Unknown"),
        education=DEGREE_MAP.get(int(row["degree"]), "Unknown"),
        # Extended demographics
        party=PARTYID_MAP.get(int(row["partyid"])) if pd.notna(row.get("partyid")) else None,
        ideology=POLVIEWS_MAP.get(int(row["polviews"])) if pd.notna(row.get("polviews")) else None,
        religion=RELIG_MAP.get(int(row["relig"])) if pd.notna(row.get("relig")) else None,
        attendance=ATTEND_MAP.get(int(row["attend"])) if pd.notna(row.get("attend")) else None,
        marital=MARITAL_MAP.get(int(row["marital"])) if pd.notna(row.get("marital")) else None,
    )


async def main():
    # Configuration
    SAMPLE_SIZE = 100  # Start small to validate
    QUESTION_ID = "cappun"
    QUESTION_TEXT = "Do you favor or oppose the death penalty for persons convicted of murder?"
    RESPONSE_SCALE = ["Favor", "Oppose"]
    MODEL = "gpt-5-mini"  # Using GPT-5 series per user preference

    print(f"=== Enhanced Baseline Calibration Experiment ===")
    print(f"Question: {QUESTION_TEXT}")
    print(f"Model: {MODEL}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print(f"Enhanced persona attributes: party, ideology, religion, attendance, marital")
    print()

    # Load GSS data
    print("Loading GSS data...")
    gss = pd.read_stata("data/gss7224_r2.dta", convert_categoricals=False)
    gss = gss[gss["year"].isin([2022, 2024])]

    # Required columns for enhanced personas
    required_cols = [
        "year", "age", "realinc", "region", "sex", "race", "degree",
        "partyid", "polviews", "relig", "attend", "marital",
        QUESTION_ID
    ]

    # Get columns that exist in the data
    available_cols = [c for c in required_cols if c in gss.columns]

    # Filter to respondents with valid core demographics and question response
    core_cols = ["year", "age", "realinc", "region", "sex", "race", "degree", QUESTION_ID]
    gss_valid = gss[available_cols].dropna(subset=core_cols)
    print(f"Valid respondents (core demographics): {len(gss_valid):,}")

    # Sample respondents
    sample = gss_valid.sample(n=min(SAMPLE_SIZE, len(gss_valid)), random_state=42)
    print(f"Sampled: {len(sample)}")

    # Check how many have extended attributes
    has_party = sample["partyid"].notna().sum() if "partyid" in sample.columns else 0
    has_ideology = sample["polviews"].notna().sum() if "polviews" in sample.columns else 0
    has_religion = sample["relig"].notna().sum() if "relig" in sample.columns else 0
    print(f"  With party ID: {has_party}")
    print(f"  With ideology: {has_ideology}")
    print(f"  With religion: {has_religion}")

    # Map actual responses (GSS codes: 1=Favor, 2=Oppose)
    actual_responses = sample[QUESTION_ID].values
    actual_labels = ["Favor" if r == 1 else "Oppose" for r in actual_responses]

    # Initialize survey
    survey = LLMSurvey(model=MODEL)

    # Query LLM for each persona
    print(f"\nQuerying {MODEL} for {len(sample)} personas...")
    results = []
    total_cost = 0.0

    for i, (idx, row) in enumerate(tqdm(sample.iterrows(), total=len(sample))):
        try:
            persona = create_enhanced_persona(row)
            actual = actual_labels[i]

            response = await survey.query(
                persona=persona,
                question=QUESTION_TEXT,
                response_type="likert",
                scale=RESPONSE_SCALE,
            )

            results.append({
                "index": i,
                "age": persona.age,
                "income": persona.income,
                "region": persona.region,
                "sex": persona.sex,
                "race": persona.race,
                "education": persona.education,
                "party": persona.party,
                "ideology": persona.ideology,
                "religion": persona.religion,
                "attendance": persona.attendance,
                "marital": persona.marital,
                "actual_response": actual,
                "llm_raw": response.raw_response,
                "llm_parsed": response.parsed_response,
                "tokens_input": response.tokens_input,
                "tokens_output": response.tokens_output,
                "cost_usd": response.cost_usd,
            })
            total_cost += response.cost_usd

        except Exception as e:
            print(f"Error for row {i}: {e}")
            results.append({
                "index": i,
                "error": str(e),
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate accuracy
    valid_results = results_df[results_df["llm_parsed"].notna()]
    matches = (valid_results["actual_response"] == valid_results["llm_parsed"]).sum()
    accuracy = matches / len(valid_results) * 100

    print(f"\n=== Results ===")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Valid responses: {len(valid_results)}/{len(results_df)}")
    print(f"Accuracy: {accuracy:.1f}%")
    print()

    # Distribution comparison
    print("Actual GSS distribution:")
    print(valid_results["actual_response"].value_counts(normalize=True))
    print()
    print("LLM predicted distribution:")
    print(valid_results["llm_parsed"].value_counts(normalize=True))

    # Accuracy by party (if available)
    if "party" in valid_results.columns and valid_results["party"].notna().any():
        print("\nAccuracy by party:")
        for party in valid_results["party"].dropna().unique():
            party_data = valid_results[valid_results["party"] == party]
            party_acc = (party_data["actual_response"] == party_data["llm_parsed"]).mean() * 100
            print(f"  {party}: {party_acc:.1f}% (n={len(party_data)})")

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"baseline_enhanced_{QUESTION_ID}_{MODEL}_{timestamp}.json"

    output_data = {
        "metadata": {
            "question_id": QUESTION_ID,
            "question_text": QUESTION_TEXT,
            "model": MODEL,
            "sample_size": len(results_df),
            "total_cost_usd": total_cost,
            "accuracy": accuracy,
            "timestamp": timestamp,
            "enhanced_attributes": ["party", "ideology", "religion", "attendance", "marital"],
        },
        "results": results_df.to_dict(orient="records"),
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Also save as CSV for easy analysis
    csv_file = output_dir / f"baseline_enhanced_{QUESTION_ID}_{MODEL}_{timestamp}.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"CSV saved to: {csv_file}")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Run baseline calibration experiment.

This script:
1. Loads GSS data for 2022-2024
2. Samples N respondents with valid cappun (death penalty) responses
3. Creates personas matching their demographics
4. Queries GPT-4o-mini for each persona
5. Compares LLM responses to actual GSS responses
6. Saves results for calibration analysis
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm.asyncio import tqdm

from hivesight_calibration import GSSLoader, PersonaGenerator, LLMSurvey
from hivesight_calibration.data import DEGREE_MAP, RACE_MAP, REGION_MAP, SEX_MAP


async def main():
    # Configuration
    SAMPLE_SIZE = 100  # Start small to validate
    QUESTION_ID = "cappun"
    QUESTION_TEXT = "Do you favor or oppose the death penalty for persons convicted of murder?"
    RESPONSE_SCALE = ["Favor", "Oppose"]
    MODEL = "gpt-4o-mini"

    print(f"=== Baseline Calibration Experiment ===")
    print(f"Question: {QUESTION_TEXT}")
    print(f"Model: {MODEL}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print()

    # Load GSS data
    print("Loading GSS data...")
    loader = GSSLoader(data_dir=Path("data"))

    # Load with pandas directly for speed
    gss = pd.read_stata("data/gss7224_r2.dta", convert_categoricals=False)
    gss = gss[gss["year"].isin([2022, 2024])]

    # Filter to respondents with valid responses for our question
    required_cols = ["year", "age", "realinc", "region", "sex", "race", "degree", QUESTION_ID]
    gss_valid = gss[required_cols].dropna()
    print(f"Valid respondents: {len(gss_valid):,}")

    # Sample respondents
    sample = gss_valid.sample(n=min(SAMPLE_SIZE, len(gss_valid)), random_state=42)
    print(f"Sampled: {len(sample)}")

    # Generate personas
    generator = PersonaGenerator()
    personas = generator.from_dataframe(sample)
    print(f"Generated {len(personas)} personas")

    # Map actual responses (GSS codes: 1=Favor, 2=Oppose)
    actual_responses = sample[QUESTION_ID].values
    actual_labels = ["Favor" if r == 1 else "Oppose" for r in actual_responses]

    # Initialize survey
    survey = LLMSurvey(model=MODEL)

    # Query LLM for each persona
    print(f"\nQuerying {MODEL} for {len(personas)} personas...")
    results = []
    total_cost = 0.0

    for i, (persona, actual) in enumerate(tqdm(zip(personas, actual_labels), total=len(personas))):
        try:
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
                "actual_response": actual,
                "llm_raw": response.raw_response,
                "llm_parsed": response.parsed_response,
                "tokens_input": response.tokens_input,
                "tokens_output": response.tokens_output,
                "cost_usd": response.cost_usd,
            })
            total_cost += response.cost_usd

        except Exception as e:
            print(f"Error for persona {i}: {e}")
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

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"baseline_{QUESTION_ID}_{MODEL}_{timestamp}.json"

    output_data = {
        "metadata": {
            "question_id": QUESTION_ID,
            "question_text": QUESTION_TEXT,
            "model": MODEL,
            "sample_size": len(results_df),
            "total_cost_usd": total_cost,
            "accuracy": accuracy,
            "timestamp": timestamp,
        },
        "results": results_df.to_dict(orient="records"),
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Also save as CSV for easy analysis
    csv_file = output_dir / f"baseline_{QUESTION_ID}_{MODEL}_{timestamp}.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"CSV saved to: {csv_file}")


if __name__ == "__main__":
    asyncio.run(main())

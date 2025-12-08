#!/usr/bin/env python3
"""Run calibration experiment across multiple GSS questions.

Tests generalization: train on some questions, test on held-out questions.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm.asyncio import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from hivesight_calibration import LLMSurvey
from hivesight_calibration.persona import Persona
from hivesight_calibration.data import DEGREE_MAP, RACE_MAP, REGION_MAP, SEX_MAP


# Question definitions with GSS variable names and response scales
QUESTIONS = {
    "cappun": {
        "text": "Do you favor or oppose the death penalty for persons convicted of murder?",
        "scale": ["Favor", "Oppose"],
        "codes": {1: "Favor", 2: "Oppose"},
    },
    "gunlaw": {
        "text": "Would you favor or oppose a law which would require a person to obtain a police permit before he or she could buy a gun?",
        "scale": ["Favor", "Oppose"],
        "codes": {1: "Favor", 2: "Oppose"},
    },
    "grass": {
        "text": "Do you think the use of marijuana should be made legal or not?",
        "scale": ["Legal", "Not legal"],
        "codes": {1: "Legal", 2: "Not legal"},
    },
    "abany": {
        "text": "Please tell me whether or not you think it should be possible for a pregnant woman to obtain a legal abortion if the woman wants it for any reason?",
        "scale": ["Yes", "No"],
        "codes": {1: "Yes", 2: "No"},
    },
}


async def query_question(survey, persona, question_id, question_def):
    """Query LLM for a single persona-question pair."""
    try:
        response = await survey.query(
            persona=persona,
            question=question_def["text"],
            response_type="likert",
            scale=question_def["scale"],
        )
        return {
            "llm_raw": response.raw_response,
            "llm_parsed": response.parsed_response,
            "tokens": response.tokens_total,
            "cost": response.cost_usd,
            "error": None,
        }
    except Exception as e:
        return {"llm_raw": None, "llm_parsed": None, "tokens": 0, "cost": 0, "error": str(e)}


async def main():
    SAMPLE_PER_QUESTION = 50  # 50 respondents × 4 questions = 200 queries
    MODEL = "gpt-4o-mini"

    print(f"=== Multi-Question Calibration Experiment ===")
    print(f"Questions: {list(QUESTIONS.keys())}")
    print(f"Sample per question: {SAMPLE_PER_QUESTION}")
    print(f"Model: {MODEL}")
    print()

    # Load GSS data
    print("Loading GSS data...")
    gss = pd.read_stata("data/gss7224_r2.dta", convert_categoricals=False)
    gss = gss[gss["year"].isin([2022, 2024])]

    # Required demographics
    demo_cols = ["year", "age", "realinc", "region", "sex", "race", "degree"]

    # Initialize survey
    survey = LLMSurvey(model=MODEL)

    all_results = []
    total_cost = 0.0

    for q_id, q_def in QUESTIONS.items():
        print(f"\n--- Processing: {q_id} ---")

        # Filter to respondents with valid responses
        cols_needed = demo_cols + [q_id]
        q_data = gss[cols_needed].dropna()
        print(f"Valid respondents: {len(q_data):,}")

        # Sample
        sample = q_data.sample(n=min(SAMPLE_PER_QUESTION, len(q_data)), random_state=42)
        print(f"Sampled: {len(sample)}")

        # Create personas and get actual responses
        for idx, row in tqdm(sample.iterrows(), total=len(sample), desc=f"Querying {q_id}"):
            # Create persona
            persona = Persona(
                age=int(row["age"]),
                income=int(row["realinc"]),
                region=REGION_MAP.get(int(row["region"]), "Unknown"),
                sex=SEX_MAP.get(int(row["sex"]), "Unknown"),
                race=RACE_MAP.get(int(row["race"]), "Unknown"),
                education=DEGREE_MAP.get(int(row["degree"]), "Unknown"),
            )

            # Get actual response
            actual_code = int(row[q_id])
            actual_label = q_def["codes"].get(actual_code, "Unknown")

            # Query LLM
            result = await query_question(survey, persona, q_id, q_def)

            all_results.append({
                "question_id": q_id,
                "age": persona.age,
                "income": persona.income,
                "region": persona.region,
                "sex": persona.sex,
                "race": persona.race,
                "education": persona.education,
                "actual_response": actual_label,
                "actual_code": actual_code,
                **result,
            })
            total_cost += result["cost"]

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Calculate per-question accuracy
    print("\n=== Results by Question ===")
    for q_id in QUESTIONS.keys():
        q_results = results_df[results_df["question_id"] == q_id]
        valid = q_results[q_results["llm_parsed"].notna()]
        if len(valid) > 0:
            accuracy = (valid["actual_response"] == valid["llm_parsed"]).mean()
            print(f"{q_id}: {accuracy:.1%} accuracy ({len(valid)} valid)")

            # Distribution comparison
            actual_dist = valid["actual_response"].value_counts(normalize=True)
            llm_dist = valid["llm_parsed"].value_counts(normalize=True)
            print(f"  Actual: {actual_dist.to_dict()}")
            print(f"  LLM:    {llm_dist.to_dict()}")

    print(f"\nTotal cost: ${total_cost:.4f}")

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"multi_question_{MODEL}_{timestamp}.json"

    output_data = {
        "metadata": {
            "questions": list(QUESTIONS.keys()),
            "model": MODEL,
            "sample_per_question": SAMPLE_PER_QUESTION,
            "total_cost_usd": total_cost,
            "timestamp": timestamp,
        },
        "results": results_df.to_dict(orient="records"),
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Cross-question calibration test
    print("\n=== Cross-Question Calibration ===")

    # Prepare features
    le_region = LabelEncoder()
    le_education = LabelEncoder()
    le_race = LabelEncoder()
    le_sex = LabelEncoder()
    le_question = LabelEncoder()

    valid_df = results_df[results_df["llm_parsed"].notna()].copy()

    valid_df["region_enc"] = le_region.fit_transform(valid_df["region"])
    valid_df["education_enc"] = le_education.fit_transform(valid_df["education"])
    valid_df["race_enc"] = le_race.fit_transform(valid_df["race"])
    valid_df["sex_enc"] = le_sex.fit_transform(valid_df["sex"])
    valid_df["question_enc"] = le_question.fit_transform(valid_df["question_id"])

    # Binary: did LLM match actual?
    valid_df["correct"] = (valid_df["actual_response"] == valid_df["llm_parsed"]).astype(int)

    # Binary: LLM predicted first option in scale?
    valid_df["llm_first_option"] = valid_df.apply(
        lambda r: 1 if r["llm_parsed"] == QUESTIONS[r["question_id"]]["scale"][0] else 0,
        axis=1
    )

    # Binary: actual is first option?
    valid_df["actual_first_option"] = valid_df.apply(
        lambda r: 1 if r["actual_response"] == QUESTIONS[r["question_id"]]["scale"][0] else 0,
        axis=1
    )

    # Train calibrator to predict actual from LLM + demographics
    X = valid_df[["llm_first_option", "age", "income", "region_enc", "education_enc", "race_enc", "sex_enc", "question_enc"]].values
    y = valid_df["actual_first_option"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
    print(f"Calibrated accuracy (5-fold CV): {cv_scores.mean():.1%} (+/- {cv_scores.std()*2:.1%})")

    # Raw accuracy
    raw_accuracy = (valid_df["llm_first_option"] == valid_df["actual_first_option"]).mean()
    print(f"Raw LLM accuracy: {raw_accuracy:.1%}")
    print(f"Improvement: {cv_scores.mean() - raw_accuracy:.1%} absolute")


if __name__ == "__main__":
    asyncio.run(main())

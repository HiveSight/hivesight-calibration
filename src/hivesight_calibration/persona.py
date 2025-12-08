"""Persona generation from demographics."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from hivesight_calibration.data import DEGREE_MAP, RACE_MAP, REGION_MAP, SEX_MAP


@dataclass
class Persona:
    """A simulated survey respondent with demographic characteristics."""

    age: int
    income: int
    region: str
    sex: str
    race: str
    education: str

    def to_prompt(self) -> str:
        """Generate a natural language description for LLM prompts.

        Returns:
            A string describing the persona for use in system prompts.
        """
        income_str = f"${self.income:,}"

        return (
            f"You are a {self.age}-year-old {self.sex.lower()} living in the {self.region} "
            f"region of the United States. You identify as {self.race}. "
            f"Your highest level of education is a {self.education.lower()}. "
            f"Your household income is approximately {income_str} per year."
        )

    @classmethod
    def from_gss(cls, respondent: dict[str, Any]) -> "Persona":
        """Create a Persona from a GSS respondent dictionary.

        Args:
            respondent: Dictionary with GSS demographic codes.

        Returns:
            Persona instance with decoded demographics.
        """
        return cls(
            age=respondent["age"],
            income=respondent["income"],
            region=REGION_MAP.get(respondent["region"], "Unknown"),
            sex=SEX_MAP.get(respondent["sex"], "Unknown"),
            race=RACE_MAP.get(respondent["race"], "Unknown"),
            education=DEGREE_MAP.get(respondent["degree"], "Unknown"),
        )


class PersonaGenerator:
    """Generate Persona objects from GSS data."""

    def __init__(self) -> None:
        """Initialize the generator."""
        pass

    def from_dataframe(
        self,
        df: pd.DataFrame,
        drop_missing: bool = True,
    ) -> list[Persona]:
        """Generate Personas from a GSS DataFrame.

        Args:
            df: DataFrame with GSS demographic columns.
            drop_missing: If True, skip rows with missing critical demographics.

        Returns:
            List of Persona objects.
        """
        personas: list[Persona] = []

        # Required columns for persona generation
        required_cols = ["age", "realinc", "region", "sex", "race", "degree"]

        for _, row in df.iterrows():
            # Check for missing values in required columns
            if drop_missing:
                missing = any(
                    col not in row.index or pd.isna(row[col]) for col in required_cols
                )
                if missing:
                    continue

            try:
                persona = Persona(
                    age=int(row["age"]),
                    income=int(row["realinc"]) if pd.notna(row["realinc"]) else 0,
                    region=REGION_MAP.get(int(row["region"]), "Unknown"),
                    sex=SEX_MAP.get(int(row["sex"]), "Unknown"),
                    race=RACE_MAP.get(int(row["race"]), "Unknown"),
                    education=DEGREE_MAP.get(
                        int(row["degree"]) if pd.notna(row["degree"]) else -1, "Unknown"
                    ),
                )
                personas.append(persona)
            except (ValueError, TypeError):
                if not drop_missing:
                    continue

        return personas

    def sample(
        self,
        df: pd.DataFrame,
        n: int,
        stratify_by: list[str] | None = None,
        random_state: int | None = None,
    ) -> list[Persona]:
        """Sample personas from GSS data.

        Args:
            df: DataFrame with GSS data.
            n: Number of personas to sample.
            stratify_by: Optional columns to stratify sampling by.
            random_state: Random seed for reproducibility.

        Returns:
            List of sampled Persona objects.
        """
        rng = np.random.default_rng(random_state)

        if stratify_by:
            # Stratified sampling
            sampled_indices: list[int] = []
            groups = df.groupby(stratify_by)
            n_per_group = n // len(groups)

            for _, group in groups:
                if len(group) >= n_per_group:
                    idx = rng.choice(group.index, size=n_per_group, replace=False)
                else:
                    idx = group.index.tolist()
                sampled_indices.extend(idx)

            sampled_df = df.loc[sampled_indices[:n]]
        else:
            # Simple random sampling
            if len(df) >= n:
                sampled_df = df.sample(n=n, random_state=random_state)
            else:
                sampled_df = df

        return self.from_dataframe(sampled_df)

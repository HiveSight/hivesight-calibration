"""GSS data loading and processing."""

from pathlib import Path
from typing import Any

import pandas as pd
import pyreadstat


# GSS region codes to names
REGION_MAP: dict[int, str] = {
    1: "New England",
    2: "Middle Atlantic",
    3: "East North Central",
    4: "West North Central",
    5: "South Atlantic",
    6: "East South Central",
    7: "West South Central",
    8: "Mountain",
    9: "Pacific",
}

# GSS degree codes to names
DEGREE_MAP: dict[int, str] = {
    0: "Less than high school",
    1: "High school",
    2: "Associate/Junior college",
    3: "Bachelor's degree",
    4: "Graduate degree",
}

# GSS sex codes
SEX_MAP: dict[int, str] = {
    1: "Male",
    2: "Female",
}

# GSS race codes (simplified)
RACE_MAP: dict[int, str] = {
    1: "White",
    2: "Black",
    3: "Other",
}

# Common GSS opinion questions (Likert-style)
OPINION_QUESTIONS: list[str] = [
    # Capital punishment
    "cappun",  # Favor or oppose death penalty
    # Government spending
    "natspac",  # Space exploration
    "natenvir",  # Environment
    "natheal",  # Health
    "natcity",  # Big cities
    "natcrime",  # Crime
    "natdrug",  # Drug addiction
    "nateduc",  # Education
    "natrace",  # Improving conditions for Blacks
    "natarms",  # Military/defense
    "nataid",  # Foreign aid
    "natfare",  # Welfare
    "natsoc",  # Social security
    "natmass",  # Mass transportation
    "natpark",  # Parks and recreation
    "natchld",  # Childcare
    "natsci",  # Scientific research
    # Confidence in institutions
    "confinan",  # Banks and financial institutions
    "conbus",  # Major companies
    "conclerg",  # Organized religion
    "coneduc",  # Education
    "confed",  # Executive branch
    "conlabor",  # Organized labor
    "conpress",  # Press
    "conmedic",  # Medicine
    "contv",  # Television
    "conjudge",  # Supreme Court
    "consci",  # Scientific community
    "conlegis",  # Congress
    "conarmy",  # Military
    # Abortion
    "abany",  # Abortion for any reason
    "abdefect",  # Abortion if defect
    "abhlth",  # Abortion if health
    "abrape",  # Abortion if rape
    "abpoor",  # Abortion if poor
    "absingle",  # Abortion if single
    # Other attitudes
    "grass",  # Legalize marijuana
    "gunlaw",  # Gun permits
    "homosex",  # Homosexual relations
    "premarsx",  # Premarital sex
    "prayer",  # Prayer in schools
    "affrmact",  # Affirmative action
    "fepol",  # Women in politics
    "fechld",  # Working mother relationship with kids
    "fefam",  # Woman's place in home
    "fepresch",  # Preschool child suffers if mother works
]


class GSSLoader:
    """Load and process GSS microdata."""

    def __init__(self, data_dir: Path | str = Path("data")) -> None:
        """Initialize loader with data directory.

        Args:
            data_dir: Directory containing GSS data files.
        """
        self.data_dir = Path(data_dir)

    def load(
        self,
        filename: str = "gss7224_r2.dta",
        years: list[int] | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load GSS data from Stata file.

        Args:
            filename: Name of the Stata file.
            years: Optional list of years to filter to.
            columns: Optional list of columns to load.

        Returns:
            DataFrame with GSS data.
        """
        filepath = self.data_dir / filename

        # Read Stata file
        if columns:
            df, _ = pyreadstat.read_dta(filepath, usecols=columns)
        else:
            df, _ = pyreadstat.read_dta(filepath)

        # Convert to pandas DataFrame
        df = pd.DataFrame(df)

        # Filter by year if specified
        if years and "year" in df.columns:
            df = df[df["year"].isin(years)]

        return df

    def extract_respondent(self, row: pd.Series) -> dict[str, Any]:
        """Extract respondent data from a GSS row.

        Args:
            row: A single row from GSS DataFrame.

        Returns:
            Dictionary with demographics and responses.
        """
        # Extract demographics
        respondent: dict[str, Any] = {
            "age": int(row.get("age", 0)) if pd.notna(row.get("age")) else None,
            "income": int(row.get("realinc", 0)) if pd.notna(row.get("realinc")) else None,
            "region": int(row.get("region", 0)) if pd.notna(row.get("region")) else None,
            "sex": int(row.get("sex", 0)) if pd.notna(row.get("sex")) else None,
            "race": int(row.get("race", 0)) if pd.notna(row.get("race")) else None,
            "degree": int(row.get("degree", 0)) if pd.notna(row.get("degree")) else None,
            "year": int(row.get("year", 0)) if pd.notna(row.get("year")) else None,
        }

        # Extract opinion responses
        responses: dict[str, int] = {}
        for q in OPINION_QUESTIONS:
            if q in row.index and pd.notna(row[q]):
                responses[q] = int(row[q])

        respondent["responses"] = responses

        return respondent

    def get_question_text(self, question_id: str) -> str:
        """Get the full question text for a GSS variable.

        Args:
            question_id: GSS variable name (e.g., 'cappun').

        Returns:
            Full question text.
        """
        # TODO: Load from GSS codebook
        question_texts: dict[str, str] = {
            "cappun": "Do you favor or oppose the death penalty for persons convicted of murder?",
            "grass": "Do you think the use of marijuana should be made legal or not?",
            "gunlaw": "Would you favor or oppose a law which would require a person to obtain a police permit before he or she could buy a gun?",
            "abany": "Please tell me whether or not you think it should be possible for a pregnant woman to obtain a legal abortion if the woman wants it for any reason?",
        }
        return question_texts.get(question_id, f"GSS question: {question_id}")

    def get_response_scale(self, question_id: str) -> list[str]:
        """Get the response scale for a GSS question.

        Args:
            question_id: GSS variable name.

        Returns:
            List of response options.
        """
        # Common scales
        favor_oppose = ["Favor", "Oppose"]
        yes_no = ["Yes", "No"]
        agree_disagree = [
            "Strongly Agree",
            "Agree",
            "Neither",
            "Disagree",
            "Strongly Disagree",
        ]

        scales: dict[str, list[str]] = {
            "cappun": favor_oppose,
            "grass": ["Legal", "Not legal"],
            "gunlaw": favor_oppose,
            "abany": yes_no,
        }

        return scales.get(question_id, agree_disagree)

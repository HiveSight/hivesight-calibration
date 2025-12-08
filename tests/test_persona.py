"""Tests for persona generation."""

import pytest


class TestPersona:
    """Test Persona dataclass."""

    def test_persona_creation(self) -> None:
        """Can create a Persona with demographics."""
        from hivesight_calibration.persona import Persona

        persona = Persona(
            age=35,
            income=75000,
            region="Midwest",
            sex="Male",
            race="White",
            education="Bachelor's degree",
        )

        assert persona.age == 35
        assert persona.income == 75000
        assert persona.region == "Midwest"

    def test_persona_to_prompt(self) -> None:
        """Persona generates a natural language description for prompts."""
        from hivesight_calibration.persona import Persona

        persona = Persona(
            age=45,
            income=120000,
            region="Pacific",
            sex="Female",
            race="Asian",
            education="Graduate degree",
        )

        prompt = persona.to_prompt()

        assert "45" in prompt
        assert "120,000" in prompt or "120000" in prompt
        assert "Pacific" in prompt or "West Coast" in prompt
        assert "female" in prompt.lower()
        assert "graduate" in prompt.lower()

    def test_persona_from_gss_respondent(self) -> None:
        """Can create Persona from GSS respondent dict."""
        from hivesight_calibration.persona import Persona

        gss_respondent = {
            "age": 52,
            "income": 85000,
            "region": 5,  # South Atlantic
            "sex": 2,  # Female
            "race": 2,  # Black
            "degree": 1,  # High school
            "responses": {"cappun": 2},
        }

        persona = Persona.from_gss(gss_respondent)

        assert persona.age == 52
        assert persona.income == 85000
        assert persona.region == "South Atlantic"
        assert persona.sex == "Female"
        assert persona.race == "Black"
        assert persona.education == "High school"


class TestPersonaGenerator:
    """Test persona generation from GSS data."""

    def test_generate_from_gss(self) -> None:
        """Can generate personas from GSS respondent data."""
        from hivesight_calibration.persona import PersonaGenerator, Persona
        import pandas as pd

        # Mock GSS data
        gss_data = pd.DataFrame({
            "age": [25, 45, 65],
            "realinc": [30000, 75000, 50000],
            "region": [1, 5, 9],
            "sex": [1, 2, 1],
            "race": [1, 2, 3],
            "degree": [2, 3, 1],
        })

        generator = PersonaGenerator()
        personas = generator.from_dataframe(gss_data)

        assert len(personas) == 3
        assert all(isinstance(p, Persona) for p in personas)

    def test_generator_handles_missing_data(self) -> None:
        """Generator handles missing demographic values gracefully."""
        from hivesight_calibration.persona import PersonaGenerator
        import pandas as pd
        import numpy as np

        gss_data = pd.DataFrame({
            "age": [35, np.nan],
            "realinc": [50000, 60000],
            "region": [3, 7],
            "sex": [1, 2],
            "race": [1, 1],
            "degree": [np.nan, 2],
        })

        generator = PersonaGenerator()
        personas = generator.from_dataframe(gss_data, drop_missing=True)

        # Should drop rows with missing critical demographics
        assert len(personas) <= 2

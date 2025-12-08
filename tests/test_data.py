"""Tests for GSS data loading."""

import pytest
import pandas as pd
from pathlib import Path


class TestGSSLoader:
    """Test GSS data loading functionality."""

    def test_loader_initialization(self) -> None:
        """GSSLoader can be initialized with a data path."""
        from hivesight_calibration.data import GSSLoader

        loader = GSSLoader(data_dir=Path("data"))
        assert loader.data_dir == Path("data")

    def test_load_returns_dataframe(self, tmp_path: Path) -> None:
        """load() returns a pandas DataFrame."""
        from hivesight_calibration.data import GSSLoader

        # Create a minimal test stata file
        df = pd.DataFrame({"year": [2024], "age": [35], "income": [50000]})
        test_file = tmp_path / "test.dta"
        df.to_stata(test_file)

        loader = GSSLoader(data_dir=tmp_path)
        result = loader.load(filename="test.dta")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_filter_by_year(self, tmp_path: Path) -> None:
        """Can filter GSS data by year."""
        from hivesight_calibration.data import GSSLoader

        df = pd.DataFrame({
            "year": [2018, 2020, 2022, 2024],
            "age": [25, 35, 45, 55],
        })
        test_file = tmp_path / "test.dta"
        df.to_stata(test_file)

        loader = GSSLoader(data_dir=tmp_path)
        result = loader.load(filename="test.dta", years=[2022, 2024])

        assert len(result) == 2
        assert set(result["year"].unique()) == {2022, 2024}

    def test_get_opinion_questions(self) -> None:
        """Can retrieve list of opinion/attitude questions from GSS."""
        from hivesight_calibration.data import GSSLoader, OPINION_QUESTIONS

        # Should have pre-defined list of GSS opinion questions
        assert isinstance(OPINION_QUESTIONS, list)
        assert len(OPINION_QUESTIONS) > 0
        # Should include common GSS opinion vars
        assert "natspac" in OPINION_QUESTIONS or "cappun" in OPINION_QUESTIONS

    def test_extract_respondent(self, tmp_path: Path) -> None:
        """Can extract a single respondent with demographics and responses."""
        from hivesight_calibration.data import GSSLoader

        df = pd.DataFrame({
            "year": [2024],
            "age": [35],
            "realinc": [50000],
            "region": [3],
            "sex": [1],
            "race": [1],
            "degree": [3],
            "cappun": [1],  # Favor death penalty
        })
        test_file = tmp_path / "test.dta"
        df.to_stata(test_file)

        loader = GSSLoader(data_dir=tmp_path)
        data = loader.load(filename="test.dta")
        respondent = loader.extract_respondent(data.iloc[0])

        assert respondent["age"] == 35
        assert respondent["income"] == 50000
        assert "cappun" in respondent["responses"]


class TestDemographicMapping:
    """Test demographic code-to-text mapping."""

    def test_region_codes(self) -> None:
        """Region codes map to readable names."""
        from hivesight_calibration.data import REGION_MAP

        assert REGION_MAP[1] == "New England"
        assert REGION_MAP[4] == "West North Central"
        assert REGION_MAP[9] == "Pacific"

    def test_education_codes(self) -> None:
        """Education degree codes map to readable names."""
        from hivesight_calibration.data import DEGREE_MAP

        assert DEGREE_MAP[0] == "Less than high school"
        assert DEGREE_MAP[1] == "High school"
        assert DEGREE_MAP[3] == "Bachelor's degree"
        assert DEGREE_MAP[4] == "Graduate degree"

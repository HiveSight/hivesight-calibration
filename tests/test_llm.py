"""Tests for LLM survey functionality."""

import pytest
from unittest.mock import Mock, patch, AsyncMock


class TestLLMSurvey:
    """Test LLM survey querying."""

    def test_survey_initialization(self) -> None:
        """LLMSurvey can be initialized with model config."""
        from hivesight_calibration.llm import LLMSurvey

        survey = LLMSurvey(model="gpt-4o-mini")
        assert survey.model == "gpt-4o-mini"

    def test_build_system_prompt(self) -> None:
        """Builds appropriate system prompt for persona."""
        from hivesight_calibration.llm import LLMSurvey
        from hivesight_calibration.persona import Persona

        persona = Persona(
            age=35,
            income=60000,
            region="Midwest",
            sex="Male",
            race="White",
            education="High school",
        )

        survey = LLMSurvey(model="gpt-4o-mini")
        prompt = survey.build_system_prompt(persona)

        assert "35" in prompt
        assert "Midwest" in prompt
        assert "survey" in prompt.lower() or "respond" in prompt.lower()

    def test_build_question_prompt_likert(self) -> None:
        """Builds question prompt for Likert-scale questions."""
        from hivesight_calibration.llm import LLMSurvey

        survey = LLMSurvey(model="gpt-4o-mini")
        question = "Do you favor or oppose the death penalty for persons convicted of murder?"

        prompt = survey.build_question_prompt(
            question=question,
            response_type="likert",
            scale=["Strongly Favor", "Favor", "Oppose", "Strongly Oppose"],
        )

        assert question in prompt
        assert "Strongly Favor" in prompt
        assert "Strongly Oppose" in prompt

    @pytest.mark.asyncio
    async def test_query_returns_response(self) -> None:
        """query() returns structured response."""
        from hivesight_calibration.llm import LLMSurvey, LLMResponse
        from hivesight_calibration.persona import Persona

        persona = Persona(
            age=40,
            income=80000,
            region="Northeast",
            sex="Female",
            race="White",
            education="Bachelor's degree",
        )

        survey = LLMSurvey(model="gpt-4o-mini")

        # Mock the OpenAI client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="FAVOR\n\nI support this because..."))]

        with patch.object(survey, "_client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            result = await survey.query(
                persona=persona,
                question="Do you favor the death penalty?",
                response_type="likert",
                scale=["Favor", "Oppose"],
            )

        assert isinstance(result, LLMResponse)
        assert result.raw_response is not None

    def test_parse_likert_response(self) -> None:
        """Parses LLM output into Likert category."""
        from hivesight_calibration.llm import LLMSurvey

        survey = LLMSurvey(model="gpt-4o-mini")
        scale = ["Strongly Agree", "Agree", "Disagree", "Strongly Disagree"]

        # Test various response formats
        assert survey.parse_likert("AGREE", scale) == "Agree"
        assert survey.parse_likert("I strongly agree with this.", scale) == "Strongly Agree"
        assert survey.parse_likert("2", scale) == "Agree"  # Numeric response
        assert survey.parse_likert("Disagree\n\nBecause...", scale) == "Disagree"


class TestLLMResponse:
    """Test LLM response dataclass."""

    def test_response_creation(self) -> None:
        """LLMResponse stores response data."""
        from hivesight_calibration.llm import LLMResponse

        response = LLMResponse(
            raw_response="FAVOR\n\nI believe...",
            parsed_response="Favor",
            tokens_input=150,
            tokens_output=50,
            model="gpt-4o-mini",
        )

        assert response.parsed_response == "Favor"
        assert response.tokens_total == 200

    def test_response_cost_calculation(self) -> None:
        """LLMResponse calculates cost."""
        from hivesight_calibration.llm import LLMResponse

        response = LLMResponse(
            raw_response="test",
            parsed_response="Favor",
            tokens_input=1000,
            tokens_output=100,
            model="gpt-4o-mini",
        )

        # gpt-4o-mini: $0.15/1M input, $0.60/1M output
        expected_cost = (1000 * 0.15 / 1_000_000) + (100 * 0.60 / 1_000_000)
        assert abs(response.cost_usd - expected_cost) < 0.0001

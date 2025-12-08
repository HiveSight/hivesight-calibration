"""LLM survey querying."""

from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from hivesight_calibration.persona import Persona


# Pricing per 1M tokens (as of Dec 2024)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}


@dataclass
class LLMResponse:
    """Response from LLM survey query."""

    raw_response: str
    parsed_response: str | None
    tokens_input: int
    tokens_output: int
    model: str
    reasoning: str | None = None

    @property
    def tokens_total(self) -> int:
        """Total tokens used."""
        return self.tokens_input + self.tokens_output

    @property
    def cost_usd(self) -> float:
        """Estimated cost in USD."""
        pricing = MODEL_PRICING.get(self.model, {"input": 0.0, "output": 0.0})
        input_cost = self.tokens_input * pricing["input"] / 1_000_000
        output_cost = self.tokens_output * pricing["output"] / 1_000_000
        return input_cost + output_cost


class LLMSurvey:
    """Query LLMs with survey questions for simulated personas."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ) -> None:
        """Initialize the survey.

        Args:
            model: OpenAI model to use.
            api_key: Optional API key (uses OPENAI_API_KEY env var if not provided).
        """
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()

    def build_system_prompt(self, persona: Persona) -> str:
        """Build system prompt for a persona.

        Args:
            persona: The persona to simulate.

        Returns:
            System prompt string.
        """
        return f"""You are participating in a survey research study. You should respond as if you are a real person with the following characteristics:

{persona.to_prompt()}

When answering survey questions:
- Respond naturally based on your life experiences and perspective
- Consider how someone with your background might think about the issue
- Give your honest opinion - there are no right or wrong answers
- Be concise but thoughtful in your responses

You will be asked opinion questions. Respond with your answer category first, then optionally provide brief reasoning."""

    def build_question_prompt(
        self,
        question: str,
        response_type: str = "likert",
        scale: list[str] | None = None,
    ) -> str:
        """Build the question prompt.

        Args:
            question: The survey question text.
            response_type: Type of response expected ('likert' or 'open').
            scale: Response scale options for Likert questions.

        Returns:
            Question prompt string.
        """
        if response_type == "likert" and scale:
            scale_str = ", ".join(f'"{s}"' for s in scale)
            return f"""Question: {question}

Please respond with one of the following options: {scale_str}

Start your response with your chosen option, then optionally explain your reasoning briefly."""

        return f"""Question: {question}

Please provide your response."""

    async def query(
        self,
        persona: Persona,
        question: str,
        response_type: str = "likert",
        scale: list[str] | None = None,
    ) -> LLMResponse:
        """Query the LLM for a persona's response.

        Args:
            persona: The persona answering the question.
            question: The survey question.
            response_type: Type of response expected.
            scale: Response scale options.

        Returns:
            LLMResponse with the result.
        """
        system_prompt = self.build_system_prompt(persona)
        question_prompt = self.build_question_prompt(question, response_type, scale)

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question_prompt},
            ],
            max_tokens=200,
            temperature=1.0,
        )

        raw_content = response.choices[0].message.content or ""
        parsed = self.parse_likert(raw_content, scale) if response_type == "likert" else None

        return LLMResponse(
            raw_response=raw_content,
            parsed_response=parsed,
            tokens_input=response.usage.prompt_tokens if response.usage else 0,
            tokens_output=response.usage.completion_tokens if response.usage else 0,
            model=self.model,
        )

    def parse_likert(self, response: str, scale: list[str] | None) -> str | None:
        """Parse LLM response into a Likert category.

        Args:
            response: Raw LLM response text.
            scale: The response scale options.

        Returns:
            Matched scale category, or None if no match.
        """
        if not scale:
            return None

        response_lower = response.lower().strip()
        first_line = response.split("\n")[0].strip().lower()

        # Try matching first line exactly
        for option in scale:
            if option.lower() == first_line:
                return option

        # Try matching first line as prefix
        for option in scale:
            if first_line.startswith(option.lower()):
                return option

        # Sort scale by length descending to match longer options first
        # (e.g., "Strongly Agree" before "Agree")
        sorted_scale = sorted(scale, key=len, reverse=True)

        # Try exact match in first line
        for option in sorted_scale:
            if option.lower() in first_line:
                return option

        # Try numeric match (1, 2, 3, etc.)
        first_word = first_line.split()[0] if first_line else ""
        if first_word.isdigit():
            idx = int(first_word) - 1
            if 0 <= idx < len(scale):
                return scale[idx]

        # Try match in full response (sorted by length)
        for option in sorted_scale:
            if option.lower() in response_lower:
                return option

        return None

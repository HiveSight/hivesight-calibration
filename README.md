# HiveSight Calibration

GSS-calibrated LLM opinion simulation for accurate synthetic polling.

## Overview

This research package develops a calibration layer that corrects LLM response distributions to match real survey data from the General Social Survey (GSS). The goal is to enable:

- **Calibrated confidence intervals** for LLM-based opinion polling
- **Optimal model selection** balancing cost and accuracy
- **Defensible methodology** for synthetic survey research

## Research Questions

1. How well do LLM persona simulations match real survey responses?
2. Can we learn correction functions P(actual | LLM_response, demographics)?
3. What demographic features matter most for accurate simulation?
4. What's the optimal cost/quality tradeoff across models?

## Installation

```bash
# Clone the repo
git clone https://github.com/HiveSight/hivesight-calibration.git
cd hivesight-calibration

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev,docs]"
```

## Data

This research uses the General Social Survey (GSS) cumulative file:

1. Download from [GSS Data](https://gss.norc.org/us/en/gss/get-the-data/stata.html)
2. Place `gss7224_r2.dta` in the `data/` directory

## Usage

```python
from hivesight_calibration import GSSLoader, PersonaGenerator, LLMSurvey, Calibrator

# Load GSS data
loader = GSSLoader(data_dir="data")
gss_data = loader.load(years=[2022, 2024])

# Generate personas from GSS respondents
generator = PersonaGenerator()
personas = generator.from_dataframe(gss_data)

# Query LLM for each persona
survey = LLMSurvey(model="gpt-4o-mini")
# ... run queries ...

# Calibrate LLM responses to GSS ground truth
calibrator = Calibrator()
calibrator.fit(llm_responses, actual_responses, features)
```

## Methodology

### Phase 1: Baseline Calibration
- Extract GSS respondent demographics + opinion responses
- Run LLM with matched persona profiles
- Learn conditional distribution P(actual | LLM, demographics)
- Evaluate with CRPS and pinball loss

### Phase 2: Feature Selection
- Test which demographic attributes improve calibration
- Compare exact values vs. buckets
- Ablation studies

### Phase 3: Model Comparison
- Compare GPT-4o-mini, GPT-4o, Gemini, Claude
- Cost/accuracy Pareto frontier

### Phase 4: Production Integration
- Deploy calibrated predictions in HiveSight
- User-facing confidence intervals

## Evaluation Metrics

- **CRPS** (Continuous Ranked Probability Score)
- **Pinball loss** at quantiles (10th, 25th, 50th, 75th, 90th)
- **Coverage** of prediction intervals
- **Calibration plots** by demographic segment

## Related Work

- [How Many Human Survey Respondents is a Large Language Model Worth?](https://arxiv.org/abs/2502.17773)
- [TreatmentEffect.app](https://www.treatmenteffect.app/)
- [Expected Parrot EDSL](https://github.com/expectedparrot/edsl)

## License

MIT

# Methodology

## Overview

Our calibration pipeline consists of four phases:

```{mermaid}
graph LR
    A[GSS Data] --> B[Persona Generation]
    B --> C[LLM Querying]
    C --> D[Calibration Model]
    D --> E[Evaluation]
```

## Phase 1: Baseline Calibration

### Data Preparation

1. Load GSS individual-level microdata (not crosstabs)
2. For each GSS respondent:
   - Extract demographics: age, income, education, region, race, sex
   - Extract their actual Likert response to opinion questions
   - Build a persona prompt matching their profile

### LLM Querying

For each GSS respondent, we:
1. Construct a system prompt with their demographic persona
2. Ask the same question they answered in GSS
3. Parse the LLM response to the Likert scale

### Calibration Model

We learn the conditional distribution:

$$
P(\text{actual\_response} | \text{LLM\_response}, \text{demographics})
$$

Using multinomial logistic regression with features:
- LLM predicted response (one-hot encoded)
- Age (continuous)
- Income (continuous, log-transformed)
- Education (ordinal)
- Region (categorical)
- Race (categorical)
- Sex (binary)

### Train/Test Split

**Critical**: We split by **question**, not respondent:
- Train on questions Q1-Q10
- Test on held-out questions Q11-Q15

This tests generalization to new questions, not just new respondents.

## Phase 2: Feature Selection

We evaluate which demographic attributes improve calibration:

- Full model vs. reduced models (ablation study)
- Exact values vs. buckets (age=32 vs. "30s")
- Interaction effects (e.g., age × education)

## Phase 3: Model Comparison

Compare calibration quality across LLM providers:

| Model | Cost/Response | CRPS | Notes |
|-------|---------------|------|-------|
| GPT-4o-mini | $0.00015 | TBD | Baseline |
| GPT-4o | $0.003 | TBD | Higher quality |
| Gemini Flash | $0.00006 | TBD | Cheapest |
| Claude Haiku | $0.0001 | TBD | Alternative |

## Phase 4: Production Integration

Deploy calibrated predictions to HiveSight:
- Users specify desired confidence interval width
- System determines optimal model + sample size
- Return calibrated distribution, not point estimate

## Evaluation Metrics

1. **CRPS** - Overall probabilistic accuracy
2. **Pinball loss** - Quantile prediction accuracy
3. **Coverage** - Do X% intervals contain X% of actuals?
4. **Calibration plots** - Visual assessment by demographic segment

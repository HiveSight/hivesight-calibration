# Background

## LLM-based Survey Simulation

Recent work has demonstrated that LLMs can simulate human responses to survey questions when given demographic personas. Key findings from the literature:

- **[How Many Human Survey Respondents is a Large Language Model Worth?](https://arxiv.org/abs/2502.17773)** (Stanford, Feb 2025) - Establishes the statistical equivalence between LLM responses and human respondents, with specific exchange rates depending on task complexity.

- **[TreatmentEffect.app](https://www.treatmenteffect.app/)** - GPT-4 predicted 91% of variation in treatment effects across 70+ economics experiments, suggesting LLMs capture meaningful population-level variation.

- **[Expected Parrot EDSL](https://github.com/expectedparrot/edsl)** - MIT-licensed framework for running surveys with LLM agents, enabling standardized comparison across models.

## The General Social Survey (GSS)

The GSS is a nationally representative survey of US adults conducted since 1972. It provides:

- **Individual-level microdata** with detailed demographics
- **Consistent question wording** across decades
- **Opinion questions** on social, political, and economic issues
- **Ground truth** for calibrating LLM simulations

## Calibration Approaches

Our calibration methodology builds on probabilistic prediction frameworks:

### CRPS (Continuous Ranked Probability Score)

CRPS measures the quality of probabilistic predictions:

$$
\text{CRPS}(F, y) = \int_{-\infty}^{\infty} (F(x) - \mathbb{1}(x \geq y))^2 dx
$$

where $F$ is the predicted CDF and $y$ is the observed value.

### Pinball Loss

For quantile predictions at level $\tau$:

$$
L_\tau(q, y) = \begin{cases}
\tau (y - q) & \text{if } y \geq q \\
(1 - \tau)(q - y) & \text{if } y < q
\end{cases}
$$

## Related Commercial Applications

This research supports HiveSight's commercial synthetic polling product, where calibrated predictions enable:

- **Accurate opinion estimates** without expensive human surveys
- **Rapid iteration** on policy messaging and positioning
- **Demographic segmentation** at lower cost than traditional polling

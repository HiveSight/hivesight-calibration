# HiveSight Calibration

**GSS-calibrated LLM opinion simulation for accurate synthetic polling**

## Motivation

Large Language Models can simulate human survey responses by adopting demographic personas. However, raw LLM responses tend to:

- **Collapse to modal responses** - reduced variance compared to real populations
- **Amplify ideological differences** - overstate polarization on contentious issues
- **Miss demographic nuances** - fail to capture regional/generational variation

This research develops a **calibration layer** that learns the mapping from LLM predictions to true population distributions, enabling:

1. **Calibrated confidence intervals** for synthetic polling
2. **Optimal model selection** based on cost/accuracy tradeoffs
3. **Defensible methodology** for commercial survey research

## Research Questions

1. How well do LLM persona simulations match real survey responses from the GSS?
2. Can we learn correction functions P(actual | LLM_response, demographics)?
3. What demographic features matter most for accurate simulation?
4. What's the optimal cost/quality tradeoff across LLM providers?

## Key Results

*Results will be added as research progresses.*

## Contents

```{tableofcontents}
```

# Data Flywheel: The HiveSight Moat

## The Problem with Open Source Calibration

The calibration methodology (code) is necessarily open - academic papers need to be reproducible. But the **value isn't in the algorithm**, it's in:

1. **The fitted model weights** trained on proprietary data
2. **Domain-specific calibration** for your question types
3. **Continuous improvement** from production usage

## The Data Flywheel

```
User asks question → LLM generates responses → User validates (implicitly or explicitly)
                                    ↓
                    Calibration model improves
                                    ↓
                    Better predictions → More users → More data
```

### Phase 1: GSS Foundation (Now)
- Train on ~5,000 GSS respondents × ~50 questions = 250K response pairs
- Baseline calibration for general opinion questions
- Open source (establishes credibility)

### Phase 2: User Question Accumulation
Every HiveSight query adds to the training set:
- **Question text** (proprietary phrasing)
- **Demographic distribution requested** (user intent)
- **LLM responses** (model behavior on new questions)
- **Implicit validation** (users who re-run surveys, adjust, or convert to paid)

### Phase 3: Explicit Validation Loop
Add optional "ground truth" collection:
- "Did this match your expectations?" (quick feedback)
- Integration with real polling data (for enterprise clients)
- A/B test calibration vs. raw LLM (measure lift)

## What Expected Parrot Doesn't Have

Expected Parrot (YC W25) is building the **infrastructure** for LLM surveys. They're the "AWS" play.

HiveSight is building the **calibrated prediction** layer. You're the "OpenAI" play on top of their infrastructure.

| Aspect | Expected Parrot | HiveSight |
|--------|-----------------|-----------|
| Focus | Survey framework | Calibrated predictions |
| Data | None (tool provider) | Accumulating question/response pairs |
| Moat | OSS community | Proprietary calibration weights |
| Value prop | "Run surveys with LLMs" | "Get accurate answers" |

## Defensibility Checklist

- [ ] **Network effects**: Each user's questions improve predictions for all users
- [ ] **Switching costs**: Calibration is trained on HiveSight's question format
- [ ] **Data moat**: Historical question bank becomes training advantage
- [ ] **Domain expertise**: Specialized for policy/political questions (your niche)

## Implementation

### Short-term (This Research)
1. Publish GSS calibration paper → establishes methodology credibility
2. Open source the training code → builds trust, attracts researchers
3. Keep the fitted weights proprietary → the actual moat

### Medium-term (Product)
1. Log all user queries (with consent)
2. Track "re-run" behavior as implicit validation signal
3. A/B test calibrated vs. uncalibrated predictions

### Long-term (Enterprise)
1. Integrate with client's proprietary polling data
2. Custom calibration per client vertical (political, market research, etc.)
3. Real-time calibration updates as new data arrives

## Baseline Finding: Why Calibration Matters

From our initial experiment (N=100 GSS respondents):

| Metric | Raw LLM | Actual GSS |
|--------|---------|------------|
| Favor death penalty | 4% | 60% |
| Oppose death penalty | 96% | 40% |
| Accuracy | 42% | - |

**The LLM has a massive liberal bias.** Without calibration, HiveSight would give users wildly wrong answers. With calibration, we correct for this systematically.

This is the insight that competitors would need to discover independently. By the time they do, HiveSight has a year's worth of proprietary question data.

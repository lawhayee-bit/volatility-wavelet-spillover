# Submission Readiness Review

Date: 2026-03-20
Target journal: MDPI Applied Sciences
Recommended section: Computing and Artificial Intelligence
Manuscript file: `paper/manuscript/main.tex`

## 1. Review Goal

This review checks whether the current manuscript has reached a high submission standard for `Applied Sciences`, with particular attention to:

- journal fit and article positioning
- methodological completeness and reproducibility
- empirical rigor and benchmark quality
- figure/table quality and page-level presentation
- submission-package completeness

## 2. High-Level Verdict

Overall verdict: strong submission-ready draft, but not yet the final polished submission package.

Current status by dimension:

| Dimension | Status | Judgment |
|---|---|---|
| Journal fit | High | The paper is framed as an applied, reproducible ML + decision-support study rather than a pure finance-theory paper. |
| Methods | High | The manuscript now includes mathematical definitions, architecture figures, notation, pseudo-code, and leakage-free protocol details. |
| Experiments | High | Strong baselines, walk-forward evaluation, robustness checks, significance tests, and warning-task evaluation are all present. |
| Writing quality | Medium-High | The logic is clear and the claims are disciplined, but some transitions and result compression can still be improved. |
| Figure/table quality | Medium-High | Most major issues have been corrected, but a few figures still have room for final visual polishing and vector replacement. |
| Submission package | Medium | Author metadata, final repository statement, and cover letter still need a final pass. |

## 3. Fit to Applied Sciences

### 3.1 Why the fit is strong

The current version aligns well with the journal's applied-science orientation because it emphasizes:

- public-data reproducibility
- interpretable hybrid machine learning
- multiscale feature engineering
- decision-oriented risk warning
- leakage-free rolling evaluation

This is consistent with the official journal guidance that `Article` submissions should report scientifically sound experiments and provide enough detail for reproducibility:

- Applied Sciences Instructions for Authors: `https://www.mdpi.com/journal/applsci/instructions`
- Applied Sciences journal page: `https://www.mdpi.com/journal/applsci`
- Computing and Artificial Intelligence section page: `https://www.mdpi.com/journal/applsci/sections/computing_artificial_intelligence`

### 3.2 Current positioning that should be preserved

The paper should continue to present itself as:

`a reproducible public-data ML framework for stock index volatility forecasting and risk early warning`

The paper should avoid drifting toward:

- pure financial market theory
- asset-pricing interpretation
- weak-baseline "AI beats classical methods" rhetoric

## 4. Section-by-Section Assessment

### Title and Abstract

Status: strong

Strengths:

- The title is explicit about reproducibility, public data, multiscale wavelet features, hybrid machine learning, volatility forecasting, and risk early warning.
- The abstract uses a clear problem-method-results-implication structure.
- Main findings are stated cautiously and do not overclaim dominance over HAR.

Remaining improvements:

- A final pass can further tighten the last two sentences to reduce repetition around "public-data" and "interpretable" language.

### Introduction and Literature Positioning

Status: strong

Strengths:

- The introduction identifies a real application problem.
- The related-literature subsection already distinguishes four research streams.
- The paper clearly positions itself between volatility forecasting and warning systems.

Remaining improvements:

- The contribution list can still be sharpened into slightly more compact and parallel sentence structure.

### Data Section

Status: strong

Strengths:

- Public sources are explicit.
- The data-source table is appropriate for an application-oriented journal.
- The narrative emphasizes reproducibility and stable availability.

Remaining improvements:

- If space allows, add one sentence clarifying the exact sample freeze date for the final merged result packages.

### Methodology

Status: strong

Strengths:

- The section now includes notation, target construction, wavelet summaries, forecast calibration, warning labels, and pseudo-code.
- Two architecture figures make the system logic and the encoder-fusion-heads design easy to follow.
- The warning design is linked back to the forecasting output rather than presented as a disconnected classifier.

Remaining improvements:

- If a final polishing round is available, convert the two main PNG architecture figures to vector format.

### Experimental Design

Status: high

Strengths:

- Walk-forward protocol is explicit.
- Leakage avoidance is clearly stated.
- Metrics and statistical tests are formally defined.

Remaining improvements:

- The wording around retained versus omitted baselines can be made slightly more explicit in one sentence to pre-empt reviewer questions.

### Results

Status: high

Strengths:

- The results are organized around main effects, richer-data robustness, alternative targets, and warning performance.
- The paper does not hide the strength of HAR.
- Mechanism-oriented figures add explanatory depth rather than only showing benchmark scores.

Remaining improvements:

- Some figure-to-text transitions can still be shortened.
- A few captions can be made even more declarative and slightly less descriptive.

### Discussion and Conclusions

Status: medium-high to high

Strengths:

- The discussion correctly frames the contribution as "structured extension" rather than "universal replacement".
- Limitations are clearly acknowledged.
- Conclusions are aligned with the paper's applied-science orientation.

Remaining improvements:

- The final conclusion paragraph can still be shortened slightly to sound more decisive.

## 5. Empirical Rigor Check

Status: high

Checklist:

- [x] strict walk-forward evaluation
- [x] strong HAR benchmark retained
- [x] multiple horizons
- [x] robustness to richer public risk data
- [x] robustness to alternative OHLC volatility target
- [x] warning-task evaluation with imbalanced-class metrics
- [x] significance testing with DM and CW
- [x] discussion of where the proposed method does and does not win

Residual concern:

- The paper still needs one concise sentence in the main text clarifying that some broader baseline families were screened during development but are not emphasized in the final main comparison because stability and relevance were prioritized.

## 6. Figure and Table Readiness

Status: medium-high

Strengths:

- Architecture figures are now strong enough for a submission draft.
- Main result figures and mechanism figures have distinct roles.
- Tables use consistent highlighting and are easier to scan than before.

Remaining issues:

- Some figures are still PNG-based and should ideally become vector files before final submission.
- One or two plots may still benefit from slightly more whitespace around tick labels and annotations.
- Minor float positioning can still be improved in the results section.

## 7. Reference Quality

Status: medium-high

Strengths:

- Core volatility, warning-system, and Applied Sciences references are present.
- The bibliography is now broad enough for a proper article rather than a draft note.

Remaining issues:

- Two recently added cross-domain citations are currently cited appropriately in discussion, but their DOI metadata could not be independently resolved through DOI/Crossref lookup on 2026-03-20. Their bibliographic role should therefore remain supplementary rather than central until final metadata are confirmed.

## 8. Submission-Package Readiness

Status: medium

Still needed before final submission:

- real author names, affiliations, emails, ORCIDs
- finalized data/code availability wording with actual repository path or URL
- final cover letter
- final proofreading pass on captions and front matter

## 9. Priority Action Plan

### P0: Must-do before final submission

1. Replace placeholder author metadata.
2. Finalize repository / data-availability statement.
3. Convert key architecture figures to vector format if possible.
4. Run one final page-level PDF pass for float spacing and visual consistency.

### P1: Strongly recommended

5. Tighten contribution sentences in the introduction.
6. Compress several transitions in Results and Conclusions.
7. Harmonize caption style one last time.

### P2: Optional if time permits

8. Add one appendix sentence clarifying the broader screened baseline pool.
9. Add one sentence in the Data section stating the final result-freeze date.

## 10. Final Recommendation

The manuscript is already close to journal-submission quality and is strongly aligned with `Applied Sciences` if it continues to emphasize:

- reproducibility
- public-data design
- multiscale feature engineering
- hybrid ML with interpretable warning output
- disciplined comparison against strong baselines

Recommendation: proceed to final polishing rather than further expanding the empirical scope.

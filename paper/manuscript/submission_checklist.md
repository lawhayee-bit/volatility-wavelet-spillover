# Submission Checklist for *Risks* (MDPI)

> **Journal change**: Originally targeting *Applied Sciences*, then *Mathematics*; current target is *MDPI Risks*.
> Desk-rejected by Applied Sciences and Mathematics. The current revision is explicitly repositioned for a finance-risk journal.

## Journal-Specific Items (Risks)

- [x] Document class changed to `risks` in `main.tex`
- [x] Title reframed toward stock-index volatility forecasting and risk warning
- [x] Abstract rewritten toward finance-risk contribution instead of generalized methodology
- [x] Introduction repositioned for volatility forecasting, spillover, and early warning
- [x] Cover letter completely rewritten for *Risks*
- [x] Over-generalized cross-domain claims removed from the conclusion

## Manuscript Core

- [x] Author names, affiliations, ORCID, and correspondence email confirmed in `main.tex`
- [x] Title identical in manuscript and cover letter
- [x] Abstract updated to include the spillover result CW \(=4.83\)
- [ ] Confirm all cross-references resolve cleanly after the next full compile
- [ ] Confirm figure and table ordering after the next full compile

## Figures and Tables

- [x] Main visual palette already standardized across the manuscript
- [x] Table highlighting already uses manuscript-level color definitions
- [ ] Decide whether any secondary robustness tables should move to the appendix for a tighter main text
- [ ] Recheck figure/table page breaks under the `Risks` template after compilation

## References

- [x] Irrelevant cross-domain citations removed from the main conclusion
- [ ] Spot-check recent references and DOI fields before submission
- [ ] Rebuild the `.bbl` and check for missing or unused entries

## Reproducibility and Data

- [x] Public sources stated in the manuscript: Stooq and FRED
- [x] Data snapshot date stated in the manuscript
- [ ] Prepare the final reviewer package ZIP of scripts, configs, outputs, and figure builders
- [ ] Confirm final data-availability wording for the target journal submission form

## Final Technical Check

- [ ] Full PDF compile under the updated `Risks` version
- [ ] Visual review of title page, abstract page, figure placement, and appendix
- [ ] Freeze a clean submission PDF and cover letter PDF

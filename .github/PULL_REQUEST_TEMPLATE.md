<!--
Keep this short. Delete every section that does not apply rather than writing
"N/A". A PR that cannot answer "Validation" honestly is not ready.
-->

## What this changes

<!-- One or two sentences. The conclusion, not a diff summary. -->

Closes #

## Why

<!-- Cite the defect, the measurement, or the decision this follows from.
     If it follows from a review comment, link it. -->

## Validation

<!-- What you actually ran, on what hardware. Paste the outcome, not a claim
     that it passed. Delete rows you did not run. -->

| check | arch | result |
|---|---|---|
| `make test-cpu` | host | |
| `make ARCH=sm_86 run-probe` | sm_86 | |
| `bash scripts/run_r0_sm86.sh` | sm_86 | |
| ncu gate (`scripts/check_ncu_gate.sh`) | | |

**Denominator used:** <!-- the measured GB/s, and where it came from. If any
efficiency number here was computed against a spec-sheet peak, say so and
explain why. -->

## Numbers this moves

<!--
Any figure in README, docs/, PREREGISTRATION.md or results/ that changes.
State the before and after. If none, write "none" so a reader knows it was
checked rather than skipped.
-->

## Gates

<!-- Any gate in docs/GATES.md whose status this changes, and its new colour.
     Do not mark a gate green on sm_86 evidence alone. -->

## Scope

- [ ] One logical change
- [ ] No fused dequant / production kernel ahead of R0 (see CONTRIBUTING)
- [ ] Authoritative performance claims are sm_90 only
- [ ] Raw CSVs and env captures supporting any new number are committed
- [ ] `PREREGISTRATION.md` untouched, or appended to rather than rewritten

## Anything I am unsure about

<!-- Optional and encouraged. Cheaper to flag here than to have it found later. -->

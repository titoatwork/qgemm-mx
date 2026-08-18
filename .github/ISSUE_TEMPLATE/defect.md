---
name: Defect or measurement anomaly
about: Something is wrong in the code, the docs, the method, or a recorded number
title: ''
labels: ''
assignees: ''
---

<!--
Everything below is meant to be answerable from the repo. If a section does not
apply, delete it rather than writing "N/A". If you cannot fill in "How it was
found", that is a signal the finding is not yet solid enough to file.
-->

## What is wrong

<!-- One or two sentences. State the defect, not the symptom. -->

## Where

<!--
file:line, pinned to a commit, so the reference does not rot.
Example: include/qgemm/formats.cuh:104 at 2bc11ce
-->

```
```

## How it was found

<!--
A command, a derivation, or a repro. Not "I noticed". If the finding came from
reading rather than running, say so plainly and show the reasoning.
-->

## What it affects

Tick every one that applies, and delete the rest.

- [ ] A number already committed under `results/`
- [ ] A prediction in `PREREGISTRATION.md`
- [ ] A gate in `docs/GATES.md` (which: )
- [ ] A claim in `README.md` or `docs/`
- [ ] Nothing yet, latent until a later rung

**Does any recorded result or prediction become wrong if this is confirmed?**
<!-- Yes/no, and which. Preregistration is append-only, so a wrong prediction is
     corrected in Outcomes with a date, never rewritten. -->

## Architecture

- [ ] sm_86 only (dev box, non-authoritative)
- [ ] sm_90 (authoritative)
- [ ] Architecture independent

<!-- If a conclusion was drawn on sm_86, say whether it transfers. A 20-SM
     device with 1.5 MB of L2 hides wave quantization and L2 residency. -->

## Suggested fix

<!-- Optional. If you are not going to write it, say so, so it is clear the
     issue is a report rather than a claim on the work. -->

## How to verify the fix

<!-- The check that would fail before and pass after. A test, a command, or a
     re-derived number. If there is no such check, the fix is not verifiable and
     that should be said out loud. -->

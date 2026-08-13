# Contributing

## Commit style

- **One concern per commit** (e.g. `pack: MXFP4 host roundtrip tests`).
- **Message form:** `area: short description` — match existing history (`docs:`, `probe:`, `bench:`, `ci:`, …).
- Do **not** commit `.venv/`, `build/`, secrets, or large checkpoints.
- **Do** commit raw CSVs that support claims, env captures, and plot scripts.

## Preregistration

- Predictions live in `PREREGISTRATION.md` and are **append-only**.
- Commit predictions **before** any measurement they cover.
- Never rewrite past predictions; record outcomes in the Outcomes section with a date.

## Git / PR hygiene

- **No force-push to `main`** without explicit maintainer approval.
- Prefer `git pull --rebase origin main` before push when histories diverge.
- Keep PRs small and reviewable; one logical change when possible.

## R0-first

- **No fused dequant / production kernels** until the R0 measurement harness is proven and predictions are committed.
- Correctness order: structured probes (one-hot, primes, ramp) **before** random tolerance tests.
- Authoritative performance numbers are **sm_90** only; sm_86 is for harness bring-up and portability.
- Efficiency claims use **empirical** bandwidth as denominator, never the spec-sheet peak.

See `docs/MASTER_EXECUTION_PLAN.md`, `docs/PROTOCOL.md`, and `docs/GATES.md`.

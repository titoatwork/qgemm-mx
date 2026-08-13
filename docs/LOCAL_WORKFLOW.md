# Local workflow (no public firehose)

**Policy (from 2026-08):** development continues **locally**. Do **not** push every commit to GitHub — high commit frequency looks non-human and is unnecessary during heavy build-out.

## Rules

1. Work in `/home/titoisalive/Documents/projects/qgemm-mx` (or clone path).
2. **No `git push`** unless the owner explicitly asks for a curated release push.
3. Prefer **few, large local commits** (or uncommitted WIP) over many micro-commits.
4. When a public push is requested: squash/rebase into a short, human-paced history if needed.
5. Results CSVs and env logs stay in `results/` locally; cherry-pick what belongs in a public drop later.

## Still authoritative

- `docs/MASTER_EXECUTION_PLAN.md`
- `docs/PROJECT-v2.0-fp4-hopper.md`
- `PREREGISTRATION.md` (append outcomes only)

## Owner actions (external)

See `docs/GATES.md` and `docs/STATUS.md` — DGX, ncu, upstream demand.

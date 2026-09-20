# Contributing

## Branches & PRs
- Branch from `main` using your prefix: `p1/…` (pipeline), `p2/…` (model), `p3/…` (api), `p4/…` (dashboard).
- Keep PRs inside your own folder. A PR that touches `contracts/` needs approval from all four owners and a `CONTRACT_VERSION` bump.
- Squash-merge once CI is green.

## Local checks
```bash
uvx pre-commit install     # once
make lint                  # ruff, mypy, eslint, tsc
make test
```

## Commit messages
Use Conventional Commits with the module as the scope, for example `feat(api): add tile cache` or `fix(pipeline): handle all-nodata windows`.

## Daily rhythm
- 15-minute stand-up: yesterday, today, blockers. Contract changes are raised here.
- Integration happens on Days 13–14 (see [PROJECT_PLAN.md](PROJECT_PLAN.md)). Until then, work against `contracts/fixtures`.

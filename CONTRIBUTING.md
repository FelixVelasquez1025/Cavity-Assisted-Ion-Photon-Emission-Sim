# Contributing

## Development setup

Use Python 3.12 or newer and install the checkout with `uv sync`. The active
application is `src/caipe_sim/yb171_simulator.py`; the older physics and batch
modules are retained for comparison. Read [model status](docs/model-status.md)
before changing scientific conventions.

Run the relevant checks before submitting a change:

```bash
uv run pytest
uv run caipe-sim --demo
```

The default test suite lives in `tests/portfolio/`. Historical tests under
`tests/tests/` and `caipe_sim/` use obsolete APIs and are not part of the active
suite. When migrating a historical feature, replace its checks with tests that
exercise the current public behavior.

## Changes and review

Use a descriptive feature or fix branch and open a pull request for review.
Describe the resulting behavior, validation performed, and any changes to the
model's assumptions. Keep simulation outputs under `outputs/`; commit selected
illustrative figures under `docs/images/` with their generating command.

For physics changes, state the units, normalization, rotating-frame conventions,
and source of constants. Add a limiting-case calculation or independent
reference comparison when the change affects predicted dynamics. Numerical
population conservation alone does not establish physical accuracy.

Preserve attribution to the original project and contributors. Coordinate the
license and public-release scope with those contributors before publication.

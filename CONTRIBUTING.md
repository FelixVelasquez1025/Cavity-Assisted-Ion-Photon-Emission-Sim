# Contributing

## Development setup

Use Python 3.12 or newer and install the checkout with `uv sync`. The active
application is `src/caipe_sim/yb171_simulator.py`; historical physics and batch
modules are preserved under `archive/` and excluded from distribution/CI. Read [model status](docs/model-status.md)
before changing scientific conventions.

Run the relevant checks before submitting a change:

```bash
uv run pytest
uv run caipe-sim --demo
uv run python scripts/make_readme_figures.py
uv run python scripts/check_fock_truncation.py
```

The default test suite lives in `tests/portfolio/`. Historical tests under `archive/tests/` use obsolete APIs and are not
part of the active suite. When migrating a historical feature, replace its checks with tests that
exercise the current public behavior.

## Changes and review

Use a descriptive feature or fix branch and open a pull request for review.
Describe the resulting behavior, validation performed, and any changes to the
model's assumptions. Keep simulation outputs under `outputs/`; commit selected
illustrative figures under `docs/figures/` with their generating command.

For physics changes, state the units, normalization, rotating-frame conventions,
and source of constants. Add a limiting-case calculation or independent
reference comparison when the change affects predicted dynamics. Numerical
population conservation alone does not establish physical accuracy.

Preserve attribution to the original project and contributors. The repository
uses the MIT license in `LICENSE`; retain notices when distributing code.

# Contributing to crypto-quant

Thanks for your interest in **crypto-quant**! Contributions of all kinds are welcome: bug fixes, new experiments,
documentation improvements, reports of results (including negative ones), or simply questions.

By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).
New to crypto, machine learning or backtesting? A beginner-friendly explainer (in Vietnamese) is available at
[docs/GIAI-THICH.md](docs/GIAI-THICH.md).

## Reporting bugs and proposing changes

- Search the [issue tracker](https://github.com/the-daonm/crypto-quant/issues) first to avoid duplicates.
- Otherwise open a new issue using one of the templates: **Bug report**, **Feature request** or **Experiment**.
- Issues labelled [`good first issue`](https://github.com/the-daonm/crypto-quant/labels/good%20first%20issue) are a good place to start.

## Picking up an issue

Tasks are not pre-assigned: pick what interests you. Issues are grouped by `area:` labels
(`area: data`, `area: models`, `area: strategy`, `area: docs`, `area: infra`).

1. Find an **unassigned** issue and comment *"I'll take this"*. A maintainer will assign it to you.
2. Work on at most **two** assigned issues at a time, so tasks are not blocked.
3. Open a **draft pull request** early and link the issue (`Closes #N`), so others can see progress.
4. If you stop working on an issue, say so in a comment. Issues with no activity for a week may be unassigned.

## Development setup

First **fork** the repository on GitHub (button *Fork* at the top right), then:

```bash
git clone https://github.com/<your-username>/crypto-quant.git
cd crypto-quant
git remote add upstream https://github.com/the-daonm/crypto-quant.git
python -m venv .venv && source .venv/bin/activate              # fish: source .venv/bin/activate.fish
pip install -e ".[dev,notebook]"
pytest                                                         # should pass before you start
```

## Workflow

All changes go through **pull requests**; `main` is protected and cannot be pushed to directly.
Contributors open PRs from their forks. Maintainers may push branches to the main repository instead of a fork.

1. Sync your fork before starting:
   ```bash
   git checkout main
   git fetch upstream
   git merge --ff-only upstream/main
   git push origin main
   ```
2. Create a branch from `main` with a descriptive name: `feat/funding-rate`, `fix/backtest-fee`, `docs/sharpe`, `exp/ablation`.
3. Make your changes, add tests, and run the checks:
   ```bash
   ruff check . && ruff format --check .
   pytest
   ```
4. Commit (see the rules below), push the branch **to your fork** (`git push origin <branch>`), and open a
   **pull request** from your fork against `the-daonm/crypto-quant:main` using the template.
5. If `main` moves on while you work, update your branch: `git fetch upstream && git rebase upstream/main`, then `git push --force-with-lease`.
6. Every PR needs **at least one review** (approval or comments) from a team member other than the author, and CI must pass.
   Reviewers should run the code, not just read it. Anyone can review a PR on this public repository.
7. A maintainer merges the PR: **squash merge** when the PR has a single purpose, **rebase merge** when it contains several
   single-purpose commits. Either way the PR author stays the commit author.

## Commit guidelines

- **One commit, one purpose.** Never mix a bug fix, formatting changes and a new feature in one commit.
- Write the subject line in the imperative mood and keep it short (≤ 72 characters),
  e.g. `Add funding rate features`, `Fix fee double-count on exit`.
- If the subject needs an "and", consider splitting the commit.
- Each commit should build and pass the tests on its own.
- Commit with your **own GitHub account**; contributions are tracked through commits, PRs and reviews.

## Code guidelines

- Python ≥ 3.11, formatted with `ruff format` and linted with `ruff check` (configured in `pyproject.toml`).
- Code, comments, docstrings and user-facing strings are written in English.
- New code needs tests in `tests/`. **Tests must not access the network**: use synthetic data,
  see `_synthetic_ohlcv()` in `tests/test_features.py`.
- Never commit raw data (`data/`), results (`results/`) or secrets (API keys, …).

## Methodology rules (required for experiments)

This project evaluates trading strategies, so **sound methodology matters more than good-looking numbers**:

1. **No look-ahead.** A feature at bar t may only use data up to t, and must be covered by a test such as
   `test_features_do_not_use_future_data`.
2. **Select on the design period only** (before `holdout_start` in `config.yaml`). The holdout period is evaluated
   **once**, for the final configuration.
3. Always compare against **Buy & Hold** and report results **after fees**.
4. Record **how many configurations you tried**; this is needed for multiple-testing corrections.
5. Report failed experiments too (open a PR or comment on the issue); negative results are results.

## Releases

The project follows [Semantic Versioning](https://semver.org/) and records changes in [CHANGELOG.md](CHANGELOG.md)
using the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. PRs with notable changes should add an entry
under `[Unreleased]`.

## License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

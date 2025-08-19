# Contributing to Syntrix-Base

Thank you for your interest in contributing! This guide explains how to propose changes, report issues, and participate in the community.

## Table of Contents
- Getting Started
- Code of Conduct
- Development Environment
- Branching & Commits
- Coding Standards
- Testing Standards
- Documentation Standards
- CLI Standards
- Submitting Changes (PRs)
- Issue Reporting
- Release Process

## Getting Started
- Star and watch the repository to stay updated.
- Check open issues, especially those labeled “good first issue” and “help wanted”.
- Discuss larger ideas in a GitHub issue before investing significant effort.

## Code of Conduct
By participating, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Development Environment
1. Fork and clone the repo
2. Create a virtual environment and install in editable mode:
```bash
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -e .
```
3. Run tests to verify setup:
```bash
pytest -q
```
4. Optional: set default threading for reproducible CPU benchmarking:
```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
```

## Branching & Commits
- Create feature branches from `main`: `feat/<topic>`, `fix/<topic>`, `docs/<topic>`
- Use Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `ci:`, `chore:`
- Keep commits focused and atomic; reference issue numbers when applicable

## Coding Standards
Follow the project Coding Standards and general guidance:
- Snake_case for functions and variables; PascalCase for classes
- Type hints for public functions; docstrings where appropriate
- Prefer clarity over cleverness; early returns; meaningful names
- Determinism: use `set_seed()` and `set_threads()` from `syntrix.utils.seed`
- Performance: honor microbatching/grad-accum; consider `torch.compile` (optional)

## Testing Standards
- Add or update tests for all changes
- Ensure determinism tests remain green
- Run `pytest -q` locally; CI must pass before merge

## Documentation Standards
- Update `README.md` and `docs/*.md` when behavior or CLI changes
- Provide examples and command snippets for reproducibility
- Keep configuration examples in `configs/` up-to-date

## CLI Standards
- Provide descriptive help text and defaults
- Validate input files and parameter ranges
- Support config overrides and dot-notation where applicable

## Submitting Changes (PRs)
1. Ensure tests pass locally: `pytest -q`
2. Run linters/formatters if applicable (e.g., black/ruff)
3. Update docs and changelog entries if user-facing changes
4. Open a PR with a clear title and description including:
   - Motivation and context
   - Summary of changes
   - Tests and benchmarking evidence
   - Breaking changes and migration notes
5. Respond to review feedback promptly

## Issue Reporting
- Use the issue templates (bug report/feature request) if available
- Include environment details (OS, Python, PyTorch versions)
- Provide reproduction steps, logs, and expected vs. actual behavior

## Release Process
- Maintainers tag releases with semantic versioning (e.g., v0.1.0)
- Update `pyproject.toml` version and `CHANGELOG.md`
- Build and upload to PyPI when appropriate

## Acknowledgments
Thanks to all contributors and the open-source community for inspiration and prior art.

# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2025-08-19
### Added
- CLI UX: improved help, `-v/--verbose` levels; new `syntrix.eval` and `syntrix.config` entry points
- Config: dot-notation overrides (e.g., `--model.n_layer`, `--train.batch_size`) and validation with clear errors
- Logging: `elapsed_s` per eval log; initial `env` includes `git_commit`
- Developer tooling: pre-commit with `ruff` and `ruff-format`; CodeQL workflow
- Documentation: expanded `README` with badges and publishing guide; `docs/architecture.md` with equations, references, Troubleshooting/FAQ
- Governance: SECURITY policy, Issue/PR templates

### Tests
- CLI help coverage; parameter-count scaling tests; gradient-flow tests

### Fixes
- Resolved compile validation placement and linter issues in `Trainer`

## [0.1.0] - 2025-08-19
### Added
- CLI `--compile` with `--compile.validate` and `--compile.auto` for optional `torch.compile` speedups with throughput validation
- Dtype-aware tolerance helpers applied across tests
- Comprehensive CPU benchmarks in `docs/benchmarks.md`
- GitHub Actions CI for Python 3.10/3.11/3.12
- CONTRIBUTING.md and CODE_OF_CONDUCT.md; README expanded with detailed Quickstart and guidance

### Fixed
- Corrected `try_compile` implementation and environment fallback

### Notes
- `runs/` is gitignored; logs and checkpoints are written per `--out_dir`

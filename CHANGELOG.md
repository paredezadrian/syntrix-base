# Changelog

All notable changes to this project will be documented in this file.

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

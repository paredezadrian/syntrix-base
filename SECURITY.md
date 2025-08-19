# Security Policy

We take the security of Syntrix-Base seriously. This document explains how to report vulnerabilities and what to expect from us.

## Supported Versions

We provide security updates for the following versions:

- Main branch (active development)
- Latest minor release (currently v0.1.x)

Older versions may receive fixes at the maintainers’ discretion.

## Reporting a Vulnerability

- Do not open a public GitHub issue for security vulnerabilities.
- Use GitHub Security Advisories to report privately: https://github.com/paredezadrian/syntrix-base/security/advisories/new
- Alternatively, you can contact the maintainers via the email listed in the repository metadata.

Please include the following:
- Affected versions/commit hashes and environment (OS, Python, PyTorch)
- Reproduction steps or proof-of-concept
- Impact assessment (confidentiality/integrity/availability)
- Any known mitigations or workarounds

## Coordinated Disclosure

- A maintainer will acknowledge your report within 72 hours.
- We will investigate, develop a fix, and prepare a coordinated release.
- We aim to release patches within 14 days of report acknowledgement, depending on severity and complexity.
- Credit will be given in the changelog unless you request anonymity.

## Severity Classification (Guidance)

- Critical: RCE, supply-chain compromise, credential exfiltration
- High: Privilege escalation, arbitrary file write/read, DoS by default inputs
- Medium: Information disclosure, limited DoS on unusual inputs
- Low: Best-practice deviations, hardening opportunities

## Scope

In scope:
- Package source, training/evaluation scripts, configuration handling
- Release artifacts and packaging (PyPI, tags)

Out of scope:
- Third-party dependencies (report upstream)
- User-specific deployments or modifications

## Security Best Practices (For Users)

- Pin dependencies with hashes when possible
- Run training jobs with least privilege
- Keep Python and PyTorch updated
- Validate data sources; do not execute untrusted configs/scripts

Thank you for helping keep Syntrix-Base and its users safe.

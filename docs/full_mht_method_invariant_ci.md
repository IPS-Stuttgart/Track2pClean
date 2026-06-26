# FullMHT Method-Invariant CI, 2026-06-26

The branch has a focused GitHub Actions workflow:

```text
.github/workflows/full-mht-method-invariants.yml
```

It runs the lightweight Python 3.12 checks that protect the paper-facing method story:

- frozen FullMHT manifest row composition;
- identity-history, history-dynamics, terminal-completion, local-context, prior-survival, no-prior-continuation, and growth-history decision helpers;
- promotion gates for sensitivity and label-free exposure;
- no-GT leakage source scan;
- constructed MHT-vs-greedy conflict witnesses;
- exposure-audit parsing and aggregate gates.

This workflow is intentionally not the benchmark bundle. It does not load the Track2p data, does not compute manual-GT F1, and does not prove promotion. It only establishes that the method-layer invariants and interpretation gates still run in a clean Python 3.12 environment.

The server-side promotion bundle remains the authority for the method claim:

```text
docs/full_mht_identity_history_validation.md
```

A promotable FullMHT row still requires manifest-level real-data evidence, stable sensitivity, bounded label-free exposure, and recorded no-GT test results.

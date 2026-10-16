# MI50 evidence

This directory contains compact, publication-safe evidence for the gfx906 branch. Raw model responses, logs,
service configuration, host identifiers, credentials, and private infrastructure paths are intentionally excluded.

- `gfx906-q2-lds-gate-up-summary.json`: accepted Q2_0 performance and repeated quality comparison.
- `gfx906-deep-optimization-summary.json`: Top-K correctness, QFUSE rejection, and two-plane dot8 rejection.
- `source-manifest.json`: exact upstream base and LF-normalized SHA-256 values of the tested source snapshot.

Performance acceptance requires a reproducible median improvement above 3%, focused parity, and no deterministic
quality loss. Correctness fixes may be retained without a speed claim.

# MI50 evidence

This directory contains compact, publication-safe evidence for the gfx906 branch. Raw model responses, logs,
service configuration, host identifiers, credentials, and private infrastructure paths are intentionally excluded.

- `gfx906-q2-lds-gate-up-summary.json`: accepted Q2_0 performance and repeated quality comparison.
- `gfx906-deep-optimization-summary.json`: Top-K correctness, QFUSE rejection, and two-plane dot8 rejection.
- `2026-10-19-wave64-candidate-ab.json`: logical-wave32 persistent hyper-connection parity, microbench gains, end-to-end rejection, and router/compiler negative controls.
- `2026-10-19-hcp-off.json`, `2026-10-19-hcp-on-prewarm3.json`, `2026-10-19-hcp-on.json`: raw 256-token A/B samples for that persistent-kernel experiment.
- `2026-10-19-qwen3-coder-30b-a3b-q4.json`: full-VRAM llama.cpp fallback-model measurements, including short, 4K, and 16K context decode.
- `2026-10-19-qwen36-35b-a3b-ud-q4-k-xl.json`: production Qwen3.6/MTP measurements, context scaling, MTP-depth A/B, quality gates, and service state.
- `2026-10-19-qwen36-production.md`: readable deployment and benchmark report for the selected 72 GiB RAM / MI50 configuration.
- `source-manifest.json`: exact upstream base and LF-normalized SHA-256 values of the tested source snapshot.

Performance acceptance requires a reproducible median improvement above 3%, focused parity, and no deterministic
quality loss. Correctness fixes may be retained without a speed claim.

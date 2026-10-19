# Qwen3.6 production selection on MI50

Date: 2026-10-19

## Outcome

`Qwen3.6-35B-A3B-UD-Q4_K_XL` is now the selected production model for the 32 GiB AMD Instinct MI50 and 72 GiB system-RAM host. The model is served by Strata's native Qwen3.6 HIP engine with two MTP drafts and a 32,768-token context. Every routed expert fits in VRAM.

The production 10-run, 256-token, no-thinking decode result was **81.55 tok/s median** (74.8 minimum, 88.4 maximum). A 1,024-token continuous generation ran at **79.5 tok/s**. This exceeds the 50 tok/s target with enough margin for ordinary short and medium conversations.

The previous Qwen3.8-Flash-Next IQ3_S reference median was 31.6815 tok/s. The measured llama.cpp fallback, Qwen3-Coder-30B-A3B Q4_K_M, reached 61.308 tok/s only at short context and fell to 48.30 tok/s near 4K, so it was not selected.

## Context scaling

| Prompt tokens | Prompt tok/s | Decode tok/s |
|---:|---:|---:|
| 2,869 | 1,014.6 | 75.8 |
| 11,719 | 825.5 | 45.9 |

Long-context attention is therefore the remaining performance boundary. The 50 tok/s target is achieved in the normal operating range, but not at roughly 12K prompt tokens.

## MTP depth A/B

| MTP drafts | Short median tok/s | 2,869-token prompt | 11,719-token prompt |
|---:|---:|---:|---:|
| 1 | 90.1 | 72.0 | 47.5 |
| 2 | **96.0** | **75.8** | 45.9 |
| 4 | 74.9 | 59.4 | 39.3 |

Two drafts are retained for the best normal-workload balance. Four drafts were consistently slower. Six drafts were not tested after the clear regression at four.

The generic Strata KV flags (`int8`, `q4_0`, and `k8v4`) are not implemented by the dedicated Qwen3.6 engine; it explicitly ignores them and currently stores KV in FP16. Implementing a validated quantized-KV path is the most direct candidate for improving the long-context result.

## Quality and API behavior

The UD-Q4_K_XL candidate passed deterministic checks for Japanese explanation, arithmetic, strict JSON, Python code, and Bayesian reasoning. Qwen3.6 thinks by default. Clients that want direct answers should send:

```json
{"chat_template_kwargs":{"enable_thinking":false}}
```

Without that field, a small `max_tokens` budget can be consumed entirely by `reasoning_content` before final `content` begins.

## Production state

The systemd Strata service is active and enabled with the Qwen3.6 configuration. The API proxy, OpenWebUI, and thermal guard are active. Both llama services remain inactive and disabled. No VM, ESXi host, or passthrough GPU reset was performed.

After the production benchmark, idle MI50 state was approximately 28.0 GiB VRAM allocated and 41 degrees C.

## Upstream and alternatives checked

- [Official Qwen3.6-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Unsloth Qwen3.6 MTP GGUF release](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)
- [Strata PR #3089: gfx906 Qwen3.6 results and MTP request conditions](https://github.com/Niko1221/Strata/pull/3089)
- [Strata PR #3094: gfx906 pipeline work](https://github.com/Niko1221/Strata/pull/3094)
- [Strata PR #3168: adaptive KV ring for the Flash-Next path](https://github.com/Niko1221/Strata/pull/3168)
- [Project Maya](https://github.com/merbanan/project-maya), rejected for this machine because its current model/runtime requirements do not fit the single-MI50 target and measured speed objective.

The current upstream Qwen3.6 source was compared with this branch; there was no newer Qwen3.6-specific source delta to import. The recent adaptive-KV-ring and HIP QSA PRs target the Flash-Next engine rather than this dedicated Qwen3.6 engine.

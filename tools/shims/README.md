# Import shims

Pure-torch stand-ins for packages `transformers` expects but that are painful to
build in this environment (glibc 2.31 vs the wheels' GLIBC_2.32, CUDA 12.1 vs
torch cu128). Put this directory on `PYTHONPATH` and they take over:

| Shim | Satisfies | Used by |
| --- | --- | --- |
| `fla/` | `import fla`, `fla.modules.FusedRMSNormGated`, `fla.ops.gated_delta_rule` | `transformers` Qwen3.5 / Qwen3-Next `linear_attn` (gated delta-net) |
| `causal_conv1d/` | `import causal_conv1d` (`causal_conv1d_fn` / `_update`) | same `linear_attn` path (short conv) |
| `sitecustomize.py` | auto-run at interpreter startup | installs a torchcodec stub so vLLM's text-model introspection doesn't fail |

Usage:

    PYTHONPATH=/path/to/BQQ:/path/to/BQQ/tools/shims python your_script.py

These are pure torch — no Triton, no CUDA build. They match the real `fla`
kernels to fp16 rounding (identical next-token argmax) and run the
transformers-direct decode path ~11% slower; vLLM is unaffected because it
vendors its own `fla`. See `tools/env/README.md` for the full rationale and how
to switch to the real Triton kernels if you need that 11%.

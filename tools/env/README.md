# Environment: CUDA extensions that must be built from source

Four packages this repo depends on ship prebuilt wheels that **cannot load on this
machine**, so they have to be compiled locally. `setup_env.sh` does that.

## Why the published wheels don't work here

| Constraint | This machine |
| --- | --- |
| glibc | **2.31** (Ubuntu 20.04) — the PyPI/GitHub-release wheels for `fast-hadamard-transform` and `causal-conv1d` are built on newer distros and need `GLIBC_2.32` |
| CUDA toolkit | **12.1** (`/usr/local/cuda`), while torch is `cu128` — pip's *build isolation* pulls a different torch (cu130) and then aborts with a bogus "CUDA version mismatch". Building with `--no-build-isolation` uses the real torch and the minor 12.1-vs-12.8 gap is only a warning |
| host compiler | **gcc 9.4** — does not accept `-std=c++20` (only `-std=c++2a`), which `quiptools` requests |

So: always `--no-build-isolation`, always build from source, never let pip resolve
dependencies (`--no-deps`) — a plain `pip install` here has silently removed
already-working packages.

## What each package is for

| Package | Used by | Notes |
| --- | --- | --- |
| **fast-hadamard-transform** | The RHT (incoherence) transform in `bqqkernel/hadamard.py::matmul_hadU_cuda`, used by `IncoherentBinaryQuadratic` / `PackedIncoherentBinaryQuadratic` | If missing, the code silently falls back to a pure-torch Hadamard that is far slower **and not CUDA-graph capturable** for non-power-of-two dims |
| **quiptools** (optional) | Only needed to benchmark **QuIP#'s E8P kernel** against BQQ | Built from the local `quip-sharp` checkout with `-std=c++20` rewritten to `-std=c++17` |

## fla / causal_conv1d are repo shims — do NOT pip install them

`transformers/models/qwen3_next` (and qwen3_5) import `fla.modules.FusedRMSNormGated`,
`fla.ops.gated_delta_rule`, and `causal_conv1d_fn` for the `linear_attn` blocks.
The repo ships pure-torch stand-ins, grouped in one directory:

    tools/shims/fla/
    tools/shims/causal_conv1d/
    tools/shims/sitecustomize.py   # torchcodec stub for vLLM introspection

**Put `tools/shims` on `PYTHONPATH`** so `import fla` / `import causal_conv1d`
resolve to them and `sitecustomize.py` auto-runs, e.g.

    PYTHONPATH=/path/to/BQQ:/path/to/BQQ/tools/shims python ...

**Keep them as the single copy** — a site-packages `flash-linear-attention` /
`causal-conv1d` is redundant:

- vLLM uses its **own vendored** `vllm.model_executor.layers.fla`, not this package;
- the shim matches the real kernels to fp16 rounding (same next-token argmax) and
  is only ~11% slower, on the transformers-direct decode path alone;
- a pip copy just shadows the shim depending on path order and drifts out of sync
  with transformers (the PyPI wheel is also broken — it omits `fla/modules`).

If you truly need the Triton speed for direct inference, build them from source
(`git clone … && pip install . --no-build-isolation`) and place site-packages
ahead of `tools/shims` on `PYTHONPATH`.

## Usage

```bash
bash tools/env/setup_env.sh            # fast-hadamard-transform (fla/causal_conv1d are repo shims)
bash tools/env/setup_env.sh --quip     # also build quiptools for the QuIP# comparison
bash tools/env/setup_env.sh --check    # just verify what is importable
```

Sources are cloned into a temp dir and thrown away; nothing is vendored here.

## Known pickle incompatibility

Models saved with `dill` (e.g. `quantized_models/*.pth`) embed references to
`transformers` internals. `transformers` 5.5.3 dropped
`transformers.core_model_loading.PrefixChange`, so older checkpoints fail to
unpickle with `AttributeError: Can't get attribute 'PrefixChange'`.
`tools/env/pickle_compat.py` installs a stub for it — import it before
`torch.load`:

```python
import tools.env.pickle_compat  # noqa: F401  (installs the PrefixChange stub)
model = torch.load(path, weights_only=False, pickle_module=dill)
```

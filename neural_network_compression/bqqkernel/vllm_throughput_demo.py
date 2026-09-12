"""Throughput smoke/demo runner for vLLM BQQ exports.

Run this as a real Python file, not via ``python - <<EOF``.  vLLM's spawn-based
engine re-imports the main file during worker startup.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from math import lcm
from pathlib import Path

package_root = Path(__file__).resolve().parents[1]
project_root = package_root.parent
for repo_path in (project_root, package_root):
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))


def _register_vllm_bqq() -> None:
    import neural_network_compression.bqqkernel.vllm_quantization  # noqa: F401
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm.model_executor.models.utils import AutoWeightsLoader
    from vllm.model_executor.models import qwen3_5
    from vllm.v1 import core as v1_core
    from vllm.v1 import kv_cache_interface
    from vllm.v1.core import kv_cache_utils

    def load_bqq_vllm_named_weights(self, weights):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    # The BQQ exporter already writes vLLM fused names. The dense baseline must
    # retain the stock Hugging Face-to-vLLM mapper.
    if os.environ.get("BQQ_VLLM_FUSED_WEIGHTS", "1") != "0":
        qwen3_5.Qwen3_5Model.load_weights = load_bqq_vllm_named_weights

    # The text-only Qwen3.5 class in vLLM 0.19.1 omits this hybrid-state hook,
    # although the identical implementation exists on the VL wrapper.
    qwen3_5.Qwen3_5ForCausalLM.get_mamba_state_copy_func = classmethod(
        lambda cls: qwen3_5.Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func()
    )

    def unify_qwen35_kv_page_sizes(kv_cache_spec):
        """Align GDN pages upward without padding FlashAttention KV pages."""
        mamba_spec = kv_cache_interface.MambaSpec
        page_sizes = {spec.page_size_bytes for spec in kv_cache_spec.values()}
        if len(page_sizes) <= 1:
            return kv_cache_spec

        attention_sizes = [
            spec.page_size_bytes
            for spec in kv_cache_spec.values()
            if not isinstance(spec, mamba_spec)
        ]
        alignment = lcm(*attention_sizes) if attention_sizes else 1
        target = ((max(page_sizes) + alignment - 1) // alignment) * alignment
        unified = {}
        for name, spec in kv_cache_spec.items():
            if spec.page_size_bytes == target:
                unified[name] = spec
            elif isinstance(spec, mamba_spec):
                unified[name] = replace(spec, page_size_padded=target)
            else:
                if target % spec.page_size_bytes:
                    raise NotImplementedError(
                        f"Cannot align KV page for {name}: "
                        f"{spec.page_size_bytes} -> {target}"
                    )
                unified[name] = replace(
                    spec,
                    block_size=spec.block_size * (target // spec.page_size_bytes),
                )
        return unified

    # vLLM 0.19.1 cannot unify text-only Qwen3.5 GDN and FlashAttention pages.
    kv_cache_utils.unify_kv_cache_spec_page_size = unify_qwen35_kv_page_sizes
    v1_core.kv_cache_utils.unify_kv_cache_spec_page_size = unify_qwen35_kv_page_sizes

    # vLLM 0.25 has the implementation class but may not expose the text-only
    # architecture in its default registry.
    ModelRegistry.register_model(
        "Qwen3_5ForCausalLM",
        "vllm.model_executor.models.qwen3_5:Qwen3_5ForCausalLM",
    )


_register_vllm_bqq()


@dataclass
class ThroughputResult:
    requests: int
    generated_tokens: int
    elapsed_sec: float
    tokens_per_sec: float
    token_ids: list[int]
    generated_text: str


def run_demo(args: argparse.Namespace) -> ThroughputResult:
    from vllm import LLM, SamplingParams

    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
        stop=args.stop or None,
        repetition_penalty=args.repetition_penalty,
        frequency_penalty=args.frequency_penalty,
    )

    print(f"model: {args.model}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(
        f"batch_size={args.batch_size} max_tokens={args.max_tokens} "
        f"max_model_len={args.max_model_len} enforce_eager={args.enforce_eager} "
        f"decode_only_cuda_graph={args.decode_only_cuda_graph}"
    )

    load_start = time.perf_counter()
    llm_kwargs = dict(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=not args.disable_prefix_caching,
        mamba_cache_mode=args.mamba_cache_mode,
        mamba_block_size=args.mamba_block_size,
    )
    if args.no_repeat_ngram_size:
        os.environ["BQQ_NO_REPEAT_NGRAM_SIZE"] = str(args.no_repeat_ngram_size)
        llm_kwargs["logits_processors"] = [
            "neural_network_compression.bqqkernel.vllm_logits_processors:"
            "NoRepeatNGramLogitsProcessor"
        ]
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
        llm_kwargs["max_num_seqs"] = args.batch_size
    if args.decode_only_cuda_graph:
        if args.enforce_eager:
            raise ValueError("--decode-only-cuda-graph and --enforce-eager are exclusive")
        if args.batch_size != 1:
            raise ValueError("--decode-only-cuda-graph currently requires --batch-size 1")
        # No Dynamo/Inductor tracing is used in this mode. Calling the pybind
        # extension directly avoids custom-op functionalization around the
        # mutable fp32 decode workspace during CUDA Graph replay.
        os.environ["BQQ_VLLM_RAW_CUDA_OP"] = "1"
        llm_kwargs.update(
            # vLLM otherwise profiles the A6000 throughput defaults (8192
            # tokens and 1024 sequences), which is irrelevant and very slow
            # for this batch-1 decode benchmark.
            max_num_batched_tokens=args.max_num_batched_tokens or args.max_model_len,
            max_num_seqs=1,
            cudagraph_capture_sizes=[1],
            max_cudagraph_capture_size=1,
            # Full CUDA graphs can capture the eager model directly. Avoid the
            # separate Inductor compile, which is unnecessary for decode-only
            # capture and particularly expensive for the BQQ custom op.
            compilation_config={
                "mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY",
            },
        )
    llm = LLM(**llm_kwargs)
    print(f"load_sec={time.perf_counter() - load_start:.3f}")

    prompt = args.prompt
    if args.apply_chat_template:
        prompt = llm.get_tokenizer().apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    prompts = [prompt for _ in range(args.batch_size)]

    if args.warmup:
        llm.generate(prompts[:1], sampling)

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    elapsed = time.perf_counter() - start

    generated_tokens = sum(len(item.outputs[0].token_ids) for item in outputs)
    tps = generated_tokens / elapsed if elapsed > 0 else float("nan")
    print(f"requests={len(outputs)}")
    print(f"generated_tokens={generated_tokens}")
    print(f"elapsed_sec={elapsed:.6f}")
    print(f"tokens_per_sec={tps:.3f}")
    if outputs:
        print("sample_output:")
        print(outputs[0].outputs[0].text[: args.print_chars])
    token_ids = list(outputs[0].outputs[0].token_ids) if outputs else []
    generated_text = outputs[0].outputs[0].text if outputs else ""
    return ThroughputResult(
        len(outputs), generated_tokens, elapsed, tps, token_ids, generated_text
    )


def _write_result(path: str, result: ThroughputResult, args: argparse.Namespace) -> None:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": args.result_name,
        "model": args.model,
        "prompt": args.prompt,
        "cuda_graph": args.decode_only_cuda_graph,
        "requests": result.requests,
        "generated_tokens": result.generated_tokens,
        "elapsed_sec": result.elapsed_sec,
        "tokens_per_sec": result.tokens_per_sec,
        "token_ids": result.token_ids,
        "generated_text": result.generated_text,
    }
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved result JSON to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="lm/fine_tuned_models/Qwen3.5-4B-vllm-bqq",
        help="vLLM model/export directory",
    )
    parser.add_argument("--prompt", default="Explain binary quantization in one sentence.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Scheduler/profile token limit (decode graph defaults to max-model-len).",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=0,
        help="Ban repeated output n-grams; 0 disables the constraint.",
    )
    parser.add_argument(
        "--stop",
        action="append",
        default=[],
        help="Stop string; may be provided more than once.",
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Format the prompt as a user message before generation.",
    )
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--decode-only-cuda-graph",
        action="store_true",
        help="Capture only the single-token decode graph used by batch=1 generation.",
    )
    parser.add_argument(
        "--disable-prefix-caching",
        action="store_true",
        help="Disable vLLM prefix caching (useful for hybrid GDN/attention models).",
    )
    parser.add_argument("--mamba-cache-mode", default="align")
    parser.add_argument("--mamba-block-size", type=int, default=16)
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.add_argument("--print-chars", type=int, default=500)
    parser.add_argument("--result-name", default="vLLM")
    parser.add_argument("--json-out", default=None)
    parser.set_defaults(warmup=True)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    demo_result = run_demo(cli_args)
    if cli_args.json_out:
        _write_result(cli_args.json_out, demo_result, cli_args)

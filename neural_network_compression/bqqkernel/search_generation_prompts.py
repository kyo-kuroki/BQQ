"""Generate a fixed set of explanation prompts for BQQ/Dense comparison."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

package_root = Path(__file__).resolve().parents[1]
project_root = package_root.parent
for repo_path in (project_root, package_root):
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

from bqqkernel.vllm_throughput_demo import _register_vllm_bqq


PROMPTS = [
    "Explain what gravity is in one sentence.",
    "Explain why the sky looks blue in one sentence.",
    "Explain what the Sun is in one sentence.",
    "Explain what rain is in one sentence.",
    "Explain why plants need sunlight in one sentence.",
    "Explain what a computer does in one sentence.",
    "Explain what a bicycle is in one sentence.",
    "Explain why people need sleep in one sentence.",
    "Explain what electricity is in one sentence.",
    "Explain what a book is in one sentence.",
    "Explain how a refrigerator keeps food cold in one sentence.",
    "Explain how a traffic light works in one sentence.",
    "Explain why exercise is healthy in one sentence.",
    "Explain what photosynthesis is in one sentence.",
    "Explain how to boil an egg in one sentence.",
    "Explain why ice melts when heated in one sentence.",
    "Explain why day and night happen in one sentence.",
    "Explain what a cat is in one sentence.",
    "Explain what music is in one sentence.",
    "Explain how soap cleans hands in one sentence.",
]

STORY_PROMPTS = [
    "Once upon a time, a young girl named Mia found a small robot in the forest. The robot was lost, so Mia",
    "On a rainy evening, a small robot named Bolt stood alone at a bus stop. A child named Sam noticed it and",
    "In a quiet village, an old baker woke before sunrise. He opened his shop and",
    "A little fox discovered a glowing key beneath an ancient oak tree. Curious, it",
    "The spaceship landed gently in a field outside the town. When the door opened,",
    "Every night, Leo watched the lighthouse beam sweep across the sea. One evening, he saw",
    "Nora found an injured bird beside the garden gate. She carried it inside and",
    "The library was empty when a book began to whisper from the highest shelf. Ben climbed the ladder and",
    "A tiny boat drifted toward the island as the sun rose. Its only passenger was",
    "After the storm, Emma discovered a wooden box on the beach. Inside, she found",
    "The old clock in the town square stopped at midnight. The next morning, a curious child",
    "A friendly robot delivered bread around the city every morning. One day, it noticed",
]

PENALTY_CONFIGS = [
    (1.02, 0.0),
    (1.05, 0.0),
    (1.08, 0.0),
    (1.10, 0.0),
    (1.15, 0.0),
    (1.05, 0.1),
    (1.08, 0.1),
    (1.10, 0.1),
    (1.05, 0.2),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument(
        "--prompt-set",
        choices=["explanation", "story", "penalty"],
        default="explanation",
    )
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    _register_vllm_bqq()
    os.environ["BQQ_VLLM_RAW_CUDA_OP"] = "1"
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        max_num_seqs=1,
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        mamba_block_size=16,
        cudagraph_capture_sizes=[1],
        max_cudagraph_capture_size=1,
        compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"},
    )
    tokenizer = llm.get_tokenizer()
    if args.prompt_set == "story":
        prompts = STORY_PROMPTS
    elif args.prompt_set == "penalty":
        prompts = [
            "Write a short story about a lost robot that finds its way home "
            "with help from a kind child."
        ] * len(PENALTY_CONFIGS)
    else:
        prompts = PROMPTS
    if args.prompt_set == "story":
        formatted = prompts
    else:
        formatted = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for prompt in prompts
        ]
    if args.prompt_set == "penalty":
        sampling_params = [
            SamplingParams(
                temperature=0.0,
                max_tokens=args.max_tokens,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
            )
            for repetition_penalty, frequency_penalty in PENALTY_CONFIGS
        ]
    else:
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=args.max_tokens)
    outputs = llm.generate(formatted, sampling_params)
    records = [
        {
            "prompt": prompt,
            "text": output.outputs[0].text,
            "token_ids": list(output.outputs[0].token_ids),
            **(
                {
                    "repetition_penalty": PENALTY_CONFIGS[index][0],
                    "frequency_penalty": PENALTY_CONFIGS[index][1],
                }
                if args.prompt_set == "penalty"
                else {}
            ),
        }
        for index, (prompt, output) in enumerate(
            zip(prompts, outputs, strict=True)
        )
    ]
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"Saved {len(records)} prompt results to {path}")


if __name__ == "__main__":
    main()

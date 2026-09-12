"""Live generation speed demo for packed BQQ vs fp16 models.

The demo runs real autoregressive generation and prints a small terminal
dashboard while tokens are produced.  It is intentionally dependency-light:
only PyTorch and Transformers are required.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

import dill
import torch

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from transformers import AutoModelForCausalLM, AutoTokenizer


def _add_repo_to_path() -> None:
    package_root = Path(__file__).resolve().parents[1]
    project_root = package_root.parent
    for path in (str(project_root), str(package_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _load_bqq_model(path: str, dtype: torch.dtype, device: torch.device) -> torch.nn.Module:
    from bqqkernel.benchmark_decode import _load_bqq_model as _load_bqq_model_core

    model = _load_bqq_model_core(path, dtype)
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config"):
        setattr(model.config, "use_cache", True)
    print("[demo] model.to(device) start", flush=True)
    model.to(device)
    print("[demo] model move done", flush=True)
    _ensure_qwen35_block_type(model)
    return model


def _set_bqq_kernel_env(kernel: str | None, col_splits: str | None) -> tuple[str | None, str | None]:
    old_kernel = os.environ.get("BQQ_CUDA_DECODE_KERNEL")
    old_splits = os.environ.get("BQQ_CUDA_COL_SPLITS")
    if kernel is None:
        os.environ.pop("BQQ_CUDA_DECODE_KERNEL", None)
    else:
        os.environ["BQQ_CUDA_DECODE_KERNEL"] = kernel
    if col_splits is None:
        os.environ.pop("BQQ_CUDA_COL_SPLITS", None)
    else:
        os.environ["BQQ_CUDA_COL_SPLITS"] = col_splits
    return old_kernel, old_splits


def _restore_bqq_kernel_env(old_kernel: str | None, old_splits: str | None) -> None:
    if old_kernel is None:
        os.environ.pop("BQQ_CUDA_DECODE_KERNEL", None)
    else:
        os.environ["BQQ_CUDA_DECODE_KERNEL"] = old_kernel
    if old_splits is None:
        os.environ.pop("BQQ_CUDA_COL_SPLITS", None)
    else:
        os.environ["BQQ_CUDA_COL_SPLITS"] = old_splits


def _load_fp16_model(model_name: str, device: torch.device) -> torch.nn.Module:
    _add_repo_to_path()
    package_root = Path(__file__).resolve().parents[1]
    for name in ("causal_conv1d", "fla"):
        pkg_init = package_root / name / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            name,
            pkg_init,
            submodule_search_locations=[str(pkg_init.parent)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load local shim package: {name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        )
    except Exception as exc:
        print(f"[fp16] flash_attention_2 unavailable, using default attention: {exc}")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config"):
        setattr(model.config, "use_cache", True)
    model.eval()
    model.to(device)
    _ensure_qwen35_block_type(model)
    return model


def _ensure_qwen35_block_type(model: torch.nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "layer_type") and not hasattr(module, "block_type"):
            module.block_type = module.layer_type


def _load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _format_bar(value: float, reference: float, width: int = 28) -> str:
    if reference <= 0:
        frac = 0.0
    else:
        frac = min(max(value / reference, 0.0), 1.0)
    filled = int(round(frac * width))
    return "#" * filled + "-" * (width - filled)


def _compact_text(text: str, limit: int) -> str:
    text = text.replace("\r", "\\r")
    if len(text) <= limit:
        return text
    return "..." + text[-limit + 3 :]


def _video_font_path() -> str:
    requested = os.environ.get("BQQ_VIDEO_FONT")
    candidates = [
        requested,
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise RuntimeError(
        "No usable video font found. Set BQQ_VIDEO_FONT to a TrueType/OpenType font."
    )


def _wrapped_tail(text: str, width: int, max_lines: int) -> list[str]:
    clean = text.replace("\r", "\\r").replace("\t", "    ")
    lines: list[str] = []
    for paragraph in clean.splitlines() or [""]:
        lines.extend(textwrap.wrap(
            paragraph,
            width=width,
            replace_whitespace=False,
            drop_whitespace=False,
        ) or [""])
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
        lines[0] = "..." + lines[0][3:]
    return lines


def _render_video_frame(
    *,
    name: str,
    prompt: str,
    generated: str,
    stat: "TokenStat",
    max_new_tokens: int,
    prefill_ms: float,
    reference_tps: float,
    width: int,
    height: int,
):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("--video-out requires Pillow") from exc

    scale = min(width / 1280.0, height / 720.0)
    margin = max(18, int(34 * scale))
    title_size = max(22, int(34 * scale))
    body_size = max(14, int(20 * scale))
    small_size = max(12, int(16 * scale))
    font_path = _video_font_path()
    title_font = ImageFont.truetype(font_path, title_size)
    body_font = ImageFont.truetype(font_path, body_size)
    small_font = ImageFont.truetype(font_path, small_size)

    image = Image.new("RGB", (width, height), "#09131f")
    draw = ImageDraw.Draw(image)

    def line_capacity(font, available_width: float) -> int:
        sample = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        average_width = draw.textlength(sample, font=font) / len(sample)
        return max(12, int(available_width / max(average_width, 1.0)))

    draw.rectangle((0, 0, width, max(6, int(8 * scale))), fill="#26d7a0")
    draw.text((margin, margin), name, font=title_font, fill="#f4f7f5")
    token_label = f"TOKEN {stat.index:03d} / {max_new_tokens}"
    token_box = draw.textbbox((0, 0), token_label, font=body_font)
    draw.text(
        (width - margin - (token_box[2] - token_box[0]), margin + 6 * scale),
        token_label,
        font=body_font,
        fill="#26d7a0",
    )

    metrics_y = margin + title_size + int(28 * scale)
    metrics = (
        f"PREFILL  {prefill_ms:8.2f} ms     "
        f"LAST  {stat.latency_ms:7.2f} ms/token     "
        f"AVERAGE  {stat.cumulative_tps:7.2f} tok/s"
    )
    draw.text((margin, metrics_y), metrics, font=small_font, fill="#b7c7d6")

    bar_y = metrics_y + small_size + int(18 * scale)
    bar_h = max(10, int(14 * scale))
    bar_width = width - 2 * margin
    fraction = min(max(stat.cumulative_tps / reference_tps, 0.0), 1.0) if reference_tps > 0 else 0.0
    draw.rounded_rectangle(
        (margin, bar_y, margin + bar_width, bar_y + bar_h),
        radius=bar_h // 2,
        fill="#26384a",
    )
    if fraction > 0:
        draw.rounded_rectangle(
            (margin, bar_y, margin + max(bar_h, int(bar_width * fraction)), bar_y + bar_h),
            radius=bar_h // 2,
            fill="#26d7a0",
        )

    panel_top = bar_y + bar_h + int(30 * scale)
    panel_gap = int(18 * scale)
    prompt_h = max(100, int(130 * scale))
    panel_color = "#102333"
    draw.rounded_rectangle(
        (margin, panel_top, width - margin, panel_top + prompt_h),
        radius=max(8, int(12 * scale)),
        fill=panel_color,
    )
    draw.text((margin + 18 * scale, panel_top + 13 * scale), "PROMPT", font=small_font, fill="#f2b84b")
    text_width = width - 2 * margin - 36 * scale
    prompt_lines = _wrapped_tail(
        prompt, line_capacity(small_font, text_width), 3)
    draw.multiline_text(
        (margin + 18 * scale, panel_top + 43 * scale),
        "\n".join(prompt_lines),
        font=small_font,
        fill="#e7edf1",
        spacing=max(3, int(5 * scale)),
    )

    generated_top = panel_top + prompt_h + panel_gap
    generated_bottom = height - margin
    draw.rounded_rectangle(
        (margin, generated_top, width - margin, generated_bottom),
        radius=max(8, int(12 * scale)),
        fill=panel_color,
    )
    draw.text(
        (margin + 18 * scale, generated_top + 13 * scale),
        "GENERATED",
        font=small_font,
        fill="#42bcec",
    )
    line_height = body_size + max(4, int(7 * scale))
    max_lines = max(1, int((generated_bottom - generated_top - 62 * scale) / line_height))
    generated_lines = _wrapped_tail(
        generated, line_capacity(body_font, text_width), max_lines)
    draw.multiline_text(
        (margin + 18 * scale, generated_top + 48 * scale),
        "\n".join(generated_lines),
        font=body_font,
        fill="#f4f7f5",
        spacing=max(4, int(7 * scale)),
    )
    return image


@dataclass
class TokenStat:
    index: int
    token_id: int
    text: str
    latency_ms: float
    instantaneous_tps: float
    cumulative_tps: float


@dataclass
class RunResult:
    name: str
    generated_text: str
    prefill_ms: float
    total_decode_ms: float
    mean_decode_ms: float
    tokens_per_second: float
    token_stats: list[TokenStat]


def _render_progress(
    *,
    name: str,
    prompt: str,
    generated: str,
    stat: TokenStat,
    max_new_tokens: int,
    prefill_ms: float,
    refresh_screen: bool,
    reference_tps: float,
) -> None:
    if refresh_screen:
        print("\033[2J\033[H", end="")
    print(f"[{name}] token {stat.index:03d}/{max_new_tokens}")
    print(f"prefill: {prefill_ms:.2f} ms")
    print(
        f"last: {stat.latency_ms:7.2f} ms/token | "
        f"inst: {stat.instantaneous_tps:7.2f} tok/s | "
        f"avg: {stat.cumulative_tps:7.2f} tok/s"
    )
    print(f"speed: [{_format_bar(stat.cumulative_tps, reference_tps)}]")
    print(f"token: {stat.text!r} (id={stat.token_id})")
    print("-" * 80)
    print("prompt:")
    print(_compact_text(prompt, 400))
    print("-" * 80)
    print("generated:")
    print(_compact_text(generated, 1200))
    print("-" * 80, flush=True)


@torch.inference_mode()
def _generate_token_ids_eager(
    *,
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[list[int], float]:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    print("[demo] prefill start", flush=True)
    prefill_start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    prefill_ms = (time.perf_counter() - prefill_start) * 1000.0
    print(f"[demo] prefill done: {prefill_ms:.2f} ms", flush=True)

    past_key_values = outputs.past_key_values
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    del outputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    generated_ids = [int(next_token.item())]
    step_ids = next_token

    eos_id = tokenizer.eos_token_id
    for _ in range(2, max_new_tokens + 1):
        print("[demo] decode step start", flush=True)
        with torch.inference_mode():
            outputs = model(input_ids=step_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        step_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        token_id = int(step_ids.item())
        generated_ids.append(token_id)
        if eos_id is not None and token_id == eos_id:
            break
    _sync_if_cuda(device)
    return generated_ids, prefill_ms


@torch.inference_mode()
def run_cuda_graph_replay_demo(
    *,
    name: str,
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    refresh_screen: bool,
    reference_tps: float,
    graph_warmup: int,
) -> RunResult:
    if device.type != "cuda":
        raise ValueError("CUDA graph mode requires a CUDA device")

    # Tokens are generated eagerly for display.  The speed shown below is from
    # CUDA graph replay of one fixed-context decode step, matching
    # benchmark_decode.py --use-cuda-graph.
    print("[demo] graph mode eager warmup", flush=True)
    generated_ids, prefill_ms = _generate_token_ids_eager(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    past_key_values = outputs.past_key_values
    static_step_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    del outputs
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("[demo] graph capture prep done", flush=True)

    def graph_step():
        with torch.inference_mode():
            model(input_ids=static_step_ids, past_key_values=past_key_values, use_cache=True)

    for _ in range(max(1, graph_warmup)):
        graph_step()
    torch.cuda.synchronize()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            graph_step()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_step()
    torch.cuda.synchronize()

    token_stats: list[TokenStat] = []
    generated_so_far: list[int] = []
    decode_ms: list[float] = []
    for index, token_id in enumerate(generated_ids, start=1):
        generated_so_far.append(token_id)
        generated_text = tokenizer.decode(generated_so_far, skip_special_tokens=False)
        if index == 1:
            latency_ms = 0.0
            cumulative_tps = 0.0
            inst_tps = 0.0
        else:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            latency_ms = start.elapsed_time(end)
            decode_ms.append(latency_ms)
            total_decode_s = sum(decode_ms) / 1000.0
            cumulative_tps = len(decode_ms) / total_decode_s if total_decode_s > 0 else 0.0
            inst_tps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

        stat = TokenStat(
            index=index,
            token_id=token_id,
            text=tokenizer.decode([token_id], skip_special_tokens=False),
            latency_ms=latency_ms,
            instantaneous_tps=inst_tps,
            cumulative_tps=cumulative_tps,
        )
        token_stats.append(stat)
        _render_progress(
            name=f"{name} CUDA graph replay",
            prompt=prompt,
            generated=generated_text,
            stat=stat,
            max_new_tokens=max_new_tokens,
            prefill_ms=prefill_ms,
            refresh_screen=refresh_screen,
            reference_tps=reference_tps,
        )

    total_decode_ms = sum(decode_ms)
    mean_decode_ms = total_decode_ms / len(decode_ms) if decode_ms else 0.0
    return RunResult(
        name=f"{name} cuda_graph",
        generated_text=tokenizer.decode(generated_ids, skip_special_tokens=False),
        prefill_ms=prefill_ms,
        total_decode_ms=total_decode_ms,
        mean_decode_ms=mean_decode_ms,
        tokens_per_second=1000.0 / mean_decode_ms if mean_decode_ms > 0 else 0.0,
        token_stats=token_stats,
    )


@torch.inference_mode()
def run_generation(
    *,
    name: str,
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    refresh_screen: bool,
    reference_tps: float,
) -> RunResult:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    print(f"[demo] {name} prefill start", flush=True)
    _sync_if_cuda(device)
    prefill_start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    _sync_if_cuda(device)
    prefill_ms = (time.perf_counter() - prefill_start) * 1000.0
    print(f"[demo] {name} prefill done: {prefill_ms:.2f} ms", flush=True)

    past_key_values = outputs.past_key_values
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated_ids: list[int] = [int(next_token.item())]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    token_stats: list[TokenStat] = []

    # The first token comes from prefill logits, so report it with zero decode
    # latency and start true decode timing from the second generated token.
    first = TokenStat(
        index=1,
        token_id=generated_ids[-1],
        text=tokenizer.decode(generated_ids[-1:], skip_special_tokens=False),
        latency_ms=0.0,
        instantaneous_tps=0.0,
        cumulative_tps=0.0,
    )
    token_stats.append(first)
    _render_progress(
        name=name,
        prompt=prompt,
        generated=generated_text,
        stat=first,
        max_new_tokens=max_new_tokens,
        prefill_ms=prefill_ms,
        refresh_screen=refresh_screen,
        reference_tps=reference_tps,
    )

    decode_ms: list[float] = []
    step_ids = next_token
    for index in range(2, max_new_tokens + 1):
        print(f"[demo] {name} decode step {index}", flush=True)
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = model(input_ids=step_ids, past_key_values=past_key_values, use_cache=True)
        latency_ms = (time.perf_counter() - start) * 1000.0

        past_key_values = outputs.past_key_values
        step_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        token_id = int(step_ids.item())
        generated_ids.append(token_id)
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        decode_ms.append(latency_ms)
        total_decode_s = sum(decode_ms) / 1000.0
        cumulative_tps = len(decode_ms) / total_decode_s if total_decode_s > 0 else 0.0
        stat = TokenStat(
            index=index,
            token_id=token_id,
            text=tokenizer.decode([token_id], skip_special_tokens=False),
            latency_ms=latency_ms,
            instantaneous_tps=1000.0 / latency_ms if latency_ms > 0 else 0.0,
            cumulative_tps=cumulative_tps,
        )
        token_stats.append(stat)
        _render_progress(
            name=name,
            prompt=prompt,
            generated=generated_text,
            stat=stat,
            max_new_tokens=max_new_tokens,
            prefill_ms=prefill_ms,
            refresh_screen=refresh_screen,
            reference_tps=reference_tps,
        )

        eos_id = tokenizer.eos_token_id
        if eos_id is not None and token_id == eos_id:
            break

    total_decode_ms = sum(decode_ms)
    mean_decode_ms = total_decode_ms / len(decode_ms) if decode_ms else 0.0
    tokens_per_second = 1000.0 / mean_decode_ms if mean_decode_ms > 0 else 0.0
    return RunResult(
        name=name,
        generated_text=generated_text,
        prefill_ms=prefill_ms,
        total_decode_ms=total_decode_ms,
        mean_decode_ms=mean_decode_ms,
        tokens_per_second=tokens_per_second,
        token_stats=token_stats,
    )


def _free_model(model: torch.nn.Module | None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _write_csv(path: str, results: list[RunResult]) -> None:
    rows = []
    for result in results:
        for stat in result.token_stats:
            rows.append({
                "model": result.name,
                "token_index": stat.index,
                "token_id": stat.token_id,
                "token_text": stat.text,
                "latency_ms": stat.latency_ms,
                "instantaneous_tps": stat.instantaneous_tps,
                "cumulative_tps": stat.cumulative_tps,
            })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def _write_json(path: str, results: list[RunResult], args: argparse.Namespace) -> None:
    payload = {
        "args": vars(args),
        "results": [
            {
                "name": r.name,
                "generated_text": r.generated_text,
                "prefill_ms": r.prefill_ms,
                "total_decode_ms": r.total_decode_ms,
                "mean_decode_ms": r.mean_decode_ms,
                "tokens_per_second": r.tokens_per_second,
                "tokens": [stat.__dict__ for stat in r.token_stats],
            }
            for r in results
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_video(
    path: str,
    results: list[RunResult],
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    reference_tps: float,
    fps: int,
    token_duration: float,
    width: int,
    height: int,
    use_measured_timing: bool = False,
    time_scale: float = 1.0,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("--video-out requires ffmpeg on PATH")
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("video width and height must be positive even numbers")
    if fps <= 0 or token_duration <= 0 or time_scale <= 0:
        raise ValueError("video FPS, token duration, and time scale must be positive")

    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{width}x{height}",
        "-framerate", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]
    ffmpeg_env = os.environ.copy()
    # Conda CUDA environments can shadow system libffi/ncurses dependencies
    # required by the distro ffmpeg binary.
    ffmpeg_env.pop("LD_LIBRARY_PATH", None)
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=ffmpeg_env,
    )
    if process.stdin is None or process.stderr is None:
        process.kill()
        raise RuntimeError("failed to open ffmpeg pipes")

    frames_per_token = max(1, round(fps * token_duration))
    write_error: Exception | None = None
    try:
        for result in results:
            generated_ids: list[int] = []
            measured_elapsed_s = 0.0
            emitted_frames = 0
            for position, stat in enumerate(result.token_stats):
                generated_ids.append(stat.token_id)
                generated = tokenizer.decode(generated_ids, skip_special_tokens=False)
                frame = _render_video_frame(
                    name=result.name,
                    prompt=prompt,
                    generated=generated,
                    stat=stat,
                    max_new_tokens=max_new_tokens,
                    prefill_ms=result.prefill_ms,
                    reference_tps=reference_tps,
                    width=width,
                    height=height,
                )
                repeats = frames_per_token
                if use_measured_timing and stat.latency_ms > 0:
                    measured_elapsed_s += stat.latency_ms / 1000.0 * time_scale
                    target_frames = max(
                        position + 1,
                        round(fps * measured_elapsed_s),
                    )
                    repeats = max(1, target_frames - emitted_frames)
                elif position == 0 or position == len(result.token_stats) - 1:
                    repeats = max(repeats, fps)
                frame_bytes = frame.tobytes()
                for _ in range(repeats):
                    process.stdin.write(frame_bytes)
                emitted_frames += repeats
    except Exception as exc:
        write_error = exc
    finally:
        process.stdin.close()

    stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
    return_code = process.wait()
    if write_error is not None:
        raise RuntimeError(f"video encoding failed: {write_error}; ffmpeg: {stderr}") from write_error
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with status {return_code}: {stderr}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--bqq-model-path", required=True)
    parser.add_argument("--mode", choices=["bqq", "fp16", "both"], default="both")
    parser.add_argument("--prompt", default="Explain why quantized language models can be faster during autoregressive decoding.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bqq-dtype", choices=["float16", "bfloat16", "bf16", "float32"], default="float16")
    parser.add_argument("--refresh-screen", action="store_true",
                        help="Redraw one live dashboard instead of printing one block per token.")
    parser.add_argument("--use-cuda-graph", action="store_true",
                        help="Visualize CUDA graph replay speed. Token text is generated eagerly first; "
                             "per-token speed is measured by replaying one fixed-context decode graph.")
    parser.add_argument("--graph-warmup", type=int, default=20)
    parser.add_argument("--reference-tps", type=float, default=180.0,
                        help="Scale used for the terminal speed bar.")
    parser.add_argument("--csv-out", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--video-out", default=None,
                        help="Save the token-by-token dashboard as an H.264 MP4 after generation.")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-token-duration", type=float, default=0.15,
                        help="Seconds to show each generated token in the saved video.")
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument(
        "--bqq-decode-kernel",
        default=None,
        help="Override BQQ_CUDA_DECODE_KERNEL for the packed BQQ run. Use this to force a stable decode kernel such as two_stage_warp.",
    )
    parser.add_argument(
        "--bqq-col-splits",
        default=None,
        help="Override BQQ_CUDA_COL_SPLITS for the packed BQQ run.",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)

    print("[demo] tokenizer load start", flush=True)
    tokenizer = _load_tokenizer(args.model_name)
    print("[demo] tokenizer load done", flush=True)
    results: list[RunResult] = []

    if args.mode in {"bqq", "both"}:
        print("Loading packed BQQ model...", flush=True)
        old_kernel = old_splits = None
        try:
            old_kernel, old_splits = _set_bqq_kernel_env(args.bqq_decode_kernel, args.bqq_col_splits)
            bqq = _load_bqq_model(args.bqq_model_path, _dtype_from_name(args.bqq_dtype), device)
            runner = run_cuda_graph_replay_demo if args.use_cuda_graph else run_generation
            kwargs = {"graph_warmup": args.graph_warmup} if args.use_cuda_graph else {}
            results.append(runner(
                name="BQQ packed",
                model=bqq,
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                device=device,
                refresh_screen=args.refresh_screen,
                reference_tps=args.reference_tps,
                **kwargs,
            ))
            _free_model(bqq)
        finally:
            _restore_bqq_kernel_env(old_kernel, old_splits)

    if args.mode in {"fp16", "both"}:
        print("Loading fp16 baseline model...", flush=True)
        fp16 = _load_fp16_model(args.model_name, device)
        runner = run_cuda_graph_replay_demo if args.use_cuda_graph else run_generation
        kwargs = {"graph_warmup": args.graph_warmup} if args.use_cuda_graph else {}
        results.append(runner(
            name="fp16",
            model=fp16,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            device=device,
            refresh_screen=args.refresh_screen,
            reference_tps=args.reference_tps,
            **kwargs,
        ))
        _free_model(fp16)

    print("\nSummary")
    print("model,prefill_ms,mean_decode_ms,tokens_per_s,total_decode_ms")
    for result in results:
        print(
            f"{result.name},{result.prefill_ms:.2f},{result.mean_decode_ms:.2f},"
            f"{result.tokens_per_second:.2f},{result.total_decode_ms:.2f}"
        )
    if len(results) == 2 and results[1].tokens_per_second > 0:
        speedup = results[0].tokens_per_second / results[1].tokens_per_second
        print(f"decode_speedup_vs_fp16,{speedup:.2f}x")

    if args.csv_out:
        _write_csv(args.csv_out, results)
        print(f"Saved CSV to {args.csv_out}")
    if args.json_out:
        _write_json(args.json_out, results, args)
        print(f"Saved JSON to {args.json_out}")
    if args.video_out:
        print(f"Encoding generation video to {args.video_out}...", flush=True)
        _write_video(
            args.video_out,
            results,
            tokenizer,
            args.prompt,
            args.max_new_tokens,
            args.reference_tps,
            args.video_fps,
            args.video_token_duration,
            args.video_width,
            args.video_height,
        )
        print(f"Saved video to {args.video_out}")


if __name__ == "__main__":
    main()

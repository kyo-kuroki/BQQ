"""Render vLLM CUDA Graph result JSON files as a paced comparison video."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

package_root = Path(__file__).resolve().parents[1]
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

from bqqkernel.generation_speed_demo import (
    RunResult,
    TokenStat,
    _load_tokenizer,
    _render_video_frame,
    _write_video,
)


def _load_result(path: str, tokenizer) -> RunResult:
    with Path(path).expanduser().open() as f:
        payload = json.load(f)
    token_ids = [int(token_id) for token_id in payload["token_ids"]]
    elapsed_sec = float(payload["elapsed_sec"])
    token_count = len(token_ids)
    mean_ms = elapsed_sec * 1000.0 / token_count if token_count else 0.0
    tps = token_count / elapsed_sec if elapsed_sec > 0 else 0.0
    stats = [
        TokenStat(
            index=index,
            token_id=token_id,
            text=tokenizer.decode([token_id], skip_special_tokens=False),
            latency_ms=mean_ms,
            instantaneous_tps=tps,
            cumulative_tps=tps,
        )
        for index, token_id in enumerate(token_ids, start=1)
    ]
    return RunResult(
        name=str(payload["name"]),
        generated_text=str(payload.get("generated_text", "")),
        prefill_ms=0.0,
        total_decode_ms=elapsed_sec * 1000.0,
        mean_decode_ms=mean_ms,
        tokens_per_second=tps,
        token_stats=stats,
    )


def _write_side_by_side_video(
    path: str,
    results: list[RunResult],
    tokenizer,
    prompt: str,
    fps: int,
    time_scale: float,
    width: int,
    height: int,
) -> None:
    from PIL import Image, ImageDraw

    if len(results) != 2:
        raise ValueError("side-by-side rendering requires exactly two results")
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("video width and height must be positive even numbers")
    if fps <= 0 or time_scale <= 0:
        raise ValueError("FPS and time scale must be positive")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("video rendering requires ffmpeg on PATH")

    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{width}x{height}", "-framerate", str(fps),
        "-i", "-", "-an",
    ]
    if output_path.suffix.lower() == ".gif":
        command.extend([
            "-filter_complex",
            "[0:v]split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
            "[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
            "-loop", "0", str(output_path),
        ])
    else:
        command.extend([
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(output_path),
        ])
    ffmpeg_env = os.environ.copy()
    ffmpeg_env.pop("LD_LIBRARY_PATH", None)
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stderr=subprocess.PIPE, env=ffmpeg_env)
    if process.stdin is None or process.stderr is None:
        process.kill()
        raise RuntimeError("failed to open ffmpeg pipes")

    panel_width = width // 2
    max_tokens = max(len(result.token_stats) for result in results)
    reference_tps = max(result.tokens_per_second for result in results)
    duration_s = max(result.total_decode_ms for result in results) / 1000.0
    frame_count = max(1, math.ceil(duration_s * time_scale * fps))

    write_error: Exception | None = None
    try:
        for frame_index in range(frame_count):
            source_time_s = (frame_index + 1) / fps / time_scale
            panels = []
            for result in results:
                token_count = min(
                    len(result.token_stats),
                    int(source_time_s * result.tokens_per_second),
                )
                if token_count:
                    stat = result.token_stats[token_count - 1]
                    token_ids = [
                        item.token_id for item in result.token_stats[:token_count]
                    ]
                    generated = tokenizer.decode(
                        token_ids, skip_special_tokens=False)
                else:
                    stat = TokenStat(
                        index=0,
                        token_id=-1,
                        text="",
                        latency_ms=result.mean_decode_ms,
                        instantaneous_tps=result.tokens_per_second,
                        cumulative_tps=result.tokens_per_second,
                    )
                    generated = ""
                display_name = result.name.replace(" | vLLM CUDA Graph", "")
                panels.append(_render_video_frame(
                    name=display_name,
                    prompt=prompt,
                    generated=generated,
                    stat=stat,
                    max_new_tokens=max_tokens,
                    prefill_ms=result.prefill_ms,
                    reference_tps=reference_tps,
                    width=panel_width,
                    height=height,
                ))

            frame = Image.new("RGB", (width, height), "#09131f")
            frame.paste(panels[0], (0, 0))
            frame.paste(panels[1], (panel_width, 0))
            draw = ImageDraw.Draw(frame)
            draw.rectangle(
                (panel_width - 2, 0, panel_width + 2, height), fill="#26d7a0")
            process.stdin.write(frame.tobytes())
    except Exception as exc:
        write_error = exc
    finally:
        process.stdin.close()

    stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
    return_code = process.wait()
    if write_error is not None:
        raise RuntimeError(
            f"video encoding failed: {write_error}; ffmpeg: {stderr}") from write_error
    if return_code != 0:
        raise RuntimeError(f"video encoding failed with code {return_code}: {stderr}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bqq-json", required=True)
    parser.add_argument("--dense-json", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--video-out", required=True)
    parser.add_argument("--fps", type=int, default=240,
                        help="Use at least the fastest token rate for real-time playback.")
    parser.add_argument("--time-scale", type=float, default=1.0,
                        help="Playback time multiplier; 1.0 preserves measured speed.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--layout",
        choices=["side-by-side", "sequential"],
        default="side-by-side",
    )
    args = parser.parse_args()

    tokenizer = _load_tokenizer(args.model_name)
    results = [
        _load_result(args.bqq_json, tokenizer),
        _load_result(args.dense_json, tokenizer),
    ]
    with Path(args.bqq_json).expanduser().open() as f:
        prompt = str(json.load(f)["prompt"])
    max_tokens = max(len(result.token_stats) for result in results)
    reference_tps = max(result.tokens_per_second for result in results)
    if args.layout == "side-by-side":
        _write_side_by_side_video(
            args.video_out,
            results,
            tokenizer,
            prompt,
            args.fps,
            args.time_scale,
            args.width,
            args.height,
        )
    else:
        _write_video(
            args.video_out,
            results,
            tokenizer,
            prompt,
            max_tokens,
            reference_tps,
            args.fps,
            0.1,
            args.width,
            args.height,
            use_measured_timing=True,
            time_scale=args.time_scale,
        )
    print(f"Saved vLLM comparison video to {args.video_out}")


if __name__ == "__main__":
    main()

"""Media helpers for the dubbing pipeline.

This module intentionally keeps all subprocess-heavy logic in one place so the
pipeline stages stay focused on orchestration.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


class MediaToolError(RuntimeError):
    """Raised when a required media tool fails."""


@dataclass(frozen=True)
class CommandResult:
    """Structured subprocess result."""

    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def which(program: str) -> str | None:
    return shutil.which(program)


def require_executable(program: str, hint: str | None = None) -> str:
    resolved = which(program)
    if not resolved:
        message = f"Required executable '{program}' was not found on PATH."
        if hint:
            message = f"{message} {hint}"
        raise MediaToolError(message)
    return resolved


def run_command(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = True,
    check: bool = True,
) -> CommandResult:
    proc = subprocess.run(
        list(args),
        cwd=str(cwd) if cwd else None,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
    )
    result = CommandResult(
        args=tuple(args),
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )
    if check and result.returncode != 0:
        raise MediaToolError(
            "Command failed with exit code "
            f"{result.returncode}: {' '.join(result.args)}\n{result.stderr.strip()}"
        )
    return result


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def chunk_filter_value(value: float, minimum: float = 0.5, maximum: float = 2.0) -> list[float]:
    """Split a speed factor into ffmpeg atempo-safe pieces.

    ffmpeg's atempo filter accepts 0.5-2.0 per filter. This helper decomposes
    larger or smaller factors into a stable chain.
    """

    if value <= 0:
        raise ValueError("speed factor must be positive")

    pieces: list[float] = []
    remaining = value
    while remaining > maximum:
        pieces.append(maximum)
        remaining /= maximum
    while remaining < minimum:
        pieces.append(minimum)
        remaining /= minimum
    pieces.append(remaining)
    return pieces


def atempo_filter_chain(speed_factor: float) -> str:
    pieces = chunk_filter_value(speed_factor)
    return ",".join(f"atempo={piece:.10f}".rstrip("0").rstrip(".") for piece in pieces)


def ffprobe_duration_seconds(path: Path) -> float:
    ffprobe = require_executable("ffprobe", "Install FFmpeg to inspect media duration.")
    result = run_command(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise MediaToolError(f"Could not parse duration for {path}") from exc


def extract_audio_from_video(
    input_video: Path,
    output_audio: Path,
    *,
    sample_rate: int = 48000,
    channels: int = 2,
) -> Path:
    ffmpeg = require_executable("ffmpeg", "Install FFmpeg to extract and mux media.")
    ensure_parent_dir(output_audio)
    run_command(
        [
            ffmpeg,
            "-y",
            "-i",
            str(input_video),
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output_audio),
        ]
    )
    return output_audio


def run_ffmpeg_filter(
    inputs: Sequence[Path],
    output: Path,
    filter_complex: str,
    output_args: Sequence[str],
) -> Path:
    ffmpeg = require_executable("ffmpeg", "Install FFmpeg to process media.")
    ensure_parent_dir(output)
    command: list[str] = [ffmpeg, "-y"]
    for item in inputs:
        command.extend(["-i", str(item)])
    command.extend(["-filter_complex", filter_complex])
    command.extend(list(output_args))
    command.append(str(output))
    run_command(command)
    return output


def stretch_audio_ffmpeg(input_audio: Path, output_audio: Path, speed_factor: float) -> Path:
    if math.isclose(speed_factor, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        ensure_parent_dir(output_audio)
        shutil.copyfile(input_audio, output_audio)
        return output_audio
    filter_chain = atempo_filter_chain(speed_factor)
    return run_ffmpeg_filter(
        [input_audio],
        output_audio,
        filter_chain,
        ["-vn", "-c:a", "pcm_s16le"],
    )


def stretch_audio_rubberband(input_audio: Path, output_audio: Path, speed_factor: float) -> Path:
    rubberband = require_executable("rubberband", "Install Rubber Band for higher quality time stretching.")
    ensure_parent_dir(output_audio)
    run_command(
        [
            rubberband,
            "-t",
            str(speed_factor),
            str(input_audio),
            str(output_audio),
        ]
    )
    return output_audio


def concatenate_audio(inputs: Sequence[Path], output_audio: Path) -> Path:
    if not inputs:
        raise MediaToolError("concatenate_audio requires at least one input.")
    ffmpeg = require_executable("ffmpeg", "Install FFmpeg to concatenate media.")
    ensure_parent_dir(output_audio)
    command: list[str] = [ffmpeg, "-y"]
    for item in inputs:
        command.extend(["-i", str(item)])
    filter_parts = [f"[{index}:a]" for index in range(len(inputs))]
    filter_complex = "".join(filter_parts) + f"concat=n={len(inputs)}:v=0:a=1[out]"
    command.extend(["-filter_complex", filter_complex, "-map", "[out]", "-c:a", "pcm_s16le", str(output_audio)])
    run_command(command)
    return output_audio


def merge_audio_segments(
    input_audio: Path,
    segments: Sequence[tuple[float, float]],
    output_audio: Path,
) -> Path:
    cleaned_segments: list[tuple[float, float]] = []
    for start_seconds, end_seconds in segments:
        start = max(0.0, float(start_seconds))
        end = max(start, float(end_seconds))
        if end - start < 0.05:
            continue
        if cleaned_segments and start <= cleaned_segments[-1][1] + 0.02:
            previous_start, previous_end = cleaned_segments[-1]
            cleaned_segments[-1] = (previous_start, max(previous_end, end))
            continue
        cleaned_segments.append((start, end))
    if not cleaned_segments:
        raise MediaToolError("merge_audio_segments requires at least one non-empty segment.")

    ffmpeg = require_executable("ffmpeg", "Install FFmpeg to merge audio segments.")
    ensure_parent_dir(output_audio)
    filter_parts = []
    concat_inputs = []
    for index, (start, end) in enumerate(cleaned_segments):
        duration = end - start
        filter_parts.append(
            f"[0:a]atrim=start={start:.6f}:duration={duration:.6f},asetpts=PTS-STARTPTS[s{index}]"
        )
        concat_inputs.append(f"[s{index}]")
    filter_complex = ";".join(filter_parts + ["".join(concat_inputs) + f"concat=n={len(cleaned_segments)}:v=0:a=1[out]"])
    run_command(
        [
            ffmpeg,
            "-y",
            "-i",
            str(input_audio),
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-c:a",
            "pcm_s16le",
            str(output_audio),
        ]
    )
    return output_audio


def detect_nonsilent_spans(
    input_audio: Path,
    *,
    noise_db: float = -35.0,
    minimum_silence_seconds: float = 0.35,
) -> list[tuple[float, float]]:
    ffmpeg = require_executable("ffmpeg", "Install FFmpeg to analyze silence.")
    result = run_command(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            str(input_audio),
            "-af",
            f"silencedetect=noise={noise_db}dB:d={minimum_silence_seconds}",
            "-f",
            "null",
            "-",
        ]
    )
    log_output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    spans: list[tuple[float, float]] = []
    current_start = 0.0
    for line in log_output.splitlines():
        start_match = re.search(r"silence_start:\s*([0-9.]+)", line)
        if start_match:
            silence_start = float(start_match.group(1))
            if silence_start > current_start:
                spans.append((current_start, silence_start))
            continue
        end_match = re.search(r"silence_end:\s*([0-9.]+)", line)
        if end_match:
            current_start = float(end_match.group(1))
    total_duration = ffprobe_duration_seconds(input_audio)
    if total_duration > current_start:
        spans.append((current_start, total_duration))
    return [(start, end) for start, end in spans if end - start >= 0.05]


def rms_level_db(
    input_audio: Path,
) -> float | None:
    ffmpeg = require_executable("ffmpeg", "Install FFmpeg to analyze audio loudness.")
    result = run_command(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            str(input_audio),
            "-af",
            "astats=metadata=1:reset=0",
            "-f",
            "null",
            "-",
        ]
    )
    log_output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    matches = re.findall(r"RMS level dB:\s*(-?[0-9.]+)", log_output)
    if not matches:
        return None
    return float(matches[-1])


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frame_rate = handle.getframerate()
        if frame_rate <= 0:
            raise MediaToolError(f"Unsupported frame rate for {path}.")
        return handle.getnframes() / float(frame_rate)


def pad_audio_with_silence(
    input_audio: Path,
    output_audio: Path,
    leading_seconds: float,
) -> Path:
    if leading_seconds <= 0:
        if input_audio != output_audio:
            ensure_parent_dir(output_audio)
            shutil.copyfile(input_audio, output_audio)
        return output_audio
    ffmpeg = require_executable("ffmpeg", "Install FFmpeg to create padded audio.")
    ensure_parent_dir(output_audio)
    delay_ms = int(round(leading_seconds * 1000))
    run_command(
        [
            ffmpeg,
            "-y",
            "-i",
            str(input_audio),
            "-filter_complex",
            f"adelay={delay_ms}|{delay_ms}",
            "-c:a",
            "pcm_s16le",
            str(output_audio),
        ]
    )
    return output_audio


def mix_audio_tracks(tracks: Sequence[Path], output_audio: Path) -> Path:
    if not tracks:
        raise MediaToolError("mix_audio_tracks requires at least one track.")
    filter_inputs = []
    for index, _ in enumerate(tracks):
        filter_inputs.append(f"[{index}:a]")
    mix = "".join(filter_inputs) + f"amix=inputs={len(tracks)}:duration=longest:normalize=0[aout]"
    return run_ffmpeg_filter(tracks, output_audio, mix, ["-map", "[aout]", "-c:a", "pcm_s16le"])


def mix_voice_and_instrumental(
    voice_track: Path,
    instrumental_track: Path,
    output_audio: Path,
    *,
    voice_gain_db: float = 0.0,
    instrumental_gain_db: float = -6.0,
) -> Path:
    filter_complex = (
        f"[0:a]volume={voice_gain_db}dB[v];"
        f"[1:a]volume={instrumental_gain_db}dB[m];"
        "[v][m]amix=inputs=2:duration=longest:normalize=0[out]"
    )
    return run_ffmpeg_filter(
        [voice_track, instrumental_track],
        output_audio,
        filter_complex,
        ["-map", "[out]", "-c:a", "pcm_s16le"],
    )


def trim_audio_segment(
    input_audio: Path,
    output_audio: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
) -> Path:
    ffmpeg = require_executable("ffmpeg", "Install FFmpeg to cut audio segments.")
    ensure_parent_dir(output_audio)
    run_command(
        [
            ffmpeg,
            "-y",
            "-ss",
            str(max(0.0, start_seconds)),
            "-i",
            str(input_audio),
            "-t",
            str(max(0.0, duration_seconds)),
            "-c:a",
            "pcm_s16le",
            str(output_audio),
        ]
    )
    return output_audio


def normalize_audio(
    input_audio: Path,
    output_audio: Path,
    *,
    target_i: str = "-16",
    target_tp: str = "-1.5",
) -> Path:
    ffmpeg = require_executable("ffmpeg", "Install FFmpeg to normalize audio.")
    ensure_parent_dir(output_audio)
    run_command(
        [
            ffmpeg,
            "-y",
            "-i",
            str(input_audio),
            "-af",
            f"loudnorm=I={target_i}:TP={target_tp}",
            "-c:a",
            "pcm_s16le",
            str(output_audio),
        ]
    )
    return output_audio


def mux_audio_into_video(
    input_video: Path,
    audio_track: Path,
    output_video: Path,
    *,
    video_copy: bool = True,
) -> Path:
    ffmpeg = require_executable("ffmpeg", "Install FFmpeg to mux media.")
    ensure_parent_dir(output_video)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(input_video),
        "-i",
        str(audio_track),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
    ]
    command.extend(["-c:v", "copy" if video_copy else "libx264", "-c:a", "aac", "-b:a", "192k", str(output_video)])
    run_command(command)
    return output_video

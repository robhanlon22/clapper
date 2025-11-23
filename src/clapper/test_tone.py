"""Play a simple test tone for audio debugging."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from numpy.typing import NDArray

import numpy as np
import sounddevice as sd


def build_tone(
    frequency: float,
    duration: float,
    sample_rate: int,
    level: float,
    fade: float,
) -> NDArray[np.float32]:
    """Return a mono tone with optional fade-in/out to avoid clicks."""
    samples = max(1, int(duration * sample_rate))
    t: NDArray[np.float32] = np.linspace(
        0, duration, samples, endpoint=False, dtype=np.float32
    )
    tone: NDArray[np.float32] = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    tone *= np.float32(level)

    if fade > 0:
        fade_samples = min(int(fade * sample_rate), tone.size // 2)
        if fade_samples > 0:
            ramp: NDArray[np.float32] = np.linspace(
                0.0, 1.0, fade_samples, endpoint=False, dtype=np.float32
            )
            tone[:fade_samples] *= ramp
            tone[-fade_samples:] *= ramp[::-1]
    return tone


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a test tone.")
    parser.add_argument(
        "--freq", type=float, default=440.0, help="Frequency in Hz (default: 440)."
    )
    parser.add_argument(
        "--duration", type=float, default=1.0, help="Tone duration in seconds."
    )
    parser.add_argument(
        "--level", type=float, default=0.2, help="Amplitude 0.0-1.0 (default: 0.2)."
    )
    parser.add_argument(
        "--sample-rate", type=int, default=44_100, help="Sample rate in Hz."
    )
    parser.add_argument(
        "--fade",
        type=float,
        default=0.01,
        help="Seconds for fade-in/out to reduce clicks (default: 0.01).",
    )
    parser.add_argument(
        "--loop", action="store_true", help="Repeat the tone until Ctrl+C."
    )
    return parser.parse_args(argv)


def play_tone(args: argparse.Namespace) -> None:
    tone = build_tone(
        frequency=args.freq,
        duration=args.duration,
        sample_rate=args.sample_rate,
        level=args.level,
        fade=args.fade,
    )
    try:
        if args.loop:
            print("Playing tone on loop. Press Ctrl+C to stop.", file=sys.stderr)
            while True:
                sd.play(tone, args.sample_rate)
                sd.wait()
        else:
            sd.play(tone, args.sample_rate)
            sd.wait()
    except KeyboardInterrupt:
        pass


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    play_tone(args)


if __name__ == "__main__":
    main()

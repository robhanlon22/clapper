"""Play a simple test tone for audio debugging."""

from __future__ import annotations

import sys
from typing import Optional, Sequence

import click
import numpy as np
import sounddevice as sd
from numpy.typing import NDArray


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


def play_tone(
    freq: float,
    duration: float,
    sample_rate: int,
    level: float,
    fade: float,
    loop: bool,
) -> None:
    tone = build_tone(
        frequency=freq,
        duration=duration,
        sample_rate=sample_rate,
        level=level,
        fade=fade,
    )
    try:
        if loop:
            print("Playing tone on loop. Press Ctrl+C to stop.", file=sys.stderr)
            while True:
                sd.play(tone, sample_rate)
                sd.wait()
        else:
            sd.play(tone, sample_rate)
            sd.wait()
    except KeyboardInterrupt:
        pass


CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.command(context_settings=CONTEXT_SETTINGS, help="Play a simple test tone.")
@click.option(
    "--freq", type=float, default=440.0, show_default=True, help="Frequency in Hz."
)
@click.option(
    "--duration",
    type=float,
    default=1.0,
    show_default=True,
    help="Tone duration in seconds.",
)
@click.option(
    "--level", type=float, default=0.2, show_default=True, help="Amplitude 0.0-1.0."
)
@click.option(
    "--sample-rate",
    type=int,
    default=44_100,
    show_default=True,
    help="Sample rate in Hz.",
)
@click.option(
    "--fade",
    type=float,
    default=0.01,
    show_default=True,
    help="Seconds for fade-in/out to reduce clicks.",
)
@click.option("--loop", is_flag=True, help="Repeat the tone until Ctrl+C.")
def cli(
    freq: float,
    duration: float,
    level: float,
    sample_rate: int,
    fade: float,
    loop: bool,
) -> None:
    play_tone(
        freq=freq,
        duration=duration,
        sample_rate=sample_rate,
        level=level,
        fade=fade,
        loop=loop,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    cli.main(
        args=list(argv) if argv is not None else None,
        prog_name="python -m clapper.test_tone",
    )


if __name__ == "__main__":
    main()

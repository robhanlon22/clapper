from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray


def clap_like(
    block_size: int = 1024,
    amplitude: float = 0.8,
    seed: int = 0,
) -> NDArray[np.float32]:
    """Generate a broadband, impulsive block that resembles a hand clap."""
    rng = np.random.default_rng(seed)

    impulse: NDArray[np.float32] = np.zeros(block_size, dtype=np.float32)
    impulse[0] = amplitude

    noise = np.asarray(rng.standard_normal(block_size), dtype=np.float32)
    noise *= amplitude * 0.25
    envelope = np.exp(-np.linspace(0.0, 5.0, block_size, dtype=np.float32))

    clap = impulse + noise * envelope
    clipped = np.clip(clap, -1.0, 1.0).astype(np.float32, copy=False)
    return cast(NDArray[np.float32], clipped)


def low_tone(
    *,
    frequency: float = 180.0,
    amplitude: float = 0.8,
    block_size: int = 1024,
    sample_rate: int = 44_100,
) -> NDArray[np.float32]:
    """Generate a boomy low-frequency tone that should not count as a clap."""
    t = np.arange(block_size, dtype=np.float32) / float(sample_rate)
    sine = amplitude * np.sin(2 * np.pi * frequency * t)
    wave: NDArray[np.float32] = np.asarray(sine, dtype=np.float32)
    return wave


__all__ = ["clap_like", "low_tone"]

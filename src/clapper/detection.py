from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator

if TYPE_CHECKING:  # pragma: no cover - circular import protection for type checking
    from .cli import CliOptions


class DetectorConfig(BaseModel):
    sample_rate: int
    block_size: int
    warmup_seconds: float
    threshold_multiplier: float
    min_absolute_peak: float
    double_clap_min: float
    double_clap_max: float
    min_clap_interval: float
    noise_floor_halflife: float
    min_crest_factor: float = 3.5
    min_spectral_centroid_hz: float = 900.0

    model_config = ConfigDict(frozen=True)

    @field_validator("double_clap_min", "double_clap_max", "min_clap_interval")
    @classmethod
    def validate_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Values must be positive.")
        return float(value)

    @field_validator("double_clap_max", mode="after")
    @classmethod
    def validate_double_clap_max(
        cls,
        value: float,
        info: ValidationInfo,
    ) -> float:
        double_min = info.data.get("double_clap_min")
        if double_min is not None and value <= double_min:
            raise ValueError("double_clap_max must be greater than double_clap_min.")
        return value

    @field_validator("min_clap_interval", mode="after")
    @classmethod
    def validate_min_clap_interval(
        cls,
        value: float,
        info: ValidationInfo,
    ) -> float:
        double_min = info.data.get("double_clap_min")
        double_max = info.data.get("double_clap_max")
        if double_max is not None and value >= double_max:
            raise ValueError("min_clap_interval must be less than double_clap_max.")
        if double_min is not None and value > double_min:
            raise ValueError(
                "min_clap_interval should not exceed double_clap_min "
                "or double claps become impossible."
            )
        return value

    @field_validator("min_crest_factor", "min_spectral_centroid_hz")
    @classmethod
    def validate_clap_shape_thresholds(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Clap shape thresholds must be positive.")
        return float(value)


class DoubleClapDetector:
    def __init__(self, config: DetectorConfig, now: float) -> None:
        self.config = config
        self.noise_floor = config.min_absolute_peak
        self.last_clap_at = -1e9
        self.pending_first_clap: float | None = None
        self.ready_at = now + config.warmup_seconds
        self._freq_cache: dict[int, NDArray[np.float64]] = {}

    def _update_noise_floor(self, energy: float, duration: float) -> None:
        halflife = self.config.noise_floor_halflife
        if halflife <= 0:
            return
        # Exponential decay toward the most recent block's energy.
        alpha = 1.0 - np.exp(-duration / halflife)
        self.noise_floor = (1.0 - alpha) * self.noise_floor + alpha * energy

    def _spectral_centroid(self, samples: NDArray[np.float32]) -> float:
        if samples.size <= 1:
            return 0.0

        freqs = self._freq_cache.get(samples.size)
        if freqs is None:
            freqs = cast(
                NDArray[np.float64],
                np.asarray(
                    np.fft.rfftfreq(
                        samples.size,
                        d=1.0 / float(self.config.sample_rate),
                    ),
                    dtype=np.float64,
                ),
            )
            self._freq_cache[samples.size] = freqs

        real_samples = samples.astype(np.float64, copy=False)
        power = np.square(np.abs(np.fft.rfft(real_samples)))
        total_power = float(np.sum(power))
        if total_power <= 0.0:
            return 0.0

        return float(np.dot(freqs, power) / total_power)

    def _is_clap_shaped(
        self,
        samples: NDArray[np.float32],
        rms: float,
        peak: float,
    ) -> bool:
        # Claps are short, spiky, and broadband; filter out boomy or sustained peaks.
        if rms <= 0.0 or peak <= 0.0:
            return False

        crest_factor = peak / max(rms, 1e-12)
        if crest_factor < self.config.min_crest_factor:
            return False

        centroid = self._spectral_centroid(samples)
        return centroid >= self.config.min_spectral_centroid_hz

    def process_block(self, samples: NDArray[np.float32], now: float) -> bool:
        """Return True when a double clap is recognized."""
        if samples.size == 0:
            return False
        duration = samples.size / float(self.config.sample_rate)
        rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
        peak = float(np.max(np.abs(samples)))
        self._update_noise_floor(rms, duration)

        if now < self.ready_at:
            return False

        threshold = max(
            self.config.min_absolute_peak,
            self.noise_floor * self.config.threshold_multiplier,
        )
        if peak < threshold:
            return False

        if (now - self.last_clap_at) < self.config.min_clap_interval:
            return False

        if not self._is_clap_shaped(samples, rms, peak):
            return False

        self.last_clap_at = now
        if self.pending_first_clap is None:
            self.pending_first_clap = now
            return False

        delta = now - self.pending_first_clap
        if self.config.double_clap_min <= delta <= self.config.double_clap_max:
            self.pending_first_clap = None
            return True

        # Treat this clap as a fresh first clap when the gap is too wide.
        self.pending_first_clap = now
        return False


def build_detector_config(args: "CliOptions") -> DetectorConfig:
    double_min, double_max = args.double_window
    return DetectorConfig(
        sample_rate=args.sample_rate,
        block_size=args.block_size,
        warmup_seconds=args.warmup,
        threshold_multiplier=args.threshold_multiplier,
        min_absolute_peak=args.min_absolute_peak,
        min_clap_interval=args.clap_cooldown,
        double_clap_min=double_min,
        double_clap_max=double_max,
        noise_floor_halflife=args.noise_floor_halflife,
    )


default_config = DetectorConfig(
    sample_rate=44_100,
    block_size=1024,
    warmup_seconds=0.3,
    threshold_multiplier=6.0,
    min_absolute_peak=0.04,
    double_clap_min=0.16,
    double_clap_max=0.65,
    min_clap_interval=0.12,
    noise_floor_halflife=2.0,
    min_crest_factor=3.5,
    min_spectral_centroid_hz=900.0,
)

__all__ = [
    "DetectorConfig",
    "DoubleClapDetector",
    "build_detector_config",
    "default_config",
]

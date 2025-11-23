from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from clapper.cli import CliOptions
from clapper.detection import DoubleClapDetector, build_detector_config, default_config
from tests.audio_samples import clap_like, low_tone


def test_double_clap_detector_detects_within_window() -> None:
    detector = DoubleClapDetector(default_config, now=0.0)
    samples = clap_like(default_config.block_size)

    assert detector.process_block(samples, now=0.29) is False  # still warming up
    assert detector.process_block(samples, now=0.31) is False  # first clap
    assert detector.process_block(samples, now=0.5) is True  # second clap in window


def test_double_clap_detector_rejects_sustained_or_low_tone() -> None:
    detector = DoubleClapDetector(default_config, now=0.0)

    sustained = np.full(default_config.block_size, 0.5, dtype=np.float32)
    assert detector.process_block(sustained, now=0.31) is False

    bassy = low_tone(block_size=default_config.block_size)
    assert detector.process_block(bassy, now=0.5) is False


def test_build_detector_config_validates_window() -> None:
    with pytest.raises(ValidationError):
        args = CliOptions(
            command=["echo"],
            device=None,
            sample_rate=default_config.sample_rate,
            block_size=default_config.block_size,
            threshold_multiplier=default_config.threshold_multiplier,
            min_absolute_peak=default_config.min_absolute_peak,
            double_window=(0.5, 0.2),
            warmup=default_config.warmup_seconds,
            clap_cooldown=default_config.min_clap_interval,
            noise_floor_halflife=default_config.noise_floor_halflife,
        )
        build_detector_config(args)


def test_build_detector_config_rejects_cooldown_overlaps() -> None:
    with pytest.raises(ValidationError):
        args = CliOptions(
            command=["echo"],
            device=None,
            sample_rate=44_100,
            block_size=1024,
            threshold_multiplier=6.0,
            min_absolute_peak=0.04,
            double_window=(0.16, 0.2),
            warmup=0.3,
            clap_cooldown=0.2,
            noise_floor_halflife=default_config.noise_floor_halflife,
        )

        build_detector_config(args)


def test_build_detector_config_rejects_cooldown_above_min() -> None:
    with pytest.raises(ValidationError):
        args = CliOptions(
            command=["echo"],
            device=None,
            sample_rate=44_100,
            block_size=1024,
            threshold_multiplier=6.0,
            min_absolute_peak=0.04,
            double_window=(0.16, 0.2),
            warmup=0.3,
            clap_cooldown=0.17,
            noise_floor_halflife=default_config.noise_floor_halflife,
        )

        build_detector_config(args)

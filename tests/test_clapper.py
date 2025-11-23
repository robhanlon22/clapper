from __future__ import annotations

import logging
import queue
from typing import Callable, Iterable, Sequence, cast
from unittest import mock

import numpy as np
import pytest
import sounddevice as sd
from click.testing import CliRunner

from clapper import (
    CliOptions,
    DetectorConfig,
    DoubleClapDetector,
    ProcessToggler,
    build_detector_config,
    listen_and_toggle,
    make_cli,
)


def _make_stream(callbacks: Iterable[Callable[[], None]]) -> mock.MagicMock:
    """Create a mock InputStream context manager that runs callbacks on enter."""
    stream = mock.MagicMock()

    def _enter() -> mock.Mock:
        for cb in callbacks:
            cb()
        return stream

    stream.__enter__.side_effect = _enter

    def _exit(
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> bool:
        return False

    stream.__exit__.side_effect = _exit
    return stream


def test_double_clap_detector_detects_within_window() -> None:
    config = DetectorConfig()
    detector = DoubleClapDetector(config, now=0.0)
    samples = np.ones(1024, dtype=np.float32)

    assert detector.process_block(samples, now=0.29) is False  # still warming up
    assert detector.process_block(samples, now=0.31) is False  # first clap
    assert detector.process_block(samples, now=0.5) is True  # second clap in window


def test_build_detector_config_validates_window() -> None:
    args = CliOptions(
        command=["echo"],
        device=None,
        sample_rate=44_100,
        block_size=1024,
        threshold_multiplier=6.0,
        min_absolute_peak=0.04,
        double_window=(0.5, 0.2),
        warmup=0.3,
        clap_cooldown=0.12,
    )
    with pytest.raises(SystemExit):
        build_detector_config(args)


def test_build_detector_config_rejects_cooldown_overlaps() -> None:
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
    )

    with pytest.raises(SystemExit):
        build_detector_config(args)


def test_build_detector_config_rejects_cooldown_above_min() -> None:
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
    )

    with pytest.raises(SystemExit):
        build_detector_config(args)


def test_cli_passes_flags_through_to_command() -> None:
    received: list[Sequence[str]] = []

    def fake_listener(opts: CliOptions) -> None:
        received.append(opts.command)

    runner = CliRunner()
    result = runner.invoke(
        make_cli(fake_listener),
        ["python", "app.py", "--debug", "-v"],
        standalone_mode=False,
    )
    assert result.exit_code == 0
    assert received == [["python", "app.py", "--debug", "-v"]]


def test_listen_and_toggle_processes_double_event() -> None:
    # Time values: creation at 0.0 -> ready_at = 0.3; callbacks at 1.0 and 1.2.
    time_values = iter([0.0, 1.0, 1.2])

    def time_fn() -> float:
        return next(time_values)

    toggler_mock = mock.create_autospec(ProcessToggler, instance=True)
    toggler_mock.toggle.side_effect = [True]  # first double event starts

    def build_stream(**kwargs: object) -> mock.Mock:
        callback = cast(
            Callable[[np.ndarray, int, dict[str, object], sd.CallbackFlags], None],
            kwargs["callback"],
        )

        def fire() -> None:
            data = np.ones((1024, 1), dtype=np.float32)
            callback(data, data.shape[0], {}, sd.CallbackFlags(0))

        return _make_stream([fire, fire])

    logger = mock.create_autospec(logging.Logger)

    listen_and_toggle(
        CliOptions(
            command=["echo", "hi"],
            device=None,
            sample_rate=44_100,
            block_size=1024,
            threshold_multiplier=6.0,
            min_absolute_peak=0.04,
            double_window=(0.16, 0.65),
            warmup=0.3,
            clap_cooldown=0.12,
        ),
        stream_factory=build_stream,
        time_fn=time_fn,
        toggler_factory=lambda cmd: cast(ProcessToggler, toggler_mock),
        event_queue=queue.SimpleQueue(),
        poll_timeout=0.1,
        max_events=1,
        logger=logger,
    )

    toggler_mock.toggle.assert_called_once()
    toggler_mock.stop.assert_called_once()

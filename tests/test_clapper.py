from __future__ import annotations

import logging
import queue
from typing import Callable, Iterable, Sequence, cast
from unittest import mock

import click
import numpy as np
import pytest
import sounddevice as sd
from click.testing import CliRunner
from pydantic import ValidationError

from clapper import (
    CliOptions,
    DoubleClapDetector,
    ProcessToggler,
    build_audio_callback,
    build_detector_config,
    cli,
    default_config,
    listen_and_toggle,
    make_cli,
    main as clapper_main,
    process_event_loop,
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
    detector = DoubleClapDetector(default_config, now=0.0)
    samples = np.ones(1024, dtype=np.float32)

    assert detector.process_block(samples, now=0.29) is False  # still warming up
    assert detector.process_block(samples, now=0.31) is False  # first clap
    assert detector.process_block(samples, now=0.5) is True  # second clap in window


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


def test_cli_passes_flags_through_to_command() -> None:
    received: list[Sequence[str]] = []

    def fake_listener(opts: CliOptions) -> None:
        received.append(opts.command)

    runner = CliRunner()
    result = runner.invoke(
        make_cli(fake_listener),
        ["--", "python", "app.py", "--debug", "-v"],
        standalone_mode=False,
    )
    assert result.exit_code == 0
    assert received == [["python", "app.py", "--debug", "-v"]]


def test_cli_reports_detector_validation_errors() -> None:
    def fake_listener(opts: CliOptions) -> None:
        build_detector_config(opts)

    runner = CliRunner()
    result = runner.invoke(
        make_cli(fake_listener),
        ["--double-window", "0.5", "0.2", "--", "echo"],
        standalone_mode=False,
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, click.ClickException)
    assert "double_clap_min" in str(result.exception)


def test_main_requires_command_separator(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_cli_main = mock.Mock()
    monkeypatch.setattr(cli, "main", fake_cli_main)

    with pytest.raises(SystemExit) as excinfo:
        clapper_main(["echo"])

    assert excinfo.value.code == 2
    fake_cli_main.assert_not_called()
    assert "--" in capsys.readouterr().err


def test_main_allows_command_with_separator(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cli_main = mock.Mock()
    monkeypatch.setattr(cli, "main", fake_cli_main)

    clapper_main(["--", "echo", "hi"])

    fake_cli_main.assert_called_once_with(
        args=["--", "echo", "hi"],
        prog_name="clapper",
    )


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
            sample_rate=default_config.sample_rate,
            block_size=default_config.block_size,
            threshold_multiplier=default_config.threshold_multiplier,
            min_absolute_peak=default_config.min_absolute_peak,
            double_window=(
                default_config.double_clap_min,
                default_config.double_clap_max,
            ),
            warmup=default_config.warmup_seconds,
            clap_cooldown=default_config.min_clap_interval,
            noise_floor_halflife=default_config.noise_floor_halflife,
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


def test_build_audio_callback_queues_double_event() -> None:
    detector = mock.create_autospec(DoubleClapDetector, instance=True)
    detector.process_block.return_value = True
    events: queue.SimpleQueue[str] = queue.SimpleQueue()
    logger = mock.create_autospec(logging.Logger)

    callback = build_audio_callback(detector, logger, events, time_fn=lambda: 1.0)

    indata = np.ones((4, 1), dtype=np.float32)
    callback(indata, 4, {}, sd.CallbackFlags(1))

    assert events.get_nowait() == "double"
    detector.process_block.assert_called_once()
    logger.warning.assert_called_once()


def test_process_event_loop_toggles_and_logs() -> None:
    events: queue.SimpleQueue[str] = queue.SimpleQueue()
    events.put("double")
    events.put("double")

    toggler = mock.create_autospec(ProcessToggler, instance=True)
    toggler.toggle.side_effect = [True, False]
    logger = mock.create_autospec(logging.Logger)

    processed = process_event_loop(
        events=events,
        toggler=toggler,
        logger=logger,
        command=["echo", "hi"],
        poll_timeout=0.01,
        max_events=2,
        processed=0,
    )

    assert processed == 2
    assert toggler.toggle.call_count == 2
    assert logger.info.call_count == 2

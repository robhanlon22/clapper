from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Iterable, cast
from unittest import mock

import numpy as np
import sounddevice as sd

from clapper.cli import CliOptions
from clapper.detection import DoubleClapDetector, default_config
from clapper.events import build_audio_callback, listen_and_toggle, process_event_loop
from clapper.process import ProcessToggler
from tests.audio_samples import clap_like


def _make_stream(callbacks: Iterable[Callable[[], None]]) -> mock.Mock:
    """Create a mock InputStream context manager that runs callbacks on enter."""
    stream = mock.Mock()

    def _enter() -> mock.Mock:
        for cb in callbacks:
            cb()
        return stream

    stream.configure_mock(
        __enter__=mock.Mock(side_effect=_enter),
        __exit__=mock.Mock(return_value=False),
    )
    return stream


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
            samples = clap_like(default_config.block_size).reshape(-1, 1)
            data = np.asarray(samples, dtype=np.float32)
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


def test_process_event_loop_stops_after_stop_event_set() -> None:
    events: queue.SimpleQueue[str] = queue.SimpleQueue()
    events.put("double")

    toggler = mock.create_autospec(ProcessToggler, instance=True)
    toggler.toggle.return_value = True
    logger = mock.create_autospec(logging.Logger)
    stop_event = threading.Event()
    stop_event.set()

    processed = process_event_loop(
        events=events,
        toggler=toggler,
        logger=logger,
        command=["echo", "hi"],
        poll_timeout=0.01,
        max_events=None,
        processed=0,
        stop_event=stop_event,
    )

    assert processed == 1
    toggler.toggle.assert_called_once()


def test_listen_and_toggle_drains_events_on_keyboard_interrupt() -> None:
    events: queue.SimpleQueue[str] = queue.SimpleQueue()
    events.put("double")

    def build_stream(**kwargs: object) -> mock.Mock:
        raise KeyboardInterrupt

    toggler_mock = mock.create_autospec(ProcessToggler, instance=True)
    toggler_mock.toggle.return_value = True
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
        time_fn=lambda: 0.0,
        toggler_factory=lambda cmd: cast(ProcessToggler, toggler_mock),
        event_queue=events,
        poll_timeout=0.01,
        logger=logger,
    )

    toggler_mock.toggle.assert_called_once()
    toggler_mock.stop.assert_called_once()

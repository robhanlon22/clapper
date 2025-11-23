from __future__ import annotations

import queue
from types import TracebackType
from typing import Callable, Iterable, List, Literal, Sequence, cast

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


class FakeStream:
    def __init__(self, callbacks: Iterable[Callable[[], None]]) -> None:
        self._callbacks = list(callbacks)

    def __enter__(self) -> "FakeStream":
        for callback in self._callbacks:
            callback()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        return False


class FakeToggler(ProcessToggler):
    def __init__(self, command: Sequence[str]) -> None:
        super().__init__(command)
        self.calls: List[str] = []

    def toggle(self) -> bool:
        self.calls.append("toggle")
        return True

    def stop(self, timeout: float = 0.0) -> None:
        self.calls.append("stop")


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


def test_cli_passes_flags_through_to_command() -> None:
    received: List[Sequence[str]] = []

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

    events = queue.SimpleQueue[str]()
    toggler = FakeToggler(["echo", "hi"])

    def build_stream(**kwargs: object) -> FakeStream:
        callback = cast(
            Callable[[np.ndarray, int, dict[str, object], sd.CallbackFlags], None],
            kwargs["callback"],
        )

        def fire() -> None:
            data = np.ones((1024, 1), dtype=np.float32)
            callback(data, data.shape[0], {}, sd.CallbackFlags(0))

        return FakeStream([fire, fire])

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
        toggler_factory=lambda cmd: toggler,
        event_queue=events,
        max_events=1,
    )

    assert toggler.calls == ["toggle", "stop"]

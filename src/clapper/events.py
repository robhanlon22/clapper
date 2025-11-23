from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from typing import Any, Callable, ContextManager, Optional, Sequence, TYPE_CHECKING

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray

from .detection import DoubleClapDetector, build_detector_config
from .process import ProcessToggler, format_command

if TYPE_CHECKING:  # pragma: no cover - runtime import cycle guard
    from .cli import CliOptions

LOGGER = logging.getLogger(__name__)


def build_audio_callback(
    detector: DoubleClapDetector,
    logger: logging.Logger,
    events: queue.SimpleQueue[str],
    time_fn: Callable[[], float],
) -> Callable[[NDArray[np.float32], int, Any, sd.CallbackFlags], None]:
    def audio_callback(
        indata: NDArray[np.float32],
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            # Stash the status string to stderr so the loop can continue listening.
            logger.warning("[clapper] Audio status: %s", status)
        now = time_fn()
        # Use the first channel only; mono is sufficient for clap detection.
        samples: NDArray[np.float32] = np.asarray(indata[:, 0], dtype=np.float32)
        if detector.process_block(samples, now):
            events.put("double")

    return audio_callback


def process_event_loop(
    *,
    events: queue.SimpleQueue[str],
    toggler: ProcessToggler,
    logger: logging.Logger,
    command: Sequence[str],
    poll_timeout: float,
    max_events: Optional[int],
    processed: int,
    stop_event: Optional[threading.Event] = None,
) -> int:
    while True:
        if stop_event is not None and stop_event.is_set() and events.empty():
            break
        try:
            event = events.get(timeout=poll_timeout)
        except queue.Empty:
            if stop_event is not None and stop_event.is_set():
                break
            continue
        if event == "stop":
            if stop_event is not None:
                stop_event.set()
            break
        if event == "double":
            starting = toggler.toggle()
            state = "started" if starting else "stopped"
            logger.info(
                "[clapper] Double clap detected, %s %s",
                state,
                format_command(command),
            )
            processed += 1
            if max_events is not None and processed >= max_events:
                break

    return processed


def listen_and_toggle(
    args: "CliOptions",
    *,
    stream_factory: Callable[..., ContextManager[Any]] = sd.InputStream,
    time_fn: Callable[[], float] = time.monotonic,
    toggler_factory: Callable[[Sequence[str]], ProcessToggler] = ProcessToggler,
    event_queue: Optional[queue.SimpleQueue[str]] = None,
    poll_timeout: float = 0.5,
    max_events: Optional[int] = None,
    logger: logging.Logger = LOGGER,
) -> None:
    if not args.command:
        logger.error("No command specified. Example: clapper -- python app.py")
        sys.exit(1)

    config = build_detector_config(args)
    detector = DoubleClapDetector(config, now=time_fn())
    events: queue.SimpleQueue[str] = event_queue or queue.SimpleQueue()
    stop_event = threading.Event()
    toggler = toggler_factory(args.command)
    audio_callback = build_audio_callback(detector, logger, events, time_fn)

    device = args.device
    logger.info(
        f"Listening for double claps (device={device or 'default'}, "
        f"double window={config.double_clap_min:.2f}-{config.double_clap_max:.2f}s, "
        f"threshold x{config.threshold_multiplier:g}).",
    )
    logger.info("Target command: %s", format_command(args.command))
    logger.info("Clap twice to toggle. Ctrl+C to quit.")

    processed = 0
    stop_sent = False

    def signal_stop() -> None:
        nonlocal stop_sent
        if stop_sent:
            return
        stop_sent = True
        stop_event.set()
        events.put("stop")

    try:
        with stream_factory(
            channels=1,
            callback=audio_callback,
            samplerate=config.sample_rate,
            blocksize=config.block_size,
            device=device,
            dtype="float32",
        ):
            processed = process_event_loop(
                events=events,
                toggler=toggler,
                logger=logger,
                command=args.command,
                poll_timeout=poll_timeout,
                max_events=max_events,
                processed=processed,
                stop_event=stop_event,
            )
    except KeyboardInterrupt:
        signal_stop()
    finally:
        signal_stop()
        processed = process_event_loop(
            events=events,
            toggler=toggler,
            logger=logger,
            command=args.command,
            poll_timeout=poll_timeout,
            max_events=max_events,
            processed=processed,
            stop_event=stop_event,
        )
        toggler.stop()


__all__ = ["build_audio_callback", "process_event_loop", "listen_and_toggle"]

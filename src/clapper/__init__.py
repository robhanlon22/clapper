from __future__ import annotations

import logging
import os
import queue
import shlex
import signal
import subprocess
import sys
import time
from typing import Any, Callable, ContextManager, Iterable, Optional, Sequence

import click
import numpy as np
import sounddevice as sd
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    ValidationInfo,
    field_validator,
)


# The detector is intentionally simple: track the noise floor, look for peaks
# that jump above it, and treat two qualifying peaks in close succession as a
# double clap.


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


class CliOptions(BaseModel):
    command: Sequence[str]
    device: Optional[str]
    sample_rate: int
    block_size: int
    threshold_multiplier: float
    min_absolute_peak: float
    double_window: tuple[float, float]
    warmup: float
    clap_cooldown: float
    noise_floor_halflife: float

    model_config = ConfigDict(frozen=True)

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: Sequence[str]) -> Sequence[str]:
        if not value:
            raise ValueError("A command is required.")
        return value


LOGGER = logging.getLogger(__name__)


class DoubleClapDetector:
    def __init__(self, config: DetectorConfig, now: float) -> None:
        self.config = config
        self.noise_floor = config.min_absolute_peak
        self.last_clap_at = -1e9
        self.pending_first_clap: Optional[float] = None
        self.ready_at = now + config.warmup_seconds

    def _update_noise_floor(self, energy: float, duration: float) -> None:
        halflife = self.config.noise_floor_halflife
        if halflife <= 0:
            return
        # Exponential decay toward the most recent block's energy.
        alpha = 1.0 - np.exp(-duration / halflife)
        self.noise_floor = (1.0 - alpha) * self.noise_floor + alpha * energy

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
        is_clap = (
            peak >= threshold
            and (now - self.last_clap_at) >= self.config.min_clap_interval
        )
        if not is_clap:
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


class ProcessToggler:
    def __init__(self, command: Sequence[str]) -> None:
        if not command:
            raise ValueError("ProcessToggler requires a non-empty command.")
        self.command = list(command)
        self.process: Optional[subprocess.Popen[bytes]] = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self) -> None:
        if self.is_running():
            return
        creationflags = 0
        start_new_session = False
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            start_new_session = True

        self.process = subprocess.Popen(
            self.command,
            stdout=None,
            stderr=None,
            stdin=None,
            start_new_session=start_new_session,
            creationflags=creationflags,
        )

    def stop(self, timeout: float = 5.0) -> None:
        if not self.is_running() or self.process is None:
            self.process = None
            return

        proc = self.process
        try:
            if os.name == "nt":
                proc.terminate()
            else:
                os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            self.process = None
            return

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                proc.kill()
            else:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        finally:
            self.process = None

    def toggle(self) -> bool:
        """Toggle the process; returns True when starting, False when stopping."""
        if self.is_running():
            self.stop()
            return False
        self.start()
        return True


def build_detector_config(args: CliOptions) -> DetectorConfig:
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


def format_command(cmd: Iterable[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def listen_and_toggle(
    args: CliOptions,
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
            )
    except KeyboardInterrupt:
        pass
    finally:
        toggler.stop()


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
) -> int:
    while True:
        try:
            event = events.get(timeout=poll_timeout)
        except queue.Empty:
            continue
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


class ContextSettings(BaseModel):
    help_option_names: list[str]
    allow_interspersed_args: bool

    model_config = ConfigDict(frozen=True)


context_settings = ContextSettings(
    help_option_names=["-h", "--help"],
    allow_interspersed_args=False,
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
)


def make_cli(
    listener: Callable[[CliOptions], None] = listen_and_toggle,
) -> click.Command:
    @click.command(
        context_settings=context_settings.model_dump(),
        help="Toggle a program on/off with a double clap.",
    )
    @click.argument("command", nargs=-1, required=True, type=click.UNPROCESSED)
    @click.option(
        "--device",
        type=str,
        default=None,
        show_default="default input device",
        help="Audio input device name or index.",
    )
    @click.option(
        "--sample-rate",
        type=int,
        default=default_config.sample_rate,
        show_default=True,
        help="Sample rate used for capture.",
    )
    @click.option(
        "--block-size",
        type=int,
        default=default_config.block_size,
        show_default=True,
        help="Frames per audio block.",
    )
    @click.option(
        "--threshold-multiplier",
        type=float,
        default=default_config.threshold_multiplier,
        show_default=True,
        help="How far above the ambient noise the peak must be to count as a clap.",
    )
    @click.option(
        "--min-absolute-peak",
        type=float,
        default=default_config.min_absolute_peak,
        show_default=True,
        help="Hard minimum peak amplitude needed to count as a clap.",
    )
    @click.option(
        "--double-window",
        nargs=2,
        type=float,
        metavar="MIN MAX",
        default=(default_config.double_clap_min, default_config.double_clap_max),
        show_default=True,
        help="Acceptable gap (seconds) between claps that forms a double clap.",
    )
    @click.option(
        "--warmup",
        type=float,
        default=default_config.warmup_seconds,
        show_default=True,
        help="Seconds to learn the room noise floor before detecting claps.",
    )
    @click.option(
        "--clap-cooldown",
        type=float,
        default=default_config.min_clap_interval,
        show_default=True,
        help="Minimum seconds between individual clap detections.",
    )
    @click.option(
        "--noise-floor-halflife",
        type=float,
        default=default_config.noise_floor_halflife,
        show_default=True,
        help="Seconds for the noise floor estimate to halve.",
    )
    def _cli(
        command: tuple[str, ...],
        device: Optional[str],
        sample_rate: int,
        block_size: int,
        threshold_multiplier: float,
        min_absolute_peak: float,
        double_window: tuple[float, float],
        warmup: float,
        clap_cooldown: float,
        noise_floor_halflife: float,
    ) -> None:
        """Entry point for the clapper CLI."""
        try:
            options = CliOptions(
                command=list(command),
                device=device,
                sample_rate=sample_rate,
                block_size=block_size,
                threshold_multiplier=threshold_multiplier,
                min_absolute_peak=min_absolute_peak,
                double_window=double_window,
                warmup=warmup,
                clap_cooldown=clap_cooldown,
                noise_floor_halflife=noise_floor_halflife,
            )
            listener(options)
        except ValidationError as exc:
            raise click.ClickException(str(exc)) from exc

    return _cli


cli = make_cli()


_SEPARATOR_ERROR = (
    "Separate clapper options from the target command with '--'. "
    "Example: clapper -- python app.py"
)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = list(argv) if argv is not None else sys.argv[1:]

    # Allow Click to handle help/usage when no arguments are provided.
    if not args or any(arg in {"-h", "--help"} for arg in args):
        cli.main(args=args if argv is not None else None, prog_name="clapper")
        return

    if "--" not in args:
        click.echo(f"Error: {_SEPARATOR_ERROR}", err=True)
        sys.exit(2)

    cli.main(args=args, prog_name="clapper")

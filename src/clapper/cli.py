from __future__ import annotations

import logging
import sys
from typing import Callable, Iterable, Optional, Sequence

import click
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from .detection import default_config
from .events import listen_and_toggle


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


class ContextSettings(BaseModel):
    help_option_names: Iterable[str]
    allow_interspersed_args: bool

    model_config = ConfigDict(frozen=True)


context_settings = ContextSettings(
    help_option_names={"-h", "--help"},
    allow_interspersed_args=False,
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
    if not args or any(arg in context_settings.help_option_names for arg in args):
        cli.main(args=args if argv is not None else None, prog_name="clapper")
        return

    if "--" not in args:
        click.echo(f"Error: {_SEPARATOR_ERROR}", err=True)
        sys.exit(2)

    cli.main(args=args, prog_name="clapper")


__all__ = [
    "CliOptions",
    "ContextSettings",
    "context_settings",
    "make_cli",
    "cli",
    "main",
]

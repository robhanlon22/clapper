from __future__ import annotations

from .cli import CliOptions, cli, main, make_cli
from .detection import (
    DetectorConfig,
    DoubleClapDetector,
    build_detector_config,
    default_config,
)
from .events import build_audio_callback, listen_and_toggle, process_event_loop
from .process import ProcessToggler, format_command

__all__ = [
    "CliOptions",
    "DetectorConfig",
    "DoubleClapDetector",
    "ProcessToggler",
    "build_audio_callback",
    "build_detector_config",
    "cli",
    "default_config",
    "format_command",
    "listen_and_toggle",
    "main",
    "make_cli",
    "process_event_loop",
]

from __future__ import annotations

import os
import signal
import subprocess
from typing import Sequence


class ProcessToggler:
    def __init__(self, command: Sequence[str]) -> None:
        if not command:
            raise ValueError("ProcessToggler requires a non-empty command.")
        self.command = list(command)
        self.process: subprocess.Popen[bytes] | None = None

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


def format_command(cmd: Sequence[str]) -> str:
    import shlex

    return " ".join(shlex.quote(part) for part in cmd)


__all__ = ["ProcessToggler", "format_command"]

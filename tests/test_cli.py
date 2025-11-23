from __future__ import annotations

from unittest import mock

import click
import pytest
from click.testing import CliRunner

from clapper.cli import CliOptions, cli, make_cli, main as clapper_main
from clapper.detection import build_detector_config


def test_cli_passes_flags_through_to_command() -> None:
    listener = mock.Mock()

    runner = CliRunner()
    result = runner.invoke(
        make_cli(listener),
        ["--", "python", "app.py", "--debug", "-v"],
        standalone_mode=False,
    )
    assert result.exit_code == 0
    listener.assert_called_once()
    called_opts = listener.call_args.args[0]
    assert isinstance(called_opts, CliOptions)
    assert list(called_opts.command) == ["python", "app.py", "--debug", "-v"]


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
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
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

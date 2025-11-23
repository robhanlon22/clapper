# Repository Guidelines

## Project Structure & Module Organization

- Core library lives in `src/clapper/`; entry point is `clapper:main`, with clap
  detection logic in `DoubleClapDetector` and process control in
  `ProcessToggler`.
- CLI helpers and the test tone module sit beside the core; sounddevice stubs
  are in `src/sounddevice/`.
- Tests live in `tests/` (pytest style, `test_*.py`). Utility scripts for setup
  and tooling live in `scripts/`.

## Build, Test, and Development Commands

- Setup runs automatically on directory enter via mise hooks (`mise install`
  then `scripts/mise-postinstall.sh`, which installs pre-commit hooks, runs
  `uv sync`, and executes `brew bundle` to pull PortAudio).
- Run the app: `clapper -- <command>` (or `python -m clapper -- <command>`). The
  target command must be separated from clapper flags with `--`; otherwise the
  CLI exits with code 2 and prints a hint. Test tone:
  `python -m clapper.test_tone --freq 880 --duration 0.5 --loop`.
- Lint/format: `ruff check` and `ruff format` (run manually if you want; the
  pre-commit hook runs them automatically on commit).
- Type check: `mypy`.
- Test suite: `pytest`.

## Coding Style & Naming Conventions

- Python 3.14+; prefer 4-space indentation and type hints everywhere. Ruff
  enforces imports/style; ruff format controls layout.
- Keep modules small; favour pure functions; avoid side effects at import time.
- CLI flags and public options should stay snake_case internally; command names
  stay kebab-case when exposed.
- Shell scripts follow POSIX sh with `set -eufo pipefail` and 2-space indents
  (see `.editorconfig`).

## Testing Guidelines

- Add or update tests in `tests/` using pytest; name files `test_<area>.py` and
  functions `test_<behavior>`.
- Use fixtures/mocks for audio and subprocess interactions; see
  `tests/test_clapper.py` for patterns.
- When changing detection thresholds or timing, add regression cases that pin
  expected clap/delay behavior.

## Commit & Pull Request Guidelines

- Use short, imperative subjects; in the body explain why the change is needed,
  paste the triggering error for `git log --grep`, and note any one-off
  diagnostic commands (e.g., `file --mime`, `iconv`) so others can retrace the
  fix.
- PRs should describe user-facing behavior changes, testing performed, and any
  tuning to detection parameters; attach screenshots or logs if CLI output
  changed.
- Commit hook already runs pre-commit (ruff, pytest, mypy); rerun
  `pre-commit run --all-files` only if you want to reproduce hook output.

## Environment & Operational Tips

- PortAudio is required for audio capture; verify devices with
  `python -m sounddevice`.
- Default thresholds assume a quiet room; note tuned defaults in PRs.

Clapper - double clap to start/stop a program.

## Quick start

- Install deps (PortAudio is required; on macOS run `brew install portaudio` or
  `brew bundle` to apply the Brewfile).
- Run: `uv run clapper -- <command to toggle>` Examples:
  - `uv run clapper -- python app.py`
  - `uv run clapper -- open -a \"Music\"`
- Clap twice to start the command, clap twice again to stop it. Ctrl+C to quit.

### Play a test tone

- `uv run python -m clapper.test_tone --freq 880 --duration 0.5 --loop`

## Flags you might tune

- `--device`: input device name or index (see `python -m sounddevice` to list).
- `--threshold-multiplier`: how far above ambient noise a clap must be (default
  6.0).
- `--min-absolute-peak`: hard minimum amplitude to count as a clap (default
  0.04).
- `--double-window MIN MAX`: gap (seconds) between claps for a double (default
  0.16-0.65).
- `--sample-rate`, `--block-size`: adjust capture fidelity vs. CPU (defaults
  44100 Hz / 1024 frames).

## Notes

- The first 0.3s is a warmup to learn the room's noise floor.
- Processes start in their own session; children are killed with the parent on
  stop.
- If clap detection is flaky, lower `--threshold-multiplier`, raise
  `--min-absolute-peak`, or widen `--double-window`.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

An LLM-driven animatronic head: it listens (microphone), sees (webcam), talks (local TTS) and moves (servos on a PCA9685, or simulated in the control panel). Only the LLM is remote, through OpenRouter. Developed on Windows; deployed on a Raspberry Pi 5 with `pi/install.sh`.

## Commands

uv project, Python 3.11.

- Run: `uv run python -m robot_head`. Control panel at http://127.0.0.1:8765.
- Test instance without mic, camera or sound: `WEB_PORT=8766 uv run python -m robot_head --text --no-camera --mute`. A robot started from this checkout may already hold port 8765; leave it running. A test instance still uses the real `robot.db` and `.env` in the project folder (everything lives under `config.PROJECT_DIR`); to keep `robot.db` untouched, start it from a small script that sets `robot_head.__main__.PROJECT_DIR` to a scratch folder and calls `main()` with `--supervised`.
- Tests: `uv run pytest -q`. One test: `uv run pytest tests/test_memory.py::test_two_people_are_told_apart -q`
- Lint (not configured in the project): `uv run --with ruff ruff check robot_head tests --line-length 130`
- Devices and voices: `--list-devices`, `--choose-devices` (saves to `.env`), `--list-voices`
- Pi-only extras (smbus2 for the servo board): `uv sync --extra pi`. Check that new dependencies have aarch64 wheels: `uv export --frozen --no-hashes --extra pi > req.txt` and `uv pip compile req.txt --python-platform aarch64-manylinux_2_36 --python-version 3.11 --only-binary :all:`

`.env` holds the real OpenRouter key. Never print it; check it with `bool(load_settings().openrouter_api_key)`. Edit `.env` only through `devices.save_settings`, which keeps every other line.

## Architecture

**Process.** `python -m robot_head` is a launcher (`supervise()` in `__main__.py`) that runs the robot as a child process (`-u -m robot_head ... --supervised`) and starts it again when it exits with code 75. That's how the control panel's Restart works, and why settings are only loaded in the child.

**Two layers.** The LLM is too slow to drive motors, so:
- `reflexes.py` runs at 30 Hz on its own thread: face tracking, saccades, blinking, breathing, gestures, the jaw following the speech audio. Each frame it produces a `Pose`, seven joints with normalized values, and hands it to `motors.write()` and the `ServoMonitor` (`wiring.py`) that feeds the control panel.
- The brain only nudges that layer through cues written inline in its streamed reply: `[smug]`, `[nod]`, `[look_left]`; `[disgusted lids=0.3 tilt=0.1 roll=-0.2 look=0,-0.3]` invents a new expression, saved with source `llm`; `[name=Alex]` tells it the name of the person in view. `cues.py` (`SentenceStream`, `parse_segment`) splits the stream into sentences, releasing a long opening clause early, and each cue fires when its sentence starts playing.

**Conversation** (`conversation.py`, on the main thread): wake name, follow-up window, echo detection, barge-in, unprompted remarks, queued suggestions, pause. `Voice` (`voice.py`, Kokoro or Piper) renders the next sentence while the current one plays. `audio.py` runs one full-duplex sounddevice stream at 48 kHz in 10 ms frames, with livekit's WebRTC echo cancellation and a `SpeechDetector`. Utterances are resampled to 16 kHz for `ears.py` (Parakeet through onnx-asr by default, Whisper as an option).

**Brain** (`brain.py`): the openai SDK pointed at OpenRouter. `reasoning.effort` goes in `extra_body`. The system prompt is assembled on every turn: the personality prompt, the `RULES` for speaking and cues, then two callables the startup code plugs in, `brain.seen` (the observer's activity timeline) and `brain.remembered` (the memory context). `ask_json()` serves memory reflection.

**Seeing.** `camera.py` reads frames on a background thread and detects faces with OpenCV YuNet. `people.py` turns faces into SFace fingerprints (128 numbers, stored in the database, never as images); an unknown face becomes a new person after three sightings in a row. `observer.py` combines N frames into one grid picture, asks the model for a one-line description (or `SAME` / `NOBODY`), builds a timeline and triggers recommendations through `conversation.suggest()`.

**Memory** (`memory.py`): everything heard, said and seen goes into a journal tagged with the closest person. After a conversation goes quiet, one JSON LLM call per person turns it into memories (`add` / `update` / `forget` / `name` / `summary`). Memories with source `user` were written in the control panel and are never changed by the model. The reflection prompt has to state today's date with the year, or the model guesses one.

**Tools** (`tools.py`, `catalog.py`): `ToolHub` keeps every enabled MCP server connected on its own thread and asyncio loop, using the `mcp` 2.x `Client`. That covers online servers over streamable HTTP or SSE, with an OAuth provider whose sign-in page returns to `/api/tools/oauth/callback`, and local ones over stdio through `npx`, `uvx` or `docker`. `brain.reply()` loops: stream, speak any text, run the tool calls through `toolbox.run()`, then ask again (up to `MAX_TOOL_ROUNDS`). Tool calls and results stay in the history. A tool that doesn't only look things up (`only_looks()`: the server's read-only hint, or a guess from its name) returns `NOT_YET` until the same call comes back in a later user turn; `from_user=False` turns don't count. Skills are SKILL.md files; enabled ones are listed in the system prompt and read with the built-in `read_skill` tool, and their scripts are never run. `catalog.py` maps registry `server.json` entries to ways of running them, with fields for the keys they need, and resolves skills.sh and GitHub links to raw SKILL.md files.

**Storage** (`store.py`): one SQLite file, `robot.db`, holds personalities, expressions, state, people, faces, memories, journal, servo calibration, tool servers (with their keys and OAuth tokens, never sent to the page), tool choices and skills. Access goes through a lock. The schema is `create table if not exists`, with no migrations, so a change can add new tables but not alter old ones.

**Control panel.** `control.py`'s `Controller` holds every runtime operation and the `SETTINGS` list. Each `.env` key has a group, a type, help text, and whether it applies live or needs a restart. `web.py` is a thin FastAPI layer, served on a background thread, that turns `ValueError` into HTTP 400. `web/index.html` is a single page of vanilla JavaScript: the face SVG drawn from the `/api/live` server-sent-events pose stream, the wiring diagram, servo calibration, personalities, expressions, memory and settings tabs.

**Motors** (`motors.py`): `JOINTS` holds the default channels and angle ranges. Calibration saved from the Wiring tab (the `servos` table) overrides them through `calibrated()`. The `Motors` base class turns poses into angles, sends only changes, and supports `hold()` for trying angles live. `Pca9685` drives the board directly over I2C with smbus2.

Models (YuNet, SFace, Kokoro, Piper voices, speech recognition) download on first use into `models/` (`downloads.fetch`).

## Conventions

- Everything should be configurable from the control panel. A new tunable needs a field in `config.Settings` and `load_settings()`, a `SettingSpec` in `control.py`, live application in the Controller's settings update if it can apply live, and a commented line in `.env.example`.
- Tests use stand-ins and never touch real devices or the network: `FakeBrain` and `FakeVoice` in `test_conversation.py`, the `robot` Controller fixture in `test_control.py` (other test files import it with `from test_control import robot  # noqa: F401`), `FakeBus` in `test_servos.py`, and `tests/tool_server.py`, a tiny real MCP server that `test_tools.py` starts over stdio.
- The README stays general: it doesn't name the default character, which comes from `personality.md`.
- `pi/install.sh` and `pi/robot-head.service` must keep LF line endings (`.gitattributes`). Git on Windows doesn't record the executable bit, so set it with `git update-index --chmod=+x`. The README says `bash pi/install.sh`, which works either way.

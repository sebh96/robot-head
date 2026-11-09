"""Run the head: ``uv run python -m robot_head`` (see --help)."""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading

import cv2

from .audio import AudioEngine
from .brain import Brain
from .config import PROJECT_DIR, load_settings
from .control import Controller, starter_personalities
from .conversation import Conversation
from .devices import choose_devices, describe_audio, describe_camera, find_audio_device, list_devices
from .memory import Memory
from .motors import Pca9685Motors, SimulatedMotors, calibrated
from .observer import Observer
from .people import People
from .reflexes import Reflexes
from .store import Store
from .tools import SEARCH_ENGINES, Internet, ToolHub
from .voice import Voice, make_speech, print_voices
from .wiring import ServoMonitor

RESTART_EXIT_CODE = 75  # the robot asks its launcher to start it again


def supervise() -> None:
    """Run the robot as a child process and start it again whenever it asks to be restarted.

    A fresh process re-reads .env and reopens every device, so restarting applies any setting,
    and after a ``git pull`` it runs the new code too.
    """
    while True:
        # Unbuffered, so its lines show up right away in logs too (like a Raspberry Pi service's journal).
        child = subprocess.Popen([sys.executable, "-u", "-m", "robot_head", *sys.argv[1:], "--supervised"])
        try:
            code = child.wait()
        except KeyboardInterrupt:  # Ctrl+C reaches the child too; let it shut down cleanly
            child.wait()
            return
        if code != RESTART_EXIT_CODE:
            sys.exit(code)
        print("\nRestarting...\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-driven animatronic head prototype.")
    parser.add_argument("--text", action="store_true", help="type instead of talking (no microphone needed)")
    parser.add_argument("--no-camera", action="store_true", help="run without the webcam")
    parser.add_argument("--choose-devices", action="store_true", help="pick microphone, speaker and camera, then exit")
    parser.add_argument("--list-devices", action="store_true", help="list microphones, speakers and cameras and exit")
    parser.add_argument("--list-voices", action="store_true", help="list the voices of TTS_ENGINE and exit")
    parser.add_argument("--mute", action="store_true", help="start with the voice muted (turn it on in the control panel)")
    parser.add_argument("--no-web", action="store_true", help="don't start the control panel")
    parser.add_argument("--supervised", action="store_true", help=argparse.SUPPRESS)  # set by supervise()
    args = parser.parse_args()

    if args.choose_devices:
        try:
            choose_devices()
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled, nothing saved.")
        return
    if args.list_devices:
        list_devices()
        return
    if args.list_voices:
        print_voices(load_settings().tts_engine)
        return
    if not args.supervised:
        # Before load_settings(): settings loaded here would be inherited and shadow .env changes.
        supervise()
        return
    settings = load_settings()
    web = settings.web_ui and not args.no_web
    if not settings.openrouter_api_key:
        env_file = PROJECT_DIR / ".env"
        if web:
            print("No OpenRouter API key yet: paste it under Settings in the control panel.")
        elif not env_file.exists():
            sys.exit(f"There is no {env_file} yet. Copy .env.example to .env and put your OpenRouter key in it.")
        else:
            sys.exit(f"OPENROUTER_API_KEY is empty in {env_file}. Paste your key right after the = and save the file.")
    try:
        mic = find_audio_device(settings.mic_device, "input")
        speaker = find_audio_device(settings.speaker_device, "output")
    except ValueError as error:
        sys.exit(str(error))

    print(f"Speaker:    {describe_audio(speaker, 'output')}")
    if not args.text:
        print(f"Microphone: {describe_audio(mic, 'input')}")
    if not args.no_camera:
        print(f"Camera:     {describe_camera(settings.camera_index)}")
    if settings.mic_device is None and settings.speaker_device is None:
        print("Not the right devices? Run with --choose-devices to pick them.")

    camera = None
    if not args.no_camera:
        from .camera import Camera

        try:
            camera = Camera(settings.camera_index, flip=settings.camera_flip)
        except RuntimeError as error:
            sys.exit(str(error))
    store = Store(PROJECT_DIR / "robot.db")
    joints = calibrated(store.servos())
    motors_problem = None
    if settings.motors == "pca9685":
        try:
            motors = Pca9685Motors(joints)
        except ImportError:
            sys.exit("MOTORS=pca9685 needs the Raspberry Pi extras: uv sync --extra pi")
        except OSError as error:  # not wired up yet: keep going, so the rest can be tried and set up
            motors_problem = (
                f"Can't reach the servo board ({error}), so the servos are only simulated. Is I2C on, and is the PCA9685 "
                "wired to Pi pins 1, 3, 5 and 6? `i2cdetect -y 1` should list it at 40. Restart once it's connected."
            )
            print(motors_problem)
            motors = SimulatedMotors(joints)
    else:
        motors = SimulatedMotors(joints)
    monitor = ServoMonitor(motors)
    reflexes = Reflexes(motors, camera, monitor)
    stop = threading.Event()
    reflex_thread = threading.Thread(target=reflexes.run, args=(stop,), daemon=True)
    reflex_thread.start()

    store.seed(starter_personalities(settings))
    personality = store.active_personality()
    expressions = store.expressions()
    reflexes.set_expressions(expressions)

    engine = AudioEngine(
        mic, speaker, listen=not args.text, barge_in=settings.barge_in, silence_seconds=settings.silence_seconds
    )
    engine.muted = args.mute
    engine.start()
    try:
        speech = make_speech(
            settings.tts_engine, personality.voice or settings.tts_voice, personality.speed or settings.tts_speed
        )
    except ValueError as error:
        sys.exit(str(error))
    voice = Voice(engine, reflexes.set_jaw, speech)
    brain = Brain(
        settings.openrouter_api_key,
        settings.model,
        settings.reasoning_effort,
        personality.prompt,
        [expression.name for expression in expressions],
    )
    voice_mode = not args.text
    conversation = Conversation(
        brain, voice, reflexes, camera,
        wake_name=personality.wake_name if voice_mode else "",
        follow_up_seconds=settings.follow_up_seconds,
        remark_after_seconds=settings.remark_after_seconds if voice_mode else 0,
        send_frames=settings.send_camera_frames,
    )
    brain.observe_model = settings.observe_model
    brain.recommend_about = settings.recommend_about or brain.recommend_about
    observer = None
    if camera is not None:
        observer = Observer(
            camera,
            brain.observe,
            every_seconds=settings.observe_seconds,
            frames=settings.observe_frames,
            change_threshold=settings.observe_change,
            recommend_every_seconds=settings.recommend_every_seconds,
            long_activity_minutes=settings.recommend_long_minutes,
            on_recommend=lambda timeline: conversation.suggest(brain.recommendation_prompt(timeline)),
        )
        brain.seen = observer.summary
        threading.Thread(target=observer.run, args=(stop,), daemon=True).start()
    people = None
    if camera is not None:
        try:
            people = People(store, camera, match=settings.face_match)
        except (OSError, cv2.error) as error:
            print(f"Face recognition is off, it couldn't start: {error}")
        else:
            people.enabled = settings.recognize_people
            threading.Thread(target=people.run, args=(stop,), daemon=True).start()
    memory = Memory(
        store,
        brain.ask_json,
        people,
        enabled=settings.memory,
        reflect_minutes=settings.memory_reflect_minutes,
        in_prompt=settings.memory_in_prompt,
        keep_days=settings.memory_keep_days,
    )
    brain.remembered = memory.context
    panel_host = "127.0.0.1" if settings.web_host in ("0.0.0.0", "::") else settings.web_host
    toolbox = ToolHub(store, timeout=settings.tool_timeout_seconds,
                      callback=f"http://{panel_host}:{settings.web_port}/api/tools/oauth/callback")
    toolbox.enabled = settings.tools
    toolbox.internet = Internet(
        search=settings.web_search,
        read=settings.web_read,
        engine=settings.web_search_engine if settings.web_search_engine in SEARCH_ENGINES else "exa",
        results=settings.web_search_results,
    )
    toolbox.start()  # connects to the MCP servers added in the control panel, in the background
    brain.toolbox = toolbox
    brain.location = settings.location
    threading.Thread(target=memory.run, args=(stop,), daemon=True).start()
    controller = Controller(
        settings, store, brain, voice, reflexes, conversation, engine, camera, monitor, observer, people, memory, toolbox
    )
    if motors_problem:
        controller.log("error", motors_problem)
    restart = threading.Event()

    def request_restart() -> None:
        restart.set()
        engine.stop_playback(interrupted=True)  # cut the current sentence short
        stop.set()

    controller.on_restart = request_restart

    if web:
        from .web import create_app, port_free, serve

        if port_free(settings.web_host, settings.web_port):
            serve(create_app(controller), settings.web_host, settings.web_port)
            print(f"Control panel (face, wiring, settings): http://{settings.web_host}:{settings.web_port}")
        else:
            print(f"Port {settings.web_port} is taken; is the robot already running? Starting without the control panel.")
            web = False

    listen = None
    if voice_mode:
        from .ears import make_ears, save_recording

        print("Loading speech recognition (the first run downloads the model)...")
        try:
            ears = make_ears(settings.stt_engine, settings.stt_model, settings.stt_device, settings.stt_language)
        except ValueError as error:
            sys.exit(str(error))
        print("Measuring background noise, stay quiet for a second...")
        engine.calibrate()

        def listen(timeout: float) -> str | None:
            audio = engine.next_utterance(timeout)
            if audio is None or conversation.paused:  # paused: whatever it heard is dropped unheard
                return None
            text = ears.transcribe(audio)
            if controller.save_recordings:
                save_recording(audio, text)
            return text
    else:
        def read_typing() -> None:
            for line in sys.stdin:
                if line.strip():
                    conversation.inbox.put(line.strip())
            if not web:
                stop.set()  # typing ended and there's no control panel to keep it busy

        threading.Thread(target=read_typing, daemon=True).start()

    call_it = f' Say "{personality.wake_name}" to get its attention.' if personality.wake_name and voice_mode else ""
    print(f"Ready: {personality.name}, thinking with {settings.model}.{call_it}")
    print(("Type to talk. " if args.text else "") + "Press Ctrl+C to quit.")
    try:
        conversation.run(stop, listen)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        reflex_thread.join(timeout=1)
        engine.close()
        if camera:
            camera.close()
        motors.close()
        toolbox.close()
        store.close()
    if restart.is_set():
        sys.exit(RESTART_EXIT_CODE)


if __name__ == "__main__":
    main()

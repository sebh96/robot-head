"""Run the head: ``uv run python -m robot_head`` (see --help)."""

from __future__ import annotations

import argparse
import sys
import threading

from .audio import AudioEngine
from .brain import Brain
from .config import PROJECT_DIR, load_settings
from .control import Controller, starter_personalities
from .conversation import Conversation
from .devices import choose_devices, describe_audio, describe_camera, find_audio_device, list_devices
from .motors import Pca9685Motors, SimulatedMotors
from .reflexes import Reflexes
from .store import Store
from .voice import Voice, make_speech, print_voices
from .wiring import ServoMonitor


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-driven animatronic head prototype.")
    parser.add_argument("--text", action="store_true", help="type instead of talking (no microphone needed)")
    parser.add_argument("--no-camera", action="store_true", help="run without the webcam")
    parser.add_argument("--choose-devices", action="store_true", help="pick microphone, speaker and camera, then exit")
    parser.add_argument("--list-devices", action="store_true", help="list microphones, speakers and cameras and exit")
    parser.add_argument("--list-voices", action="store_true", help="list the voices of TTS_ENGINE and exit")
    parser.add_argument("--mute", action="store_true", help="start with the voice muted (turn it on in the control panel)")
    parser.add_argument("--no-web", action="store_true", help="don't start the control panel")
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
    settings = load_settings()
    if args.list_voices:
        print_voices(settings.tts_engine)
        return
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
    motors = Pca9685Motors() if settings.motors == "pca9685" else SimulatedMotors()
    monitor = ServoMonitor()
    reflexes = Reflexes(motors, camera, monitor)
    stop = threading.Event()
    reflex_thread = threading.Thread(target=reflexes.run, args=(stop,), daemon=True)
    reflex_thread.start()

    store = Store(PROJECT_DIR / "robot.db")
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
    controller = Controller(settings, store, brain, voice, reflexes, conversation, engine, camera, monitor)

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
            if audio is None:
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

    if web:
        from .web import create_app, serve

        serve(create_app(controller), settings.web_host, settings.web_port)
        print(f"Control panel (face, wiring, settings): http://{settings.web_host}:{settings.web_port}")
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
        store.close()


if __name__ == "__main__":
    main()

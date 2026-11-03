"""The control panel: a small web app the robot serves itself (http://127.0.0.1:8765 by default)."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .control import Controller

PAGE = Path(__file__).parent / "web" / "index.html"
POSE_FPS = 30
CAMERA_FPS = 12


class PersonalityIn(BaseModel):
    name: str
    prompt: str
    wake_name: str = ""
    voice: str = ""
    speed: float | None = None


class ExpressionIn(BaseModel):
    lids: float = 0.7
    tilt: float = 0.0
    roll: float = 0.0
    look_x: float = 0.0
    look_y: float = 0.0


class Text(BaseModel):
    text: str


class Name(BaseModel):
    name: str


class Flag(BaseModel):
    value: bool


def _checked(action: Callable, *args):
    """Run a controller action, turning its complaints into a 400 the page can show."""
    try:
        return action(*args)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from None


def create_app(controller: Controller) -> FastAPI:
    app = FastAPI(title="robot-head control panel", docs_url=None, redoc_url=None)

    @app.get("/")
    def page() -> FileResponse:
        return FileResponse(PAGE)

    @app.get("/api/state")
    def state(after: int = 0) -> dict:
        return controller.state(after)

    @app.get("/api/meta")
    def meta() -> dict:
        return controller.meta()

    @app.get("/api/live")
    def live() -> StreamingResponse:
        """The head's pose and every servo's state, as Server-Sent Events, for the face and wiring views."""

        def frames():
            while True:
                yield f"data: {json.dumps(controller.live_frame())}\n\n"
                time.sleep(1 / POSE_FPS)

        return StreamingResponse(frames(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/camera.mjpg")
    def camera() -> StreamingResponse:
        """What the camera sees, with the tracked face boxed, as an MJPEG stream an <img> can show."""
        if controller.camera is None:
            raise HTTPException(status_code=404, detail="Running without a camera.")

        def frames():
            while True:
                jpeg = controller.camera.preview_jpeg()
                if jpeg:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                time.sleep(1 / CAMERA_FPS)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/servos")
    def servos() -> list[dict]:
        return controller.monitor.servos() if controller.monitor else []

    @app.post("/api/say")
    def say(body: Text) -> None:
        controller.say(body.text)

    @app.post("/api/cue")
    def cue(body: Name) -> None:
        controller.cue(body.name)

    @app.post("/api/mute")
    def mute(body: Flag) -> None:
        controller.set_muted(body.value)

    @app.get("/api/personalities")
    def personalities() -> list[dict]:
        return controller.personalities()

    @app.post("/api/personalities")
    def create_personality(body: PersonalityIn) -> dict:
        return _checked(controller.save_personality, body.model_dump())

    @app.put("/api/personalities/{personality_id}")
    def update_personality(personality_id: int, body: PersonalityIn) -> dict:
        return _checked(controller.save_personality, body.model_dump(), personality_id)

    @app.delete("/api/personalities/{personality_id}")
    def delete_personality(personality_id: int) -> None:
        _checked(controller.delete_personality, personality_id)

    @app.post("/api/personalities/{personality_id}/activate")
    def activate(personality_id: int) -> None:
        _checked(controller.activate, personality_id)

    @app.get("/api/expressions")
    def expressions() -> list[dict]:
        return controller.expressions()

    @app.post("/api/expressions/preview")
    def preview(body: ExpressionIn) -> None:
        controller.preview(body.model_dump())

    @app.put("/api/expressions/{name}")
    def save_expression(name: str, body: ExpressionIn) -> dict:
        return _checked(controller.save_expression, name, body.model_dump())

    @app.delete("/api/expressions/{name}")
    def delete_expression(name: str) -> None:
        _checked(controller.delete_expression, name)

    @app.get("/api/settings")
    def settings() -> list[dict]:
        return controller.settings()

    @app.put("/api/settings")
    def update_settings(values: dict[str, str]) -> dict:
        return controller.update_settings(values)

    @app.get("/api/voices")
    def voices() -> list[str]:
        return controller.voices()

    @app.get("/api/models")
    def models() -> list[str]:
        return controller.models()

    return app


def serve(app: FastAPI, host: str, port: int) -> uvicorn.Server:
    """Run the control panel on a background thread."""
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    return server

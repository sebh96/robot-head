"""The control panel: a small web app the robot serves itself (http://127.0.0.1:8765 by default)."""

from __future__ import annotations

import html
import json
import queue
import socket
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .control import Controller

PAGE = Path(__file__).parent / "web" / "index.html"
POSE_FPS = 30
CAMERA_FPS = 12
SOUND_KEEPALIVE = 15.0  # seconds


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


class MemoryIn(BaseModel):
    text: str
    kind: str = "fact"
    person_id: int | None = None


class PersonIn(BaseModel):
    name: str


class MergeIn(BaseModel):
    into: int


class ServoIn(BaseModel):
    channel: int
    start: float = Field(alias="from")
    to: float


class HoldIn(BaseModel):
    angle: float | None = None


class ToolServerIn(BaseModel):
    # From the registry: name, title, description, the option picked and the answers to its fields.
    name: str = ""
    title: str = ""
    description: str = ""
    option: dict | None = None
    values: dict[str, str] | None = None
    # By address: url, and a header with a key if the service needs one.
    url: str = ""
    header: str = ""
    value: str = ""
    origin: str = ""  # where the page was opened, for sign-in pages to send people back to


class ToolServerUpdate(BaseModel):
    enabled: bool | None = None
    keys: dict[str, str] | None = None


class ToolChoiceIn(BaseModel):
    enabled: bool
    ask_first: bool


class SkillIn(BaseModel):
    source: str = ""  # a catalog entry or a link; or else write one:
    name: str = ""
    description: str = ""
    instructions: str = ""


class SkillUpdate(BaseModel):
    enabled: bool | None = None
    name: str = ""
    description: str = ""
    instructions: str = ""


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

    @app.get("/api/sound")
    def sound() -> StreamingResponse:
        """Its voice, every sentence as it starts playing, as Server-Sent Events for pages set to "Play here"."""
        listener = controller.sound.subscribe()

        def events():
            try:
                while True:
                    try:
                        event = listener.get(timeout=SOUND_KEEPALIVE)
                    except queue.Empty:
                        yield ": still here\n\n"  # keeps the connection open, and notices when the page is gone
                        continue
                    yield f"data: {json.dumps(event)}\n\n"
            finally:
                controller.sound.unsubscribe(listener)

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

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
        return controller.servos()

    @app.put("/api/servos/{name}")
    def save_servo(name: str, body: ServoIn) -> None:
        _checked(controller.save_servo, name, body.model_dump(by_alias=True))

    @app.delete("/api/servos/{name}")
    def reset_servo(name: str) -> None:
        _checked(controller.reset_servo, name)

    @app.post("/api/servos/{name}/hold")
    def hold_servo(name: str, body: HoldIn) -> None:
        _checked(controller.hold_servo, name, body.angle)

    @app.post("/api/say")
    def say(body: Text) -> None:
        controller.say(body.text)

    @app.post("/api/cue")
    def cue(body: Name) -> None:
        controller.cue(body.name)

    @app.post("/api/mute")
    def mute(body: Flag) -> None:
        controller.set_muted(body.value)

    @app.post("/api/restart")
    def restart() -> None:
        _checked(controller.restart)

    @app.post("/api/pause")
    def pause(body: Flag) -> None:
        controller.set_paused(body.value)

    @app.get("/api/people")
    def people() -> list[dict]:
        return controller.people_list()

    @app.put("/api/people/{person_id}")
    def rename_person(person_id: int, body: PersonIn) -> None:
        _checked(controller.rename_person, person_id, body.name)

    @app.delete("/api/people/{person_id}")
    def delete_person(person_id: int) -> None:
        controller.delete_person(person_id)

    @app.post("/api/people/{person_id}/merge")
    def merge_person(person_id: int, body: MergeIn) -> None:
        _checked(controller.merge_people, person_id, body.into)

    @app.get("/api/memories")
    def memories(person: int | None = None) -> list[dict]:
        """Memories about one person, or (without ``person``) the ones about nobody in particular."""
        return controller.memories(person)

    @app.post("/api/memories")
    def add_memory(body: MemoryIn) -> None:
        _checked(controller.save_memory, body.model_dump())

    @app.put("/api/memories/{memory_id}")
    def update_memory(memory_id: int, body: MemoryIn) -> None:
        _checked(controller.save_memory, body.model_dump(), memory_id)

    @app.delete("/api/memories/{memory_id}")
    def delete_memory(memory_id: int) -> None:
        controller.delete_memory(memory_id)

    @app.post("/api/memory/reflect")
    def reflect() -> None:
        _checked(controller.reflect_now)

    @app.get("/api/tools")
    def tools() -> dict:
        return controller.tools()

    @app.get("/api/tools/catalog")
    def tool_catalog(q: str = "", cursor: str = "") -> dict:
        """Servers from the official MCP Registry that match ``q``."""
        return _checked(controller.tool_catalog, q, cursor)

    @app.post("/api/tools/servers")
    def add_tool_server(body: ToolServerIn) -> dict:
        return {"id": _checked(controller.add_tool_server, body.model_dump())}

    @app.put("/api/tools/servers/{server_id}")
    def update_tool_server(server_id: int, body: ToolServerUpdate) -> None:
        _checked(controller.update_tool_server, server_id, body.model_dump())

    @app.delete("/api/tools/servers/{server_id}")
    def delete_tool_server(server_id: int) -> None:
        controller.delete_tool_server(server_id)

    @app.get("/api/tools/oauth/callback")
    def finish_sign_in(code: str = "", state: str = "", iss: str | None = None, error: str = "") -> HTMLResponse:
        """Where a server's sign-in page sends the person back to."""
        try:
            if error:
                raise ValueError(f"The sign-in page said: {error}")
            message, status = f"Back from signing in to {html.escape(controller.finish_sign_in(code, state, iss))}.", 200
        except ValueError as problem:
            message, status = html.escape(str(problem)), 400
        page = (
            '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<title>Robot sign-in</title><body style="font: 16px system-ui; background: #15171c; color: #e9e9ec; padding: 40px">'
            f"<p>{message}</p><p>You can close this tab: the Tools tab in the control panel shows when it's connected.</p></body>"
        )
        return HTMLResponse(page, status_code=status)

    @app.put("/api/tools/servers/{server_id}/tools/{tool}")
    def set_tool_choice(server_id: int, tool: str, body: ToolChoiceIn) -> None:
        controller.set_tool_choice(server_id, tool, body.enabled, body.ask_first)

    @app.get("/api/skills/catalog")
    def skill_catalog(source: str = "robot") -> dict:
        return controller.skill_catalog(source)

    @app.post("/api/skills")
    def add_skill(body: SkillIn) -> None:
        _checked(controller.add_skill, body.model_dump())

    @app.get("/api/skills/{skill_id}")
    def skill(skill_id: int) -> dict:
        return _checked(controller.skill, skill_id)

    @app.put("/api/skills/{skill_id}")
    def update_skill(skill_id: int, body: SkillUpdate) -> None:
        _checked(controller.update_skill, skill_id, body.model_dump())

    @app.delete("/api/skills/{skill_id}")
    def delete_skill(skill_id: int) -> None:
        controller.delete_skill(skill_id)

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


def port_free(host: str, port: int) -> bool:
    with socket.socket() as probe:
        if sys.platform != "win32":  # on Windows the option would let it share a port that is really in use
            # Like uvicorn: connections from before a restart linger for a minute and don't make the port taken.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def serve(app: FastAPI, host: str, port: int) -> uvicorn.Server:
    """Run the control panel on a background thread."""
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    return server

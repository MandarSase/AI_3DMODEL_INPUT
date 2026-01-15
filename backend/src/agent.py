#!/usr/bin/env python3
import json
import logging
import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,  # your installed version expects this
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)

from livekit.plugins import silero, deepgram, google, murf
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# -------------------------------------------------------------------------
# Config & Logging
# -------------------------------------------------------------------------
load_dotenv(".env.local")
logger = logging.getLogger("model_request_agent")
logger.setLevel(logging.DEBUG)
h = logging.StreamHandler()
h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(h)

# -------------------------------------------------------------------------
# Storage
# -------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
REQUEST_DIR = BASE / "shared-data" / "model_requests"
REQUEST_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# State model
# -------------------------------------------------------------------------
@dataclass
class RequestState:
    request_id: Optional[str] = None
    description: Optional[str] = None
    model_type: Optional[str] = None
    dimensions: Optional[str] = None
    material: Optional[str] = None
    extras: Dict[str, Any] = None
    waiting_for: Optional[str] = None

    def __post_init__(self):
        if self.extras is None:
            self.extras = {}

class UD:
    pass

# -------------------------------------------------------------------------
# Tools
# -------------------------------------------------------------------------
@function_tool
async def record_initial_request(ctx: RunContext, text: str) -> str:
    ud: UD = ctx.userdata
    ud.req.description = text.strip()
    ud.req.waiting_for = "model_type"
    return "Great! What type of 3D model do you want?"

@function_tool
async def record_model_type(ctx: RunContext, model_type: str) -> str:
    ud: UD = ctx.userdata
    ud.req.model_type = model_type.strip()
    ud.req.waiting_for = "dimensions"
    return "Understood. What dimensions should the model have?"

@function_tool
async def record_dimensions(ctx: RunContext, dimensions: str) -> str:
    ud: UD = ctx.userdata
    ud.req.dimensions = dimensions.strip()
    ud.req.waiting_for = "material"
    return "Got it. What material or texture should be used?"

@function_tool
async def record_material(ctx: RunContext, material: str) -> str:
    ud: UD = ctx.userdata
    ud.req.material = material.strip()
    ud.req.waiting_for = None
    return "Noted. Any extra details?"

@function_tool
async def record_extra(ctx: RunContext, key: str, value: str) -> str:
    ud: UD = ctx.userdata
    ud.req.extras[key] = value
    return "Extra detail added."

@function_tool
async def save_request(ctx: RunContext) -> str:
    ud: UD = ctx.userdata
    req = ud.req
    req_id = req.request_id or f"request_{len(list(REQUEST_DIR.glob('*.json'))) + 1}"
    req.request_id = req_id
    with open(REQUEST_DIR / f"{req_id}.json", "w", encoding="utf-8") as f:
        json.dump({
            "request_id": req.request_id,
            "description": req.description,
            "model_type": req.model_type,
            "dimensions": req.dimensions,
            "material": req.material,
            "extras": req.extras
        }, f, indent=2)
    return f"Saved your request as {req_id}!"

# -------------------------------------------------------------------------
# Agent
# -------------------------------------------------------------------------
class ModelRequestAgent(Agent):
    def __init__(self):
        instructions = """
You are an AI assistant that collects 3D model requirements.

Conversation flow:
1. Ask user for request description.
2. Use record_initial_request.
3. Ask for model_type → dimensions → material → extras.
4. When user says “save”, call save_request.

Always end with a question so the user continues speaking.
Keep responses short and friendly.
"""
        super().__init__(
            instructions=instructions,
            tools=[
                record_initial_request,
                record_model_type,
                record_dimensions,
                record_material,
                record_extra,
                save_request,
            ],
        )

# -------------------------------------------------------------------------
# Entrypoint helpers
# -------------------------------------------------------------------------
def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load(sensitivity=0.5)
        logger.debug("Silero VAD loaded in prewarm.")
    except Exception as e:
        proc.userdata["vad"] = None
        logger.warning("Could not load Silero VAD in prewarm, continuing without VAD: %s", e)

async def _try_session_run(session: AgentSession):
    """
    Try to run the session using the following strategy:
    1) Try session.run(user_input="") — required by many livekit-agents versions.
    2) If that raises TypeError (signature mismatch), try session.run().
    3) If that raises an API error from Gemini complaining about single-turn roles,
       fallback to session.run() as well.
    Returns True on success, False on failure.
    """
    # 1) Prefer explicit user_input for versions that require it.
    try:
        logger.debug("Attempting session.run(user_input='') ...")
        await session.run(user_input="")
        logger.info("session.run(user_input='') succeeded")
        return True
    except TypeError as te:
        logger.debug("session.run(user_input='') TypeError (signature mismatch): %s", te)
    except Exception as e:
        # Check if it's a Gemini/GenAI client error complaining about roles (400).
        msg = str(e)
        logger.debug("session.run(user_input='') raised: %s", msg)
        if "Please ensure that single turn requests end with a user role" in msg or "INVALID_ARGUMENT" in msg:
            logger.warning("Gemini single-turn role error detected; will retry without user_input.")
        else:
            # If it's some other error, log and still attempt fallback
            logger.warning("session.run(user_input='') failed; trying fallback: %s", e)

    # 2) Fallback attempt without kwargs
    try:
        logger.debug("Attempting session.run() fallback ...")
        await session.run()
        logger.info("session.run() succeeded")
        return True
    except Exception as e:
        logger.exception("session.run() fallback also failed: %s", e)
        return False

# -------------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------------
async def entrypoint(ctx: JobContext):
    ud = UD()
    ud.req = RequestState(waiting_for="initial")

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(voice="en-US-matthew"),
        vad=ctx.proc.userdata.get("vad"),
        turn_detection=MultilingualModel(),
        userdata=ud,
        preemptive_generation=False,
    )

    await ctx.connect()

    # Start session (use RoomInputOptions for your installed version)
    try:
        await session.start(
            agent=ModelRequestAgent(),
            room=ctx.room,
            room_input_options=RoomInputOptions(),
        )
        logger.info("session.start() succeeded")
    except Exception as e:
        logger.exception("session.start() failed. Check token grants (canPublish, canPublishData) and server: %s", e)
        return

    # small pause to let negotiation settle
    await asyncio.sleep(0.5)

    # say greeting
    try:
        await session.say("Hello! What 3D model request would you like to create?")
        logger.info("session.say() succeeded")
    except Exception as e:
        logger.exception("session.say() failed: %s", e)
        # continue — session may still run even if say failed

    # run session (tries both forms, handles Gemini single-turn role error)
    ok = await _try_session_run(session)
    if not ok:
        logger.error("session.run failed in all attempts. See logs above for details.")
        # stop session cleanly if possible
        try:
            await session.stop()
        except Exception:
            pass
        return

# -------------------------------------------------------------------------
# Run worker
# -------------------------------------------------------------------------
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

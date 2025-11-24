import logging
import json
import os
import re
from datetime import datetime
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
)
from livekit.agents import function_tool, RunContext

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# A local uploaded file path (kept for transparency / optional use)
UPLOADED_FILE_PATH = "/mnt/data/4b23f3bb-6765-4c4b-8e8d-6ee909236850.png"

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# -------------------------------------------------------------------------
# Paths / persistence
# -------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
LOGS_DIR = os.path.join(BASE_DIR, "wellness_logs")
os.makedirs(LOGS_DIR, exist_ok=True)


# -------------------------
# Filename / user helpers
# -------------------------
def sanitize_name(name: str) -> str:
    if not name:
        return "unknown"
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_-]", "", name)
    return name or "unknown"


def user_aggregate_file(safe_name: str) -> str:
    return os.path.join(LOGS_DIR, f"{safe_name}.json")


def user_session_filename(safe_name: str, ts: datetime) -> str:
    iso = ts.isoformat().replace(":", "-")
    return os.path.join(LOGS_DIR, f"{safe_name}_{iso}.json")


def list_user_files_prefix(safe_name: str):
    matches = []
    try:
        for fname in os.listdir(LOGS_DIR):
            if not fname.lower().endswith(".json"):
                continue
            if fname.lower().startswith(safe_name.lower()):
                matches.append(os.path.join(LOGS_DIR, fname))
    except Exception as e:
        logger.exception("error listing logs dir: %s", e)
    return matches


def load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.exception("failed to load json %s: %s", path, e)
        return None


def save_json(path: str, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception("failed to save json %s: %s", path, e)


def find_latest_user_file(safe_name: str):
    files = list_user_files_prefix(safe_name)
    if not files:
        return None, None
    try:
        files_sorted = sorted(files, key=lambda p: os.path.getmtime(p))
        latest = files_sorted[-1]
        data = load_json(latest)
        return latest, data
    except Exception as e:
        logger.exception("failed to pick latest file for %s: %s", safe_name, e)
        return None, None


# -------------------------------------------------------------------------
# Assistant: instruction ensures greeting first then asks name
# -------------------------------------------------------------------------
class Assistant(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are a warm, supportive, non-diagnostic Daily Wellness Companion.

MANDATES (must follow strictly):
1) At the start of every session, greet the user and then ask for their name.
   Example phrasing: "Hello! I'm your wellness companion. Before we begin, what's your name?"
   Do NOT ask any other wellness questions until the user provides their name.

2) When the user provides their name, call the backend function 'find_user_latest_log(name)'.
   - If the function returns a previous log, read that log and ask exactly one short personalized follow-up
     referencing the most relevant prior item (e.g., "Last time you mentioned trouble sleeping — did you sleep better today?").
   - If no previous log was found, say: "Nice to meet you, {name}. Let's start your first check-in."

3) After that, proceed to the standard short check-in flow, roughly in this order (adapt naturally):
   - "How are you feeling today?" (mood)
   - "What's your energy like right now?" (energy)
   - "Anything stressing you out at the moment?" (stress; user may say 'no')
   - "What are 1–3 things you'd like to get done today?" (objectives — allow comma separated)
   - "Is there anything you'd like to do for yourself today? (rest, exercise, hobby)" (optional self-care)

4) Offer 1–2 short, practical, non-medical suggestions (e.g., break tasks into small steps, take a 5-minute walk, 4 deep breaths).
   Do NOT provide medical or diagnostic claims. If asked for medical advice, politely refuse and recommend a professional.

5) Provide a concise recap with mood + 1–3 objectives and ask: "Does this sound right?"
   - If the user confirms, call 'save_wellness_entry(name, mood, energy, stress, objectives, self_care, agent_summary)'.
   - If the user asks to edit, accept edits for mood/energy/objectives/self-care, then save.

Keep language short, empathetic, and actionable.
"""
        )

    @function_tool
    async def find_user_latest_log(self, ctx: RunContext, name: str):
        if not name:
            return None
        safe = sanitize_name(name)
        path, data = find_latest_user_file(safe)
        if not path:
            return None
        return {"path": path, "data": data}

    @function_tool
    async def save_wellness_entry(
        self,
        ctx: RunContext,
        name: str,
        mood: str,
        energy: str,
        stress: str,
        objectives: list,
        self_care: str,
        agent_summary: str,
    ):
        try:
            safe = sanitize_name(name) if name else "unknown"
            ts = datetime.now()
            entry = {
                "timestamp": ts.isoformat(),
                "name": name or "",
                "mood": mood or "",
                "energy": energy or "",
                "stress": stress or "",
                "objectives": objectives or [],
                "self_care": self_care or "",
                "agent_summary": agent_summary or "",
            }

            # append to aggregate
            agg_path = user_aggregate_file(safe)
            agg_data = []
            if os.path.exists(agg_path):
                existing = load_json(agg_path)
                if isinstance(existing, list):
                    agg_data = existing
            agg_data.append(entry)
            save_json(agg_path, agg_data)

            # write per-session file
            session_path = user_session_filename(safe, ts)
            save_json(session_path, entry)

            return {"status": "ok", "aggregate": agg_path, "session": session_path, "entry": entry}
        except Exception as e:
            logger.exception("save_wellness_entry failed: %s", e)
            return {"status": "error", "message": str(e)}


# -------------------------------------------------------------------------
# PREWARM
# -------------------------------------------------------------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# -------------------------------------------------------------------------
# ENTRYPOINT
# -------------------------------------------------------------------------
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # Build voice pipeline (keep your existing choices)
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info("Usage: %s", usage_collector.get_summary())

    ctx.add_shutdown_callback(log_usage)

    # Create assistant
    agent = Assistant()

    # NOTE: removed agent.append_message(...) here because Agent doesn't implement it.
    # If you need to inject system context, include it in Assistant(instructions) or pass messages differently.

    # Start session
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    # Connect the worker to LiveKit room BEFORE attempting TTS
    await ctx.connect()
    logger.info("Connected to LiveKit room, now triggering initial greeting (TTS)")

    # Force the agent to speak the initial greeting and ask for name immediately.
    try:
        await session.say("Hello! I'm your wellness companion. Before we begin, what's your name?")
    except Exception:
        logger.debug("session.say failed or unavailable; model should ask naturally.")

    # Start the session conversation loop so the agent processes incoming audio and replies.
    try:
        await session.run()
    except Exception as e:
        logger.exception("session.run failed: %s", e)


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

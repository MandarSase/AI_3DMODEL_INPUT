import logging
import json
import os

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


logger = logging.getLogger("agent")

load_dotenv(".env.local")


# -------------------------------------------------------------------------
#   ASSISTANT CLASS
# -------------------------------------------------------------------------
class Assistant(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are a friendly coffee shop barista.
Ask the user for drink, size, milk, extras, and their name.
When enough information is provided, call save_order_to_json().
Be conversational and concise.
"""
        )

    # --------------------- FUNCTION TOOL (JSON SAVER) ---------------------
    @function_tool
    async def save_order_to_json(
        self,
        ctx: RunContext,
        drink: str,
        size: str,
        milk: str,
        extras: str,
        name: str
    ):
        """Save the customer's coffee order to a JSON file."""

        order = {
            "drink": drink,
            "size": size,
            "milk": milk,
            "extras": extras,
            "name": name,
        }

        # Save under backend/src/orders
        orders_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "orders"))
        os.makedirs(orders_dir, exist_ok=True)

        safe_name = name.replace(" ", "_")
        file_path = os.path.join(orders_dir, f"order_{safe_name}.json")

        with open(file_path, "w") as f:
            json.dump(order, f, indent=4)

        return f"Order saved successfully for {name}! File created."


# -------------------------------------------------------------------------
#   PREWARM
# -------------------------------------------------------------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# -------------------------------------------------------------------------
#   ENTRYPOINT
# -------------------------------------------------------------------------
async def entrypoint(ctx: JobContext):

    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Voice pipeline setup
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
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
        logger.info(f"Usage: {usage_collector.get_summary()}")
    ctx.add_shutdown_callback(log_usage)

    # Start barista assistant
    agent = Assistant()

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()


# -------------------------------------------------------------------------
#   MAIN
# -------------------------------------------------------------------------
if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm
        )
    )

import logging
import json
import os
from pathlib import Path
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
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

LOG_PATH = Path("wellness_log.json")


# -------------------------------------------------
#   WELLNESS STATE MACHINE
# -------------------------------------------------

class WellnessState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = {
            "timestamp": None,
            "mood": None,
            "energy": None,
            "stress": None,
            "goals": [],
            "summary": None,
        }

    def is_complete(self):
        return (
            self.state["mood"] is not None
            and self.state["energy"] is not None
            and self.state["stress"] is not None
            and len(self.state["goals"]) > 0
        )

    def next_question(self):
        if self.state["mood"] is None:
            return "How are you feeling today overall?"
        if self.state["energy"] is None:
            return "How would you describe your energy level today? Low, medium, or high?"
        if self.state["stress"] is None:
            return "And how are your stress levels right now?"
        if len(self.state["goals"]) == 0:
            return "What are 1–3 simple goals you want to aim for today?"
        return None


# -------------------------------------------------
#   DAY 3 WELLNESS AGENT
# -------------------------------------------------

class WellnessAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are a supportive, realistic wellness companion.
You do NOT diagnose, treat, or give medical advice.

Your tasks:
1. Ask one question at a time: mood → energy → stress → goals.
2. Use the tools to record data.
3. When all fields are filled, call save_checkin().
4. Offer brief, simple, grounded suggestions only (e.g., take a short break, split tasks).
5. Reference past data from the JSON log when useful.

Never output JSON directly. Always use the provided tools.
"""
        )
        self.state = WellnessState()

    # ------------------------- TOOLS -------------------------

    @function_tool
    async def update_field(self, ctx: RunContext, field: str, value: str) -> str:
        """Update a field in today's check-in session."""
        if field == "goals":
            self.state.state["goals"].append(value)
        else:
            self.state.state[field] = value
        return "updated"

    @function_tool
    async def save_checkin(self, ctx: RunContext) -> str:
        """Save a completed daily check-in to wellness_log.json"""

        # Fill timestamp
        self.state.state["timestamp"] = datetime.now().isoformat()

        # Load old log
        data = []
        if LOG_PATH.exists():
            try:
                with open(LOG_PATH, "r") as f:
                    data = json.load(f)
            except:
                data = []

        # Append
        data.append(self.state.state)

        # Write back
        with open(LOG_PATH, "w") as f:
            json.dump(data, f, indent=2)

        return "saved"

    @function_tool
    async def load_history(self, ctx: RunContext) -> str:
        """Load previous check-ins and return a short summary string"""
        if not LOG_PATH.exists():
            return "no_history"

        try:
            with open(LOG_PATH, "r") as f:
                data = json.load(f)
        except:
            return "no_history"

        if not data:
            return "no_history"

        last = data[-1]
        summary = f"Last time you were feeling {last.get('mood', 'unknown')} with {last.get('energy', 'unknown')} energy."
        return summary

    # ------------------------- MESSAGE HANDLER -------------------------

    async def on_user_message(self, msg, ctx):
        text = msg.text.strip().lower()

        # Simple parsing
        mood_words = ["happy", "sad", "okay", "fine", "good", "bad", "neutral"]
        energy_words = ["low", "medium", "high"]
        stress_words = ["low", "medium", "high"]

        # MOOD
        for m in mood_words:
            if m in text and self.state.state["mood"] is None:
                await ctx.tool_call(self.update_field, field="mood", value=m)
                break

        # ENERGY
        for e in energy_words:
            if e in text and self.state.state["energy"] is None:
                await ctx.tool_call(self.update_field, field="energy", value=e)
                break

        # STRESS
        for s in stress_words:
            if s in text and self.state.state["stress"] is None:
                await ctx.tool_call(self.update_field, field="stress", value=s)
                break

        # GOALS – detect sentences starting with "i want", "i will", etc.
        if any(x in text for x in ["i want", "i will", "i plan", "my goal"]):
            cleaned = text.replace("i want to", "").replace("i want", "")
            cleaned = cleaned.replace("i will", "").replace("my goal is", "").strip()
            if cleaned:
                await ctx.tool_call(self.update_field, field="goals", value=cleaned)

        # Ask next question
        if not self.state.is_complete():
            await ctx.llm_response(self.state.next_question())
            return

        # All complete → save JSON
        await ctx.tool_call(self.save_checkin)

        # Create summary safely
        st = self.state.state
        goals_text = ", ".join(st["goals"])

        summary = (
            f"Thanks! Today you're feeling {st['mood']}, "
            f"your energy is {st['energy']}, and your stress is {st['stress']}. "
            f"Your goals are: {goals_text}. "
            "A small tip: break these into small steps and take brief pauses."
        )

        await ctx.llm_response(summary)

        # Reset for next session
        self.state.reset()


# -------------------------------------------------
#  PREWARM + ENTRYPOINT
# -------------------------------------------------

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context = {"room": ctx.room.name}

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
        usage_collector.collect(ev.metrics)
        metrics.log_metrics(ev.metrics)

    async def log_usage():
        logger.info(f"Usage summary: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=WellnessAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

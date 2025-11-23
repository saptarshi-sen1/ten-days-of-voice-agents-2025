import logging
import json
import os
from datetime import datetime
from pathlib import Path

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

# ============================================================
#   Persistence Folder + Save Function
# ============================================================

LOG_DIR = Path("health_and_wellness")
LOG_DIR.mkdir(exist_ok=True)

def save_checkin_to_file(checkin_data: dict) -> str:
    """
    Saves a wellness check-in to a timestamped JSON file:
       health_and_wellness/checkin_2025-11-23_18-10-04.json
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"checkin_{timestamp}.json"
    filepath = LOG_DIR / filename

    with open(filepath, "w") as f:
        json.dump(checkin_data, f, indent=2)

    return str(filepath)


# ============================================================
#   Wellness Agent State
# ============================================================

class WellnessState:
    def __init__(self):
        self.state = {
            "mood": None,
            "energy": None,
            "stress": None,
            "goals": [],
            "summary": None
        }

    def is_complete(self):
        return (
            self.state["mood"] is not None and
            self.state["energy"] is not None and
            self.state["goals"]
        )

    def next_question(self):
        if self.state["mood"] is None:
            return "How are you feeling today? What's your mood like?"
        if self.state["energy"] is None:
            return "How would you describe your energy right now? High, medium, or low?"
        if self.state["stress"] is None:
            return "Is anything stressing you out today, or are you feeling okay?"
        if not self.state["goals"]:
            return "What are 1–3 simple goals you'd like to focus on today?"
        return None


# ============================================================
#   MAIN WELLNESS AGENT
# ============================================================

class WellnessAgent(Agent):

    def __init__(self):
        super().__init__(
            instructions="""
You are a calm, supportive, but realistic daily health & wellness companion.

Your job each day:
1. Ask about mood → energy → stress → goals (1–3).
2. After each response, call update_checkin(field, value).
3. When all fields are filled, call save_checkin().
4. After saving, give a short recap and encouragement.

Important rules:
- Avoid all medical or diagnostic statements.
- Keep suggestions simple, actionable, grounded.
- Never print JSON — always use the tools for updating or saving.
"""
        )

        self.state = WellnessState()


    # ---------------------------------------------------------
    #   TOOL: Update fields during conversation
    # ---------------------------------------------------------

    @function_tool
    async def update_checkin(self, ctx: RunContext, field: str, value: str) -> str:
        """
        Update one check-in field.
        Fields: mood, energy, stress, goals
        For 'goals', value is appended.
        """
        if field == "goals":
            self.state.state["goals"].append(value)
        else:
            self.state.state[field] = value
        return "updated"


    # ---------------------------------------------------------
    #   TOOL: Save final check-in to timestamped JSON file
    # ---------------------------------------------------------

    @function_tool
    async def save_checkin(self, ctx: RunContext) -> str:
        """
        Save the wellness check-in to a timestamped JSON file.
        """
        mood = self.state.state["mood"] or "unknown"
        goals_list = self.state.state["goals"]
        goals_text = ", ".join(goals_list) if goals_list else "no goals"

        # Generate an auto-summary
        self.state.state["summary"] = (
            f"Mood: {mood}, Goals: {goals_text}"
        )

        filepath = save_checkin_to_file(self.state.state)
        return filepath


    # ---------------------------------------------------------
    #   USER MESSAGE HANDLER
    # ---------------------------------------------------------

    async def on_user_message(self, msg, ctx):
        text = msg.text.lower()

        # Simple keyword detection
        mood_words = ["happy", "sad", "okay", "fine", "good", "bad", "stressed"]
        energy_words = ["low", "medium", "high", "tired", "energetic"]
        stress_words = ["yes", "a bit", "no", "not really", "kind of"]

        # Mood
        for w in mood_words:
            if w in text and self.state.state["mood"] is None:
                await ctx.tool_call(self.update_checkin, field="mood", value=w)
                break

        # Energy
        for w in energy_words:
            if w in text and self.state.state["energy"] is None:
                if w == "tired":
                    w = "low"
                await ctx.tool_call(self.update_checkin, field="energy", value=w)
                break

        # Stress
        if self.state.state["stress"] is None:
            if any(x in text for x in ["yes", "yeah", "yep", "stressed"]):
                await ctx.tool_call(self.update_checkin, field="stress", value="stressed")
            elif any(x in text for x in ["no", "not really", "fine", "okay"]):
                await ctx.tool_call(self.update_checkin, field="stress", value="not stressed")

        # Goals
        if "goal" in text or "today i want" in text or "i want to" in text:
            cleaned = text.replace("today i want to", "").replace("i want to", "").strip()
            await ctx.tool_call(self.update_checkin, field="goals", value=cleaned)

        # Ask next question if incomplete
        if not self.state.is_complete():
            await ctx.llm_response(self.state.next_question())
            return

        # SAVE
        filepath = await ctx.tool_call(self.save_checkin)

        # FINAL SUMMARY
        s = self.state.state
        recap = (
            f"Got it. You're feeling {s['mood']} with {s['energy']} energy. "
            f"Today's goals are: {', '.join(s['goals'])}. "
            "Thanks for checking in — I hope your day goes smoothly."
        )

        await ctx.llm_response(recap)


# ============================================================
#   PREWARM + ENTRYPOINT
# ============================================================

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
            text_pacing=True
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
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm
        )
    )

import logging
from dotenv import load_dotenv
from datetime import datetime
import json

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
    tokenize
)

from livekit.plugins import (
    murf,
    deepgram,
    google,
    silero,
    noise_cancellation
)

from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")


# -----------------------------------
# GAME MASTER AGENT (Day 8)
# -----------------------------------
class GameMasterAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are a Dungeons & Dragons style GAME MASTER running a fully voice-driven adventure.

Universe: A magical fantasy world filled with dragons, ancient ruins, elves, and mysterious portals.
Tone: Dramatic, immersive, adventurous.
Role: Narrate scenes, create tension, and ALWAYS end each message with: 'What do you do?'

Rules:
- Remember the hero’s decisions.
- Keep the story consistent using conversation history.
- Introduce NPCs, choices, and consequences.
- Keep responses short enough for TTS.
- Never take actions for the player.
- Always wait for player's next spoken action.
"""
        )

        # Internal world-state (optional)
        self.story_state = {
            "visited_locations": [],
            "inventory": [],
            "danger_level": 1,
            "started": False,
        }

    async def on_user_message(self, msg, ctx: RunContext):
        user_text = msg.text.strip().lower()
        print("USER:", user_text)

        # FIRST message → start the adventure
        if not self.story_state["started"]:
            self.story_state["started"] = True
            intro = """
The cold wind whips across your face as you awaken at the edge of an ancient forest.
A faint blue glow pulses between the trees… almost calling your name.
In the distance, a dragon’s roar echoes through the mountains.

You feel a strange key in your pocket — warm, humming with magic.

What do you do?
"""
            await ctx.llm_response(intro)
            return

        # Otherwise, continue the interactive adventure
        prompt = f"""
The player said: "{user_text}"

Continue the story in 3–6 sentences.
Respond dramatically and advance the plot.

Always end with: "What do you do?"
"""
        await ctx.llm_response(prompt)


# -----------------------------------
# PREWARM (load VAD for speed)
# -----------------------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# -----------------------------------
# ENTRYPOINT (LiveKit Worker)
# -----------------------------------
async def entrypoint(ctx: JobContext):
    ctx.add_shutdown_callback(lambda: logger.info("Game Master shutting down."))

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Narration",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await session.start(
        agent=GameMasterAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


# -----------------------------------
# MAIN
# -----------------------------------
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

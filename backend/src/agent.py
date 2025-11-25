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
    RoomInputOptions,
    WorkerOptions,
    cli,
    tokenize,
    metrics,
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# ----------------------------------------------------
# ENV + GLOBALS
# ----------------------------------------------------

load_dotenv(".env.local")
logger = logging.getLogger("agent")

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
CONTENT_DIR = BASE_DIR / "shared-data"
CONTENT_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------
# AUTO-CREATE CONTENT FILE IF MISSING
# ----------------------------------------------------

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

DEFAULT_CONTENT = [
    {
        "id": "variables",
        "title": "Variables",
        "summary": "Variables store values that you can reuse later in your program.",
        "sample_question": "What is a variable and why is it useful?"
    },
    {
        "id": "loops",
        "title": "Loops",
        "summary": "Loops allow repeating actions multiple times without rewriting code.",
        "sample_question": "Explain the difference between a for loop and a while loop."
    }
]

# Look for an existing file
existing_files = list(CONTENT_DIR.glob("day4_tutor_content*.json"))

if existing_files:
    CONTENT_PATH = existing_files[0]
else:
    CONTENT_PATH = CONTENT_DIR / f"day4_tutor_content_{timestamp}.json"
    with open(CONTENT_PATH, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONTENT, f, indent=2)

# Load content
with open(CONTENT_PATH, "r", encoding="utf-8") as f:
    COURSE_CONTENT = json.load(f)

# ----------------------------------------------------
# TUTOR AGENT
# ----------------------------------------------------

class TutorAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are an AI active recall tutor with 3 modes:
1. learn – explain a concept using its summary.
2. quiz – ask the user questions using sample_question.
3. teach_back – ask the user to explain the concept back and give basic feedback.

Rules:
• NEVER output JSON.
• Detect mode changes from user messages.
• ALWAYS speak with the correct Murf voice depending on mode.
• Keep responses concise and conversational.
""")

        self.mode = None  # "learn", "quiz", "teach_back"
        self.current_concept = None  # object from COURSE_CONTENT

    # ------------------------------------------------
    # HELPER: choose concept
    # ------------------------------------------------
    def find_concept(self, text: str):
        text = text.lower()
        for c in COURSE_CONTENT:
            if c["id"] in text or c["title"].lower() in text:
                return c
        return None

    # ------------------------------------------------
    # TOOL: Switch learning mode
    # ------------------------------------------------
    @function_tool
    async def switch_mode(self, ctx: RunContext, mode: str) -> str:
        self.mode = mode
        return f"Mode changed to {mode}"

    # ------------------------------------------------
    # TOOL: Select concept
    # ------------------------------------------------
    @function_tool
    async def pick_concept(self, ctx: RunContext, concept_id: str) -> str:
        for c in COURSE_CONTENT:
            if c["id"] == concept_id:
                self.current_concept = c
                return "concept_selected"
        return "not_found"

    # ------------------------------------------------
    # RESPOND TO USER
    # ------------------------------------------------
    async def on_user_message(self, msg, ctx):
        text = msg.text.lower()

        # Detect mode switches
        if "learn" in text:
            await ctx.tool_call(self.switch_mode, mode="learn")
        elif "quiz" in text:
            await ctx.tool_call(self.switch_mode, mode="quiz")
        elif "teach" in text or "teach back" in text:
            await ctx.tool_call(self.switch_mode, mode="teach_back")

        # Detect concept selection
        concept = self.find_concept(text)
        if concept:
            await ctx.tool_call(self.pick_concept, concept_id=concept["id"])

        # If mode or concept missing → ask user
        if not self.mode:
            await ctx.llm_response("Welcome! Would you like to Learn, Quiz, or Teach Back?")
            return

        if not self.current_concept:
            await ctx.llm_response("Great! Which topic? You can choose variables or loops.")
            return

        # -------------------- MODE: LEARN --------------------
        if self.mode == "learn":
            ctx.session.tts.voice = "en-US-matthew"  # Matthew
            summary = self.current_concept["summary"]
            await ctx.llm_response(f"Here's a quick explanation: {summary}")
            return

        # -------------------- MODE: QUIZ --------------------
        elif self.mode == "quiz":
            ctx.session.tts.voice = "en-US-alicia"  # Alicia
            question = self.current_concept["sample_question"]
            await ctx.llm_response(f"Alright! Here's your question: {question}")
            return

        # -------------------- MODE: TEACH BACK --------------------
        elif self.mode == "teach_back":
            ctx.session.tts.voice = "en-US-ken"  # Ken
            question = self.current_concept["sample_question"]
            await ctx.llm_response(
                f"Great! Teach this back to me: {question}. I'll tell you how clearly you explained it."
            )
            return


# ----------------------------------------------------
# SESSION CONFIG + ENTRYPOINT
# ----------------------------------------------------

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
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2)
        ),
        vad=ctx.proc.userdata["vad"],
        turn_detection=MultilingualModel(),
        preemptive_generation=True
    )

    await session.start(
        agent=TutorAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        )
    )

    await ctx.connect()



if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm)
    )

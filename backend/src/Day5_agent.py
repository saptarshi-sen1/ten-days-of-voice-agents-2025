# agent.py — SDR Voice Agent (Day 5)
# Place at backend/src/agent.py

import json
import os
from pathlib import Path
from datetime import datetime
import logging

from dotenv import load_dotenv
load_dotenv(".env.local")

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
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)

# -----------------------
# Paths & defaults
# -----------------------
SRC_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SRC_DIR.parent
CONTACTS_DIR = BACKEND_DIR / "contacts"
SHARED_DIR = BACKEND_DIR / "shared-data"
COMPANY_FILE = SHARED_DIR / "company_sdr.json"

CONTACTS_DIR.mkdir(parents=True, exist_ok=True)
SHARED_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_COMPANY = {
    "company": "SmartPay (example)",
    "description": "SmartPay offers a unified payments + invoicing platform for Indian SMBs.",
    "faqs": [
        {"q": "what does your product do", "a": "We provide payments, invoicing and reconciliation in one dashboard."},
        {"q": "do you have a free tier", "a": "Yes, up to 100 transactions per month."},
        {"q": "who is this for", "a": "Small and medium businesses, retail stores, and online sellers."},
        {"q": "pricing", "a": "Free, Standard (monthly fee + per-transaction), and Enterprise plans."}
    ],
    "pricing": [
        {"plan": "Free", "details": "Up to 100 transactions / month"},
        {"plan": "Standard", "details": "₹499/month + usage"},
        {"plan": "Enterprise", "details": "Custom pricing"}
    ]
}

if not COMPANY_FILE.exists():
    with open(COMPANY_FILE, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_COMPANY, f, indent=2, ensure_ascii=False)

with open(COMPANY_FILE, "r", encoding="utf-8") as f:
    COMPANY = json.load(f)

# -----------------------
# FAQ helper
# -----------------------
def find_faq_answer(query: str):
    q = query.lower()
    for entry in COMPANY.get("faqs", []):
        if entry.get("q") and entry["q"] in q:
            return entry.get("a")
    for entry in COMPANY.get("faqs", []):
        combined = (entry.get("q", "") + " " + entry.get("a", "")).lower()
        if any(w for w in q.split() if w in combined):
            return entry.get("a")
    return None

# -----------------------
# Save lead tool
# -----------------------
@function_tool
async def save_lead_tool(name: str,
                         company_name: str,
                         email: str,
                         role: str,
                         use_case: str,
                         team_size: str,
                         timeline: str) -> str:

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fname = f"lead_{timestamp}.json"
    path = CONTACTS_DIR / fname

    lead = {
        "collected_at": datetime.now().isoformat(),
        "name": name,
        "company": company_name,
        "email": email,
        "role": role,
        "use_case": use_case,
        "team_size": team_size,
        "timeline": timeline,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(lead, f, indent=2, ensure_ascii=False)

    return str(path)

# -----------------------
# SDR Agent
# -----------------------
class SDRAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=f"""
You are an SDR for {COMPANY.get('company')}.
Greet politely, ask what they need, collect name, company, email, role,
use_case, team_size, timeline (in this order).
Answer only using FAQ provided. Do not hallucinate.
When user finishes, summarize & save using save_lead_tool().
"""
        )
        self.lead = {
            "name": "",
            "company": "",
            "email": "",
            "role": "",
            "use_case": "",
            "team_size": "",
            "timeline": "",
        }
        self.collect_order = ["name", "company", "email", "role", "use_case", "team_size", "timeline"]
        self.next_index = 0
        self.ended = False

    def _extract_email(self, text):
        if "@" in text: 
            for tok in text.split():
                if "@" in tok and "." in tok:
                    return tok.strip(".,")
        return None

    def _extract_name(self, text):
        low = text.lower()
        if "my name is" in low:
            return text.split("my name is")[-1].strip().split()[0].title()
        if low.startswith(("i am ", "i'm ", "im ")):
            return " ".join(text.split()[1:3]).title()
        return None

    def _is_end(self, text):
        low = text.lower()
        return any(p in low for p in ["that's all", "i'm done", "thank you", "bye"])

    def _current_field(self):
        return self.collect_order[self.next_index] if self.next_index < len(self.collect_order) else None

    def _advance(self):
        self.next_index += 1

    async def on_user_message(self, msg, ctx: RunContext):
        text = (msg.text or "").strip()
        lower = text.lower()

        # End
        if self._is_end(text):
            filepath = await ctx.tool_call(save_lead_tool, **{
                "name": self.lead["name"],
                "company_name": self.lead["company"],
                "email": self.lead["email"],
                "role": self.lead["role"],
                "use_case": self.lead["use_case"],
                "team_size": self.lead["team_size"],
                "timeline": self.lead["timeline"],
            })

            summary = "; ".join(f"{k}: {v}" for k, v in self.lead.items() if v)
            await ctx.llm_response(f"Thanks — saved your details. Summary: {summary}. File: {filepath}. Bye!")
            self.ended = True
            return

        # FAQ
        if any(w in lower for w in ["price", "pricing", "cost", "what does", "free tier"]):
            ans = find_faq_answer(lower)
            await ctx.llm_response(ans or "I don't have that detail, but I can connect you with sales.")
            return

        # Lead collection
        field = self._current_field()

        if field == "name":
            name = self._extract_name(text)
            if name or (1 <= len(text.split()) <= 3):
                self.lead["name"] = name or text.title()
                self._advance()
            else:
                await ctx.llm_response("May I know your name?")
                return

        elif field == "company":
            self.lead["company"] = text.title()
            self._advance()

        elif field == "email":
            email = self._extract_email(text)
            if not email:
                await ctx.llm_response("Could you share your email?")
                return
            self.lead["email"] = email
            self._advance()

        elif field == "role":
            self.lead["role"] = text.title()
            self._advance()

        elif field == "use_case":
            self.lead["use_case"] = text
            self._advance()

        elif field == "team_size":
            self.lead["team_size"] = text
            self._advance()

        elif field == "timeline":
            self.lead["timeline"] = text
            self._advance()

        # Ask next field
        next_field = self._current_field()
        prompts = {
            "name": "May I know your name?",
            "company": "Which company do you represent?",
            "email": "Your email address?",
            "role": "What's your role?",
            "use_case": "What do you want to use our product for?",
            "team_size": "How big is your team?",
            "timeline": "What's your timeline (now / soon / later)?",
        }
        if next_field:
            await ctx.llm_response(prompts[next_field])
        else:
            await ctx.llm_response("Thanks! Say 'that's all' to finish.")

# -----------------------
# Prewarm + entrypoint
# -----------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
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

    await session.start(
        agent=SDRAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

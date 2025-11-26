# agent.py — SDR Voice Agent (Day X)
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
# Paths & default content
# -----------------------
SRC_DIR = Path(__file__).resolve().parent        # backend/src
BACKEND_DIR = SRC_DIR.parent                     # backend/
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
        {"q": "do you have a free tier", "a": "Yes — a limited free tier up to 100 transactions per month."},
        {"q": "who is this for", "a": "Small and medium businesses, retail stores, and online sellers in India."},
        {"q": "pricing", "a": "We offer Free, Standard (monthly fee + per-transaction), and Enterprise plans."}
    ],
    "pricing": [
        {"plan": "Free", "details": "Up to 100 transactions / month, basic dashboard"},
        {"plan": "Standard", "details": "₹499/month + 1% per transaction; advanced reporting"},
        {"plan": "Enterprise", "details": "Custom pricing, priority support"}
    ]
}

# create company file if missing
if not COMPANY_FILE.exists():
    with open(COMPANY_FILE, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_COMPANY, f, indent=2, ensure_ascii=False)

with open(COMPANY_FILE, "r", encoding="utf-8") as f:
    COMPANY = json.load(f)

# -----------------------
# FAQ helper (simple)
# -----------------------
def find_faq_answer(query: str):
    q = query.lower()
    # exact substring match in question
    for entry in COMPANY.get("faqs", []):
        if entry.get("q") and entry["q"] in q:
            return entry.get("a")
    # keyword fallback (any word overlap)
    for entry in COMPANY.get("faqs", []):
        combined = (entry.get("q", "") + " " + entry.get("a", "")).lower()
        if any(word for word in q.split() if word and word in combined):
            return entry.get("a")
    return None

# -----------------------
# Tool: save lead (NO ctx)
# -----------------------
@function_tool
async def save_lead_tool(name: str,
                         company_name: str,
                         email: str,
                         role: str,
                         use_case: str,
                         team_size: str,
                         timeline: str) -> str:
    """
    Saves lead as timestamped JSON file inside backend/contacts/.
    Returns the saved filepath string.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fname = f"lead_{timestamp}.json"
    path = CONTACTS_DIR / fname

    lead = {
        "collected_at": datetime.now().isoformat(),
        "name": name or "",
        "company": company_name or "",
        "email": email or "",
        "role": role or "",
        "use_case": use_case or "",
        "team_size": team_size or "",
        "timeline": timeline or "",
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
You are a friendly SDR for {COMPANY.get('company')}.
Greet visitors warmly, ask what they're working on, focus on understanding needs,
answer questions only from the provided FAQ/pricing content (do not invent details),
and collect lead fields naturally: name, company, email, role, use_case, team_size, timeline.
When the user indicates they are done (e.g. "that's all", "thanks", "I'm done"), summarize the lead and save it.
Keep replies short, polite, and conversational.
"""
        )
        # lead state
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

    # simple helpers to extract field heuristically
    def _extract_email(self, text: str):
        text = text.strip()
        if "@" in text and "." in text:
            # take the token containing @
            for tok in text.split():
                if "@" in tok and "." in tok:
                    return tok.strip(".,;")
        return None

    def _extract_name_from_phrase(self, text: str):
        # handle "my name is ..." or "i'm ...", short heuristic
        lower = text.lower()
        if "my name is" in lower:
            return text.split("my name is")[-1].strip().split()[0].capitalize()
        if lower.startswith("i am ") or lower.startswith("i'm ") or lower.startswith("im "):
            parts = text.split()
            # return first two tokens maybe
            candidate = " ".join(parts[1:3]).strip()
            return candidate.title()
        return None

    def _is_end(self, text: str):
        lower = text.lower()
        return any(phrase in lower for phrase in ["that's all", "i'm done", "i am done", "thanks", "thank you", "bye"])

    def _current_field(self):
        if self.next_index < len(self.collect_order):
            return self.collect_order[self.next_index]
        return None

    def _advance_field(self):
        self.next_index = min(self.next_index + 1, len(self.collect_order))

    # main message handler
    async def on_user_message(self, msg, ctx: RunContext):
        text = (msg.text or "").strip()
        lower = text.lower()

        # End of call detection
        if self._is_end(text):
            # Save and respond summary
            filepath = await ctx.tool_call(save_lead_tool,
                                           name=self.lead["name"],
                                           company_name=self.lead["company"],
                                           email=self.lead["email"],
                                           role=self.lead["role"],
                                           use_case=self.lead["use_case"],
                                           team_size=self.lead["team_size"],
                                           timeline=self.lead["timeline"])
            # build summary
            parts = []
            if self.lead["name"]:
                parts.append(f"Name: {self.lead['name']}")
            if self.lead["company"]:
                parts.append(f"Company: {self.lead['company']}")
            if self.lead["email"]:
                parts.append(f"Email: {self.lead['email']}")
            if self.lead["role"]:
                parts.append(f"Role: {self.lead['role']}")
            if self.lead["use_case"]:
                parts.append(f"Use case: {self.lead['use_case']}")
            if self.lead["team_size"]:
                parts.append(f"Team size: {self.lead['team_size']}")
            if self.lead["timeline"]:
                parts.append(f"Timeline: {self.lead['timeline']}")
            summary = "; ".join(parts) if parts else "No lead details collected."
            await ctx.llm_response(f"Thanks — quick summary: {summary} I saved this to {filepath}. We'll reach out soon. Bye!")
            self.ended = True
            return

        # If user asks pricing/product/company -> use FAQ (no hallucination)
        if any(k in lower for k in ["price", "pricing", "cost", "free tier", "what does", "what is", "who is this for", "who is this"]):
            ans = find_faq_answer(lower)
            if ans:
                await ctx.llm_response(ans)
                return
            else:
                await ctx.llm_response("I don't have that detail in the FAQ — would you like me to connect you with someone from sales?")
                return

        # Attempt to fill fields from the current message
        cur = self._current_field()

        # Try to extract name
        if cur == "name":
            name = self._extract_name_from_phrase(text)
            if name:
                self.lead["name"] = name
                self._advance_field()
            else:
                # If message is short and likely a name, accept it
                if 1 <= len(text.split()) <= 3 and len(text) < 40:
                    self.lead["name"] = text.title()
                    self._advance_field()
                else:
                    await ctx.llm_response("Hi — can I get your name for the order?")
                    return

        # company
        if self._current_field() == "company":
            # accept short response
            if len(text.split()) <= 6:
                self.lead["company"] = text.title()
                self._advance_field()
            else:
                await ctx.llm_response("Which company are you with?")
                return

        # email
        if self._current_field() == "email":
            email = self._extract_email(text)
            if email:
                self.lead["email"] = email
                self._advance_field()
            else:
                await ctx.llm_response("What's the best email to reach you at?")
                return

        # role
        if self._current_field() == "role":
            if len(text.split()) <= 6:
                self.lead["role"] = text.title()
                self._advance_field()
            else:
                await ctx.llm_response("What's your role there?")
                return

        # use_case
        if self._current_field() == "use_case":
            # allow long text here
            self.lead["use_case"] = text
            self._advance_field()

        # team_size
        if self._current_field() == "team_size":
            # try to pick number/token
            tokens = text.split()
            picked = None
            for tok in tokens:
                if tok.isdigit():
                    picked = tok
                    break
            if not picked:
                # map words
                if any(w in lower for w in ["small", "solo", "one", "two", "three", "few"]):
                    picked = "1-5"
            if picked:
                self.lead["team_size"] = picked
                self._advance_field()
            else:
                await ctx.llm_response("Roughly how large is your team? (e.g., 1-5, 10-50)")
                return

        # timeline
        if self._current_field() == "timeline":
            if any(w in lower for w in ["now", "immediately", "soon", "later", "next", "month"]):
                if "now" in lower or "immediately" in lower:
                    self.lead["timeline"] = "now"
                elif "soon" in lower or "next" in lower or "week" in lower:
                    self.lead["timeline"] = "soon"
                else:
                    self.lead["timeline"] = "later"
                self._advance_field()
            else:
                # accept short answers
                if len(text.split()) <= 6:
                    self.lead["timeline"] = text
                    self._advance_field()
                else:
                    await ctx.llm_response("What's your timeline to start? (now / soon / later)")
                    return

        # After processing, if there are more fields to ask, prompt next
        next_field = self._current_field()
        if next_field:
            prompts = {
                "name": "May I have your full name?",
                "company": "Which company are you with?",
                "email": "What's the best email to reach you at?",
                "role": "What's your role there?",
                "use_case": "What would you like to use our product for?",
                "team_size": "How big is your team (approx.)?",
                "timeline": "What's your timeline to start? (now / soon / later)"
            }
            await ctx.llm_response(prompts.get(next_field, "Could you provide that detail?"))
            return

        # If we've got everything and user hasn't said 'done' yet, ask to confirm or say next steps
        if not self.ended and all(self.lead.get(k) for k in ["name", "email", "role", "use_case"]):
            await ctx.llm_response("Thanks — I've got your details. If you'd like, say 'that's all' to finish and I'll save this. Would you like to add anything else?")
            return

        # Fallback
        await ctx.llm_response("Sorry, I didn't catch that. Could you rephrase? You can also ask product or pricing questions.")

# -----------------------
# Prewarm + Entry point
# -----------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

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
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info(f"Usage: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

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

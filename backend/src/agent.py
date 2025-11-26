# backend/src/agent.py
import json
import os
from pathlib import Path
from datetime import datetime
import logging
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
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")
logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)

# -----------------------
# Paths & helpers
# -----------------------
ROOT = Path(__file__).resolve().parent.parent  # backend/
CASES_DIR = ROOT / "fraud_cases"
CASES_DIR.mkdir(exist_ok=True)

def case_path_for(name_token: str) -> Path:
    """Return path for a case file corresponding to `name_token` (lowercased)."""
    return CASES_DIR / f"{name_token.lower()}.json"

def load_case_for(name_token: str):
    p = case_path_for(name_token)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_case_for(name_token: str, data: dict):
    p = case_path_for(name_token)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved case: {p}")

def normalize_text(s: str) -> str:
    return "".join(ch for ch in (s or "").strip().lower() if ch.isalnum() or ch.isspace()).strip()

# -----------------------
# Example schema for fraud case (for reference)
# {
#   "userName": "John",
#   "securityIdentifier": "12345",
#   "verification_question": "What is your favorite color?",
#   "verification_answer": "blue",
#   "cardEnding": "4242",
#   "transactionAmount": "₹4,200",
#   "transactionName": "ABC Electronics",
#   "transactionLocation": "Bangalore",
#   "transactionTime": "2025-10-15 14:22",
#   "transactionCategory": "e-commerce",
#   "transactionSource": "flipkart.com",
#   "status": "pending_review",
#   "note": ""
# }
# -----------------------

class FraudAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a calm, professional fraud-prevention assistant for a bank. "
                "When a session starts, greet the user, ask for their registered name, "
                "verify using the exact stored verification question, and then read the suspicious "
                "transaction details. Ask yes/no if they made it; update the case JSON with "
                "confirmed_safe / confirmed_fraud / verification_failed and add a short note. "
                "Keep answers short and polite."
            ),
        )
        # conversation state per job/instance:
        self.case = None
        self.case_name_token = None
        self.stage = "await_name"  # await_name -> await_verification -> await_confirm -> done

    async def on_user_message(self, msg, ctx):
        """
        msg.text is the user's transcribed text for spoken input / typed input.
        ctx.llm_response(text) sends the agent's reply back (and will be TTS-ed).
        """
        text = (msg.text or "").strip()
        lower = text.lower()

        # Stage: ask for name (if we haven't)
        if self.stage == "await_name":
            # If user provided a name in the first utterance, try to load case.
            # We'll take the first token as the key (common in prior examples).
            if not text:
                await ctx.llm_response("Hello — may I have the registered account name, please?")
                return

            name_token = text.split()[0]
            case = load_case_for(name_token)
            if case is None:
                # fallback: try exact match for whole phrase as filename
                case = load_case_for(text.replace(" ", "_"))
                if case is None:
                    # ask to re-provide name
                    await ctx.llm_response(
                        "I couldn't find a record under that name. Please tell me the registered name on your account."
                    )
                    return

            # found case
            self.case = case
            self.case_name_token = name_token
            self.stage = "await_verification"

            # ask verification question from case
            vq = case.get("verification_question") or case.get("security_question") or "Please provide a verification answer."
            await ctx.llm_response(f"Thank you — to verify your identity: {vq}")
            return

        # Stage: verification
        if self.stage == "await_verification":
            if not self.case:
                self.stage = "await_name"
                await ctx.llm_response("Sorry, I lost the case. Please provide your registered name again.")
                return

            expected_raw = str(self.case.get("verification_answer") or self.case.get("security_answer") or "").strip()
            if not expected_raw:
                # no verification data, fail safe
                self.case["status"] = "verification_failed"
                self.case["note"] = "No verification question/answer stored."
                save_case_for(self.case_name_token, self.case)
                await ctx.llm_response("I cannot verify your identity at this time. Please contact the bank directly.")
                self.stage = "done"
                return

            given_norm = normalize_text(text)
            expected_norm = normalize_text(expected_raw)

            if given_norm == expected_norm:
                # verified
                self.stage = "await_confirm"
                # read suspicious transaction details
                txn_name = self.case.get("transactionName") or self.case.get("merchant") or "unknown merchant"
                txn_amt = self.case.get("transactionAmount") or self.case.get("amount") or "unknown amount"
                txn_time = self.case.get("transactionTime") or self.case.get("date") or "unknown time"
                txn_loc = self.case.get("transactionLocation") or self.case.get("transactionLocation") or ""
                card_end = self.case.get("cardEnding") or self.case.get("cardEnding")
                reply = (
                    f"Thank you — verification passed. We detected a suspicious transaction: "
                    f"{txn_name} for {txn_amt}"
                )
                if card_end:
                    reply += f" on card ending {card_end}"
                if txn_time:
                    reply += f" at {txn_time}"
                if txn_loc:
                    reply += f" in {txn_loc}"
                reply += ". Did you make this transaction? Please answer yes or no."

                await ctx.llm_response(reply)
                return
            else:
                # verification failed (strict)
                self.case["status"] = "verification_failed"
                self.case["note"] = f"Verification failed. Provided: {text}"
                save_case_for(self.case_name_token, self.case)
                await ctx.llm_response(
                    "I'm sorry — that answer does not match our records. For your security, I cannot continue. Please contact the bank."
                )
                self.stage = "done"
                return

        # Stage: confirmation (yes/no)
        if self.stage == "await_confirm":
            if not self.case:
                self.stage = "await_name"
                await ctx.llm_response("I couldn't find the case — please provide your registered name.")
                return

            if any(tok in lower for tok in ["yes", "yep", "yeah", "y"]):
                self.case["status"] = "confirmed_safe"
                self.case["note"] = "Customer confirmed transaction as legitimate."
                save_case_for(self.case_name_token, self.case)
                await ctx.llm_response("Thank you. I have marked this transaction as legitimate and closed the case. Have a nice day.")
                self.stage = "done"
                return

            if any(tok in lower for tok in ["no", "nope", "nah", "n"]):
                self.case["status"] = "confirmed_fraud"
                self.case["note"] = "Customer denied the transaction. Mock card block issued and dispute created."
                # optionally add timestamped actions
                action_note = f"Mock card block issued and dispute created at {datetime.utcnow().isoformat()}Z"
                self.case.setdefault("actions", []).append(action_note)
                save_case_for(self.case_name_token, self.case)
                await ctx.llm_response(
                    "Understood. I have flagged the transaction as fraudulent, issued a mock card block, and created a dispute. Our fraud team will follow up. Stay safe."
                )
                self.stage = "done"
                return

            # unclear answer
            await ctx.llm_response("Please answer clearly with 'yes' or 'no'.")
            return

        # If done or unknown stage
        await ctx.llm_response("Thank you. If you need anything else, contact your bank's support.")

# -----------------------
# Prewarm + entrypoint
# -----------------------

def prewarm(proc: JobProcess):
    # load VAD once
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
            text_pacing=True,
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
        summary = usage_collector.get_summary()
        logger.info(f"Usage summary: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=FraudAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

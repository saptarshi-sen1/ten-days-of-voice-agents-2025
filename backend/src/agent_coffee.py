import logging
import json
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
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# -------------------------------------------------
#   COFFEE ORDER STATE MACHINE IMPLEMENTATION
# -------------------------------------------------

class CoffeeOrderState:
    def __init__(self):
        self.state = {
            "drinkType": "",
            "size": "",
            "milk": "",
            "extras": [],
            "name": ""
        }

    def is_complete(self):
        return (
            self.state["drinkType"]
            and self.state["size"]
            and self.state["milk"]
            and self.state["name"]
        )

    def next_question(self):
        if not self.state["drinkType"]:
            return "What drink would you like today? For example latte, cappuccino, americano, or cold brew."
        if not self.state["size"]:
            return "What size would you like? Small, medium, or large."
        if not self.state["milk"]:
            return "What type of milk should I use? Whole, skim, oat, almond, or soy."
        if not self.state["name"]:
            return "Can I get your name for the order?"
        return None


# -------------------------------------------------
#   AGENT LOGIC — THE BARISTA
# -------------------------------------------------

class BaristaAgent(Agent):

    def __init__(self):
        super().__init__(
            instructions="""
You are a friendly barista at Moonbeam Coffee. 
Your job is to take the customer’s order through natural conversation.
Ask one clarifying question at a time. Keep replies short and natural.
Never show JSON or code to the user.

Your order fields are:
- drinkType
- size
- milk
- extras
- name

If the user mentions something that matches a field, fill it in.
If the order is not complete, ask the next missing question.
When the order is complete, say the order summary and confirm it.
After confirming, say you are preparing the drink.
            """
        )

        # One order per conversation
        self.order = CoffeeOrderState()

    async def on_user_message(self, msg, ctx):
        text = msg.text.lower()

        # DRINK TYPE
        if any(d in text for d in ["latte", "cappuccino", "americano", "mocha", "cold brew", "espresso"]):
            words = text.split()
            for w in words:
                if w in ["latte", "cappuccino", "americano", "mocha", "espresso", "brew"]:
                    self.order.state["drinkType"] = w
                    break

        # SIZE
        if any(s in text for s in ["small", "medium", "large"]):
            for s in ["small", "medium", "large"]:
                if s in text:
                    self.order.state["size"] = s
                    break

        # MILK
        for m in ["whole", "skim", "oat", "almond", "soy"]:
            if m in text:
                self.order.state["milk"] = m
                break

        # EXTRAS
        if "extra" in text or "add" in text:
            for keyword in ["sugar", "ice", "vanilla", "caramel", "cinnamon", "cream"]:
                if keyword in text:
                    self.order.state["extras"].append(keyword)

        # NAME
        if "my name is" in text:
            name = text.split("my name is")[-1].strip()
            if len(name.split()) == 1:
                self.order.state["name"] = name.capitalize()
            else:
                self.order.state["name"] = name.split()[0].capitalize()

        # If order is not complete → ask next question
        if not self.order.is_complete():
            q = self.order.next_question()
            await ctx.llm_response(q)
            return

        # If order completed → save + confirm
        summary = self.order.state
        self.save_order(summary)

        response = (
            f"Perfect {summary['name']}. So that is a {summary['size']} "
            f"{summary['drinkType']} with {summary['milk']} milk"
        )

        if summary["extras"]:
            response += " and extra " + ", ".join(summary["extras"])

        response += ". I am preparing your drink now."

        await ctx.llm_response(response)

    def save_order(self, data):
        orders_path = Path("orders")
        orders_path.mkdir(exist_ok=True)
        file_path = orders_path / "latest_order.json"
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)


# -------------------------------------------------
#  PREWARM + ENTRYPOINT (YOUR ORIGINAL)
# -------------------------------------------------

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
    def _on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info(f"Usage: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=BaristaAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

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

# -------------------------------------------------
#   ORDER STATE MACHINE
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
            return "What drink would you like? For example latte, cappuccino, americano, or mocha?"
        if not self.state["size"]:
            return "What size would you like? Small, medium, or large?"
        if not self.state["milk"]:
            return "What type of milk should I use? Whole, skim, oat, almond, or soy?"
        if not self.state["name"]:
            return "May I know your name?"
        return None


# -------------------------------------------------
#   BARISTA AGENT WITH GEMINI TOOLS
# -------------------------------------------------

class BaristaAgent(Agent):

    def __init__(self):
        super().__init__(
            instructions="""
You are a friendly barista taking a coffee order for Moonbeam Coffee.

Your job:
1. Ask one question at a time (drinkType → size → milk → name).
2. When the user answers, call update_order(field, value).
3. If they mention an add-on such as sugar, ice, vanilla, caramel, cinnamon, or cream, call update_order("extras", that_extra).
4. When all fields are filled, call save_order().
5. After saving, tell the user the order summary and that you're preparing their drink.

Important:
- Never output JSON.
- Always use the tools to modify or save the order.

"""
        )

        self.order = CoffeeOrderState()

    # -------- TOOL: Update order fields --------

    @function_tool
    async def update_order(self, ctx: RunContext, field: str, value: str) -> str:
        """
        Update a single field in the customer's coffee order.
        Valid fields: drinkType, size, milk, extras, name
        """
        if field == "extras":
            self.order.state["extras"].append(value)
        else:
            self.order.state[field] = value
        return "updated"

    # -------- TOOL: Save order to JSON --------

    @function_tool
    async def save_order(self, ctx: RunContext) -> str:
        """
        Save the completed order into a timestamped JSON file.
        Returns the file path.
        """
        os.makedirs("orders", exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = f"orders/order_{timestamp}.json"

        with open(filepath, "w") as f:
            json.dump(self.order.state, f, indent=2)

        return filepath

    # -------- Handle user messages --------

    async def on_user_message(self, msg, ctx):
        text = msg.text.lower()

        # Mapping for simple detection
        drink_keywords = ["latte", "cappuccino", "americano", "mocha", "espresso", "cold brew"]
        size_keywords = ["small", "medium", "large"]
        milk_keywords = ["whole", "skim", "oat", "almond", "soy"]
        extra_keywords = ["sugar", "ice", "vanilla", "caramel", "cinnamon", "cream"]

        # DRINK TYPE
        for d in drink_keywords:
            if d in text:
                await ctx.tool_call(self.update_order, field="drinkType", value=d)
                break

        # SIZE
        for s in size_keywords:
            if s in text:
                await ctx.tool_call(self.update_order, field="size", value=s)
                break

        # MILK
        for m in milk_keywords:
            if m in text:
                await ctx.tool_call(self.update_order, field="milk", value=m)
                break

        # EXTRAS
        for e in extra_keywords:
            if e in text:
                await ctx.tool_call(self.update_order, field="extras", value=e)

        # NAME — detect "my name is X"
        if "my name is" in text:
            name = text.split("my name is")[-1].strip().split()[0]
            name = name.capitalize()
            await ctx.tool_call(self.update_order, field="name", value=name)

        # Ask next question if incomplete
        if not self.order.is_complete():
            await ctx.llm_response(self.order.next_question())
            return

        # SAVE ORDER using tool
        filepath = await ctx.tool_call(self.save_order)

        # Respond with summary
        o = self.order.state
        summary = (
            f"Thanks {o['name']}! So that's a {o['size']} {o['drinkType']} with {o['milk']} milk"
        )

        if o["extras"]:
            summary += f" and extra " + ", ".join(o["extras"])

        summary += ". I'm preparing your drink now!"

        await ctx.llm_response(summary)


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
        agent=BaristaAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

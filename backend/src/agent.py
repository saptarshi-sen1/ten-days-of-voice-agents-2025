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
#   SAMPLE PRODUCT CATALOG
# -------------------------------------------------

CATALOG = [
    {"id": "mug-001", "name": "Stoneware Coffee Mug", "category": "mug", "price": 800, "currency": "INR", "color": "white"},
    {"id": "mug-002", "name": "Black Ceramic Mug", "category": "mug", "price": 650, "currency": "INR", "color": "black"},
    {"id": "hoodie-001", "name": "Blue Hoodie", "category": "hoodie", "price": 1200, "currency": "INR", "color": "blue"},
    {"id": "hoodie-002", "name": "Black Premium Hoodie", "category": "hoodie", "price": 1500, "currency": "INR", "color": "black"},
    {"id": "shirt-001", "name": "Graphic T-Shirt", "category": "tshirt", "price": 900, "currency": "INR", "color": "white"},
]

ORDERS_DIR = "orders"
os.makedirs(ORDERS_DIR, exist_ok=True)

# -------------------------------------------------
#   ORDER STATE MACHINE
# -------------------------------------------------

class OrderState:
    def __init__(self):
        self.state = {
            "product_id": "",
            "quantity": 1,
            "name": ""
        }

    def is_complete(self):
        return (
            self.state["product_id"]
            and self.state["quantity"]
            and self.state["name"]
        )

    def next_question(self):
        if not self.state["product_id"]:
            return "What would you like to buy? You can ask for mugs, hoodies, or t-shirts."
        if not self.state["quantity"]:
            return "How many would you like to order?"
        if not self.state["name"]:
            return "May I know your name?"
        return None


# -------------------------------------------------
#   SHOPPING AGENT
# -------------------------------------------------

class ShoppingAgent(Agent):

    def __init__(self):
        super().__init__(
            instructions="""
You are a friendly shopping assistant. Help the user browse items and place an order.
Use the tools for:
- searching items,
- selecting the product (set_product),
- selecting quantity,
- saving the final order.

Never output JSON. Always call tools to modify order.
Ask only one question at a time.
"""
        )

        self.order = OrderState()

    # -------------------------------------------------
    # TOOL: List or search items
    # -------------------------------------------------

    @function_tool
    async def search_items(self, ctx: RunContext, query: str = "") -> list:
        """
        Returns a list of items matching the search query.
        Matches category, color, or substring in name.
        """
        query = query.lower().strip()
        results = []

        for item in CATALOG:
            if (
                query in item["name"].lower()
                or query in item["category"].lower()
                or query in item["color"].lower()
            ):
                results.append(item)

        return results if results else CATALOG

    # -------------------------------------------------
    # TOOL: Set selected product
    # -------------------------------------------------

    @function_tool
    async def set_product(self, ctx: RunContext, product_id: str) -> str:
        """
        Choose which product the user wants to order.
        """
        self.order.state["product_id"] = product_id
        return "product set"

    # -------------------------------------------------
    # TOOL: Set quantity
    # -------------------------------------------------

    @function_tool
    async def set_quantity(self, ctx: RunContext, quantity: int) -> str:
        self.order.state["quantity"] = quantity
        return "quantity set"

    # -------------------------------------------------
    # TOOL: Save order
    # -------------------------------------------------

    @function_tool
    async def save_order(self, ctx: RunContext) -> str:
        """
        Save completed order to JSON.
        """
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = f"{ORDERS_DIR}/order_{ts}.json"

        with open(filepath, "w") as f:
            json.dump(self.order.state, f, indent=2)

        return filepath

    # -------------------------------------------------
    # HANDLE USER MESSAGE
    # -------------------------------------------------

    async def on_user_message(self, msg, ctx):
        text = msg.text.lower()

        # extract number
        for word in text.split():
            if word.isdigit():
                await ctx.tool_call(self.set_quantity, quantity=int(word))

        # detect name
        if "my name is" in text:
            name = text.split("my name is")[-1].strip().split()[0]
            self.order.state["name"] = name.capitalize()

        # detect product selection
        for item in CATALOG:
            if item["name"].lower() in text or item["category"] in text:
                await ctx.tool_call(self.set_product, product_id=item["id"])
                break

        # Ask next needed field
        if not self.order.is_complete():
            await ctx.llm_response(self.order.next_question())
            return

        # Save order
        filepath = await ctx.tool_call(self.save_order)

        # Respond summary
        p = next(i for i in CATALOG if i["id"] == self.order.state["product_id"])
        summary = (
            f"Thanks {self.order.state['name']}! "
            f"You ordered {self.order.state['quantity']} × {p['name']} "
            f"for {p['price']} {p['currency']} each. Your order is saved!"
        )

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
        preemptive_generation=True
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
        agent=ShoppingAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

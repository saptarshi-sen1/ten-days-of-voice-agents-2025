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
#   PRODUCT CATALOG
# -------------------------------------------------

PRODUCT_CATALOG = {
    "laptop": 55000,
    "mouse": 800,
    "keyboard": 1200,
    "monitor": 9000,
    "headphones": 2500,
    "speakers": 1800,
    "webcam": 1500,
    "pendrive": 600,
}


# -------------------------------------------------
#   ORDER SYSTEM WITH TIMESTAMPS
# -------------------------------------------------

class OrderSystem:
    def __init__(self):
        self.current_order = []      # list of item entries
        self.order_history = self.load_history()

    def load_history(self):
        if not os.path.exists("orders.json"):
            return []
        try:
            with open("orders.json", "r") as f:
                return json.load(f)
        except:
            return []

    def save_history(self):
        with open("orders.json", "w") as f:
            json.dump(self.order_history, f, indent=2)

    def add_item(self, item, qty):
        price = PRODUCT_CATALOG.get(item, 0)

        entry = {
            "item": item,
            "qty": qty,
            "price": price,
            "timestamp": datetime.now().isoformat()
        }

        self.current_order.append(entry)
        return entry

    def compute_total(self):
        return sum(x["price"] * x["qty"] for x in self.current_order)

    def finalize_order(self, name):
        order = {
            "name": name,
            "timestamp": datetime.now().isoformat(),      # order timestamp
            "items": self.current_order,
            "total": self.compute_total()
        }

        self.order_history.append(order)
        self.save_history()
        self.current_order = []  # clear cart
        return order


# -------------------------------------------------
#   E-COMMERCE AGENT
# -------------------------------------------------

class EcommerceAgent(Agent):

    def __init__(self):
        super().__init__(instructions="""
You are an E-commerce Voice Agent that helps customers browse items and place orders.

Your capabilities:
1. Help users add products to their cart.
2. When a user wants to buy something, call add_item(item, qty).
3. When they want to checkout, call finalize_order(name).
4. When they ask about older orders, call get_order_history().
5. Never output JSON directly — always speak naturally.

Always be friendly and helpful.
""")

        self.orders = OrderSystem()

    # ---------- TOOLS ----------

    @function_tool
    async def add_item(self, ctx: RunContext, item: str, qty: int):
        """Add the selected item to the current order."""
        self.orders.add_item(item, qty)
        return f"Added {qty} {item}(s) to your cart."

    @function_tool
    async def finalize_order(self, ctx: RunContext, name: str):
        """Save the order and clear the cart."""
        order = self.orders.finalize_order(name)
        return f"Order placed for {name}. Total amount: Rs {order['total']}."

    @function_tool
    async def get_order_history(self, ctx: RunContext):
        """Fetch previous orders."""
        if not self.orders.order_history:
            return "You have no previous orders."

        summary = ""
        for i, order in enumerate(self.orders.order_history, 1):
            summary += (
                f"Order {i}: {order['name']} ordered {len(order['items'])} items "
                f"on {order['timestamp']}. Total Rs {order['total']}. "
            )
        return summary

    # ---------- MESSAGE HANDLER ----------

    async def on_user_message(self, msg, ctx):
        text = msg.text.lower()

        # detect item request
        for item in PRODUCT_CATALOG:
            if item in text:
                qty = 1
                for w in text.split():
                    if w.isdigit():
                        qty = int(w)
                        break

                await ctx.tool_call(self.add_item, item=item, qty=qty)
                await ctx.llm_response(f"Added {qty} {item}. Anything else?")
                return

        # checkout
        if "checkout" in text or "place order" in text:
            await ctx.llm_response("Sure, what name should I put on the order?")
            return

        # name detection
        if "my name is" in text:
            name = text.split("my name is")[-1].strip().split()[0]
            name = name.capitalize()

            await ctx.tool_call(self.finalize_order, name=name)
            await ctx.llm_response(f"Thanks {name}! Your order is confirmed.")
            return

        # order history
        if "previous orders" in text or "order history" in text:
            result = await ctx.tool_call(self.get_order_history)
            await ctx.llm_response(result)
            return

        await ctx.llm_response(
            "How can I help you today? You can ask for laptops, keyboards, monitors, and more!"
        )


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
        agent=EcommerceAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

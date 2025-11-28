import logging
import json
from pathlib import Path
from datetime import datetime
import uuid

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
    tokenize
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# ----------------------------
# File paths
# ----------------------------
CATALOG_FILE = Path("catalog.json")
CURRENT_CART_FILE = Path("current_cart.json")
ORDER_HISTORY_FILE = Path("order_history.json")
TRACKING_FILE = Path("order_tracking.json")

# ----------------------------
# Shopping state
# ----------------------------
class ShoppingState:
    def __init__(self):
        self.cart = self.load_cart()

    def load_cart(self):
        if CURRENT_CART_FILE.exists():
            try:
                with open(CURRENT_CART_FILE, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_cart(self):
        with open(CURRENT_CART_FILE, "w") as f:
            json.dump(self.cart, f, indent=2)

    def add_item(self, item: str, qty: int):
        self.cart[item] = self.cart.get(item, 0) + qty
        self.save_cart()

    def remove_item(self, item: str):
        if item in self.cart:
            del self.cart[item]
            self.save_cart()

    def list_cart(self):
        if not self.cart:
            return "Your cart is empty."
        msg = "Your cart:\n"
        for item, qty in self.cart.items():
            msg += f"- {item}: {qty}\n"
        return msg

    def reset(self):
        self.cart = {}
        self.save_cart()

# ----------------------------
# Recipes
# ----------------------------
RECIPES = {
    "peanut butter sandwich": ["bread", "peanut butter"],
    "pasta": ["pasta", "pasta sauce"],
    "maggi": ["maggi noodles", "water bottle"],
    "omelette": ["eggs", "milk", "salt", "pepper"],
    "grilled cheese": ["bread", "cheese", "butter"],
}

# ----------------------------
# Shopping Agent
# ----------------------------
class ShoppingAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are a friendly food & grocery ordering assistant.
You can add/remove items, list the cart, add ingredients for recipes, place orders, and track orders.
"""
        )
        self.state = ShoppingState()
        self.catalog = {}
        if CATALOG_FILE.exists():
            try:
                with open(CATALOG_FILE, "r") as f:
                    self.catalog = {k.lower(): v for k, v in json.load(f).items()}
            except:
                self.catalog = {}

        print("Catalog loaded:", self.catalog.keys())

    # ------------------------- TOOLS -------------------------
    @function_tool
    async def add_item_to_cart(self, ctx: RunContext, item: str, quantity: int) -> str:
        item = item.lower()
        if item not in self.catalog:
            return f"I am sorry, I cannot fulfill this request. '{item}' is not in the catalog."
        self.state.add_item(item, quantity)
        return f"Added {quantity} x {item} to your cart."

    @function_tool
    async def remove_item(self, ctx: RunContext, item: str) -> str:
        item = item.lower()
        self.state.remove_item(item)
        return f"Removed {item} from your cart."

    @function_tool
    async def show_cart(self, ctx: RunContext) -> str:
        return self.state.list_cart()

    @function_tool
    async def add_recipe_ingredients(self, ctx: RunContext, dish: str) -> str:
        dish = dish.lower()
        if dish not in RECIPES:
            return f"I don't know the recipe for {dish}."
        added = []
        for item in RECIPES[dish]:
            if item in self.catalog:
                self.state.add_item(item, 1)
                added.append(item)
        return f"Ingredients for {dish} have been added to your cart: {', '.join(added)}"

    @function_tool
    async def place_order(self, ctx: RunContext) -> str:
        if not self.state.cart:
            return "Your cart is empty. Please add items before placing an order."

        order_id = str(uuid.uuid4())[:8]
        total = sum(self.catalog[item]["price"] * qty for item, qty in self.state.cart.items())

        order = {
            "order_id": order_id,
            "timestamp": datetime.now().isoformat(),
            "items": [
                {
                    "name": item,
                    "quantity": qty,
                    "unit_price": self.catalog[item]["price"],
                    "total_price": self.catalog[item]["price"] * qty
                }
                for item, qty in self.state.cart.items()
            ],
            "order_total": total
        }

        # Save history
        history = []
        if ORDER_HISTORY_FILE.exists():
            try:
                with open(ORDER_HISTORY_FILE, "r") as f:
                    history = json.load(f)
            except:
                history = []
        history.append(order)
        with open(ORDER_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)

        # Save tracking
        tracking = {}
        if TRACKING_FILE.exists():
            try:
                with open(TRACKING_FILE, "r") as f:
                    tracking = json.load(f)
            except:
                tracking = {}
        tracking[order_id] = {"status": "Order Placed", "last_updated": datetime.now().isoformat()}
        with open(TRACKING_FILE, "w") as f:
            json.dump(tracking, f, indent=2)

        self.state.reset()
        return f"Order {order_id} placed! Total ₹{total}."

    @function_tool
    async def track_order(self, ctx: RunContext, order_id: str) -> str:
        tracking = {}
        if TRACKING_FILE.exists():
            try:
                with open(TRACKING_FILE, "r") as f:
                    tracking = json.load(f)
            except:
                tracking = {}
        if order_id not in tracking:
            return "Order ID not found."
        return f"Status for {order_id}: {tracking[order_id]['status']}"

    # ------------------------- MESSAGE HANDLER -------------------------
    async def on_user_message(self, msg, ctx):
        text = msg.text.lower()

        # Commands
        if "cart" in text:
            await ctx.llm_response(self.state.list_cart())
            return
        elif "place order" in text or "checkout" in text:
            await ctx.tool_call(self.place_order)
            return
        elif "track" in text:
            words = text.split()
            for word in words:
                if len(word) == 8:  # assume order_id
                    await ctx.tool_call(self.track_order, order_id=word)
                    return
        elif text.startswith("i want to make") or text.startswith("ingredients for"):
            # recipe request
            dish = text.replace("i want to make", "").replace("ingredients for", "").strip()
            await ctx.tool_call(self.add_recipe_ingredients, dish=dish)
            return
        else:
            await ctx.llm_response("How can I help you with your shopping today?")

# ----------------------------
# PREWARM + ENTRYPOINT
# ----------------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.add_shutdown_callback(lambda: logger.info("Agent shutting down."))

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

    await session.start(
        agent=ShoppingAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    await ctx.connect()

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

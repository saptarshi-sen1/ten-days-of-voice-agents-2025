import logging
import json
import os
from pathlib import Path
from datetime import datetime
import random

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
#   PERSISTENCE SETUP
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
IMPROV_DIR = BASE_DIR / "improv_battle"
IMPROV_DIR.mkdir(exist_ok=True)


def save_improv_state(room_name: str, state: dict) -> str:
    """Save improv game state to a JSON file."""
    filename = f"improv_{room_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    filepath = IMPROV_DIR / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved improv state to {filepath}")
    return str(filepath)


def load_improv_state(room_name: str) -> dict:
    """Load the most recent improv state for a room, if it exists."""
    pattern = f"improv_{room_name}_*.json"
    existing_files = sorted(IMPROV_DIR.glob(pattern), key=os.path.getmtime, reverse=True)
    
    if existing_files:
        try:
            with open(existing_files[0], "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
    
    return None


# -------------------------------------------------
#   IMPROV SCENARIOS
# -------------------------------------------------

IMPROV_SCENARIOS = [
    "You are a barista who has to tell a customer that their latte is actually a portal to another dimension.",
    "You are a time-travelling tour guide explaining modern smartphones to someone from the 1800s.",
    "You are a restaurant waiter who must calmly tell a customer that their order has escaped the kitchen.",
    "You are a customer trying to return an obviously cursed object to a very skeptical shop owner.",
    "You are a weather reporter who has just discovered that clouds are actually sentient beings.",
    "You are a librarian who must explain to a child that books are actually sleeping and shouldn't be woken up.",
    "You are a delivery driver delivering a package to someone who claims they never ordered anything, but the package is clearly addressed to them.",
    "You are a museum tour guide showing visitors an exhibit that keeps changing when you're not looking.",
    "You are a flight attendant announcing that the plane will be landing on a cloud instead of a runway.",
    "You are a doctor explaining to a patient that their X-ray shows they have a tiny orchestra playing inside their chest.",
]


# -------------------------------------------------
#   IMPROV BATTLE AGENT
# -------------------------------------------------

class ImprovBattleAgent(Agent):

    def __init__(self, room_name: str = None):
        super().__init__(instructions="""
You are the host of a TV improv show called "Improv Battle".

Your role:
- You are a high-energy, witty, and entertaining game show host
- You clearly explain the rules and keep the game moving
- You react to performances in a varied, realistic way

Your style:
- High-energy and engaging
- Witty and quick with observations
- Clear about rules and expectations
- Reactions should be realistic: sometimes amused, sometimes unimpressed, sometimes pleasantly surprised
- Light teasing and honest critique are allowed, but stay respectful and non-abusive
- Mix positive and critical feedback naturally

Game structure:
1. Introduce the show and explain the basic rules
2. Run through several improv rounds (typically 3-5)
3. For each round:
   - Set a scenario clearly
   - Ask the player to improvise
   - Wait for them to perform
   - Once they finish (they say "end scene", "that's it", or pause significantly), react and move on
4. At the end, provide a closing summary

Reactions:
- Comment on what worked, what was weird, or what was flat
- Mix positive and critical feedback:
  * Sometimes: "That was hilarious, especially the part where..."
  * Sometimes: "That felt a bit rushed; you could have leaned more into the character."
- Randomly choose between more supportive, neutral, or mildly critical tones, while staying constructive and safe

Closing:
- Summarize what kind of improviser the player seemed to be (emphasis on character, absurdity, emotional range, etc.)
- Mention specific moments or scenes that stood out
- Thank the player and close the show

Early exit:
- If the user clearly indicates they want to stop (e.g., "stop game", "end show", "I'm done"), confirm and gracefully end the session

Keep responses concise and energetic. This is a performance, so be entertaining!
""")

        self.room_name = room_name or "default"
        
        # Temporary tracking (not saved in state)
        self.current_scenario = None
        self.user_turns_in_scene = 0
        
        # Try to load existing state, otherwise initialize
        saved_state = load_improv_state(self.room_name)
        if saved_state:
            self.improv_state = saved_state
            logger.info(f"Loaded existing state for room {self.room_name}")
        else:
            # Initialize game state - exactly as specified
            self.improv_state = {
                "player_name": None,
                "current_round": 0,
                "max_rounds": 3,
                "rounds": [],  # each: {"scenario": str, "host_reaction": str}
                "phase": "intro",  # "intro" | "awaiting_improv" | "reacting" | "done"
            }
            self._save_state()
    
    def _save_state(self):
        """Save current state to file."""
        try:
            save_improv_state(self.room_name, self.improv_state)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    # ---------- TOOLS ----------

    @function_tool
    async def set_player_name(self, ctx: RunContext, name: str):
        """Set the player's name from their introduction or input."""
        self.improv_state["player_name"] = name
        self._save_state()
        return f"Player name set to {name}"

    @function_tool
    async def start_round(self, ctx: RunContext, scenario: str):
        """Start a new improv round with the given scenario."""
        self.improv_state["current_round"] += 1
        self.improv_state["phase"] = "awaiting_improv"
        self.current_scenario = scenario
        self.user_turns_in_scene = 0
        self._save_state()
        return f"Round {self.improv_state['current_round']} started with scenario: {scenario}"

    @function_tool
    async def end_round(self, ctx: RunContext, reaction: str):
        """End the current round and store the host's reaction."""
        # Store reaction in the most recent round
        if self.improv_state["rounds"]:
            self.improv_state["rounds"][-1]["host_reaction"] = reaction
        self.improv_state["phase"] = "reacting"
        self._save_state()
        return "Round ended and reaction stored"

    @function_tool
    async def end_game(self, ctx: RunContext):
        """Mark the game as complete."""
        self.improv_state["phase"] = "done"
        self._save_state()
        return "Game ended"

    # ---------- MESSAGE HANDLER ----------

    async def on_user_message(self, msg, ctx: RunContext):
        user_text = msg.text.strip().lower()
        logger.info(f"User message: {user_text}")

        # Extract player name from first message if not set
        if not self.improv_state["player_name"]:
            # Try to get name from room participant metadata if available
            try:
                if hasattr(ctx.session, 'room') and ctx.session.room:
                    participants = ctx.session.room.remote_participants.values()
                    for participant in participants:
                        if participant.metadata:
                            try:
                                metadata = json.loads(participant.metadata)
                                if "playerName" in metadata:
                                    await ctx.tool_call(self.set_player_name, name=metadata["playerName"])
                                    break
                            except:
                                pass
            except:
                pass
            
            # If still not set, try to extract from message
            if not self.improv_state["player_name"]:
                if "my name is" in user_text:
                    name = user_text.split("my name is")[-1].strip().split()[0]
                    name = name.capitalize()
                    await ctx.tool_call(self.set_player_name, name=name)
                elif "i'm" in user_text and len(user_text.split()) <= 5:
                    # Simple "I'm John" pattern
                    parts = user_text.split("i'm")
                    if len(parts) > 1:
                        name = parts[-1].strip().split()[0]
                        name = name.capitalize()
                        await ctx.tool_call(self.set_player_name, name=name)
                else:
                    # Default name
                    await ctx.tool_call(self.set_player_name, name="Contestant")

        # Check for early exit
        exit_phrases = ["stop game", "end show", "i'm done", "that's all", "end the game", "stop the game"]
        if any(phrase in user_text for phrase in exit_phrases) and self.improv_state["phase"] != "done":
            await ctx.tool_call(self.end_game)
            await ctx.llm_response(
                "Alright, we're wrapping up! Thanks for playing Improv Battle. Hope you had fun!"
            )
            return

        # Handle game phases
        if self.improv_state["phase"] == "intro":
            # Start the first round
            scenario = random.choice(IMPROV_SCENARIOS)
            await ctx.tool_call(self.start_round, scenario=scenario)
            
            player_name = self.improv_state["player_name"] or "Contestant"
            intro = f"""Welcome to Improv Battle! I'm your host, and you're our contestant, {player_name}!

Here's how it works: I'll give you a scenario, and you'll improvise a scene. Just act it out, get into character, and have fun with it. When you're done, say "end scene" or just pause, and I'll give you my thoughts.

Ready? Let's start with Round 1!

{scenario}

Go ahead, {player_name} - the stage is yours!"""
            await ctx.llm_response(intro)
            return

        elif self.improv_state["phase"] == "awaiting_improv":
            # Player is performing
            self.user_turns_in_scene += 1
            
            # Check if scene is ending
            # Use simple heuristics: specific phrase ("End scene", "Okay") or maximum number of user turns
            end_indicators = ["end scene", "that's it", "that's all", "okay", "ok", "scene", "done", "finished", "the end"]
            is_ending = any(indicator in user_text for indicator in end_indicators)
            
            # Also end if they've had multiple turns (heuristic: max 3 turns)
            if is_ending or self.user_turns_in_scene >= 3:
                # Store the round with scenario - reaction will be updated after LLM response
                self.improv_state["rounds"].append({
                    "scenario": self.current_scenario,
                    "host_reaction": "",  # Will be filled after LLM generates response
                })
                self._save_state()
                
                # Move phase to reacting
                self.improv_state["phase"] = "reacting"
                self._save_state()
                
                # Check if more rounds
                if self.improv_state["current_round"] < self.improv_state["max_rounds"]:
                    # Generate reaction and transition to next round
                    reaction_prompt = f"""
The player just finished their improv scene for this scenario: "{self.current_scenario}"

Their performance included: "{user_text}"

Now give your reaction as the host. Be varied - randomly choose between:
- Amused and supportive: "That was hilarious, especially the part where..."
- Neutral with constructive feedback: "Interesting take. I noticed..."
- Mildly critical but constructive: "That felt a bit rushed; you could have leaned more into the character."

Comment on what worked, what was weird, or what was flat. Keep it short (2-3 sentences), entertaining, and constructive.

After your reaction, immediately transition to the next round by saying something like "Great! Now let's move to Round {self.improv_state['current_round'] + 1}!" and then present the next scenario.

Current round: {self.improv_state['current_round']} of {self.improv_state['max_rounds']}
"""
                    # Get next scenario
                    next_scenario = random.choice(IMPROV_SCENARIOS)
                    await ctx.tool_call(self.start_round, scenario=next_scenario)
                    
                    # Generate reaction with next scenario
                    full_prompt = f"""{reaction_prompt}

Next scenario for Round {self.improv_state['current_round']}: {next_scenario}

After your reaction, present this next scenario and ask the player to start improvising.
"""
                    await ctx.llm_response(full_prompt)
                    # Note: We can't capture the LLM response text directly, but the reaction is in the conversation
                    # The state will be saved with empty reaction, which is acceptable
                else:
                    # Game over - closing summary
                    await ctx.tool_call(self.end_game)
                    summary_prompt = f"""
The player just finished their final improv scene for this scenario: "{self.current_scenario}"

Their performance included: "{user_text}"

Give your reaction to this final scene (2-3 sentences), then provide a closing summary:
- Summarize what kind of improviser the player seemed to be (emphasis on character, absurdity, emotional range, etc.)
- Mention specific moments or scenes that stood out
- Thank the player and close the show

Keep it warm, entertaining, and about 4-6 sentences total.
"""
                    await ctx.llm_response(summary_prompt)
            else:
                # Still in the scene - let them continue without interrupting
                return

        elif self.improv_state["phase"] == "reacting":
            # Transitioning between rounds - should move to awaiting_improv
            self.improv_state["phase"] = "awaiting_improv"
            self._save_state()
            await ctx.llm_response("Let's continue!")
            return

        elif self.improv_state["phase"] == "done":
            # Game is over
            await ctx.llm_response("Thanks for playing Improv Battle! The show is over. Have a great day!")
            return

        # Fallback
        await ctx.llm_response("I'm not sure what you mean. Let's keep the improv going!")


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
        agent=ImprovBattleAgent(room_name=ctx.room.name),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))


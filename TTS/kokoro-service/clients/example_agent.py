"""
Example LiveKit voice agent using Kokoro TTS.

This demonstrates how to use the KokoroTTS plugin with a LiveKit agent
for real-time voice conversation in Brazilian Portuguese.

Usage:
    python clients/example_agent.py
    
Requirements:
    pip install livekit-agents livekit-plugins-deepgram livekit-plugins-openai
"""

import asyncio
import logging

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import deepgram, openai

from livekit_plugin.kokoro_tts import KokoroTTS

logger = logging.getLogger(__name__)


class BrazilianAssistant(Agent):
    """
    Voice agent for Brazilian Portuguese using Kokoro TTS.
    
    Features:
    - Speech-to-text with Deepgram (pt-BR)
    - LLM with OpenAI GPT-4
    - Text-to-speech with Kokoro (pf_dora voice)
    """

    def __init__(self, tts_service_url: str = "ws://kokoro-tts:8000/ws/tts"):
        super().__init__(
            instructions="""
            Você é um assistente virtual brasileiro amigável e prestativo.
            Responda sempre em português brasileiro de forma natural e clara.
            Mantenha respostas concisas para uma experiência de voz fluida.
            Evite respostas muito longas - máximo 2-3 frases por vez.
            """,
            stt=deepgram.STT(language="pt-BR"),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=KokoroTTS(
                service_url=tts_service_url,
                voice="pf_dora",  # Female pt-BR voice
                lang_code="p",
            ),
        )

    async def on_enter(self) -> None:
        """Initial greeting when user joins."""
        await self.session.say("Olá! Como posso ajudar você hoje?")


async def entrypoint(ctx: JobContext) -> None:
    """
    Agent entrypoint.
    
    Called when a new participant joins the room.
    """
    # Get TTS service URL from environment or use default
    import os
    tts_url = os.environ.get("KOKORO_TTS_URL", "ws://localhost:8000/ws/tts")
    
    # Create and start the agent
    agent = BrazilianAssistant(tts_service_url=tts_url)
    
    await ctx.connect()
    
    # Wait for a participant to join
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")
    
    # Start the agent session
    session = AgentSession()
    await session.start(ctx.room, participant=participant, agent=agent)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

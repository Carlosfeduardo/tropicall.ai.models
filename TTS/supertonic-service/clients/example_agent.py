"""
Example LiveKit voice agent using Supertonic TTS.

This demonstrates how to use the SupertonicTTS plugin with a LiveKit agent
for real-time voice conversation with multilingual support.

Usage:
    python clients/example_agent.py
    
Requirements:
    pip install livekit-agents livekit-plugins-deepgram livekit-plugins-openai
"""

import asyncio
import logging

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import deepgram, openai

from livekit_plugin.supertonic_tts import SupertonicTTS

logger = logging.getLogger(__name__)


class MultilingualAssistant(Agent):
    """
    Voice agent using Supertonic 2 TTS.
    
    Features:
    - Speech-to-text with Deepgram
    - LLM with OpenAI GPT-4
    - Text-to-speech with Supertonic 2 (multilingual)
    
    Supported languages: en, ko, es, pt, fr
    """

    def __init__(
        self,
        tts_service_url: str = "ws://supertonic-tts:8000/ws/tts",
        lang_code: str = "pt",
        voice: str = "F1",
    ):
        # Map language codes to Deepgram languages
        deepgram_lang_map = {
            "en": "en-US",
            "ko": "ko",
            "es": "es",
            "pt": "pt-BR",
            "fr": "fr",
        }
        
        # Map language codes to instructions
        instructions_map = {
            "pt": """
            Você é um assistente virtual brasileiro amigável e prestativo.
            Responda sempre em português brasileiro de forma natural e clara.
            Mantenha respostas concisas para uma experiência de voz fluida.
            Evite respostas muito longas - máximo 2-3 frases por vez.
            """,
            "en": """
            You are a friendly and helpful virtual assistant.
            Always respond in English naturally and clearly.
            Keep responses concise for a smooth voice experience.
            Avoid very long responses - maximum 2-3 sentences at a time.
            """,
            "es": """
            Eres un asistente virtual amigable y servicial.
            Responde siempre en español de forma natural y clara.
            Mantén las respuestas concisas para una experiencia de voz fluida.
            Evita respuestas muy largas - máximo 2-3 oraciones a la vez.
            """,
            "fr": """
            Vous êtes un assistant virtuel amical et serviable.
            Répondez toujours en français de manière naturelle et claire.
            Gardez les réponses concises pour une expérience vocale fluide.
            Évitez les réponses très longues - maximum 2-3 phrases à la fois.
            """,
            "ko": """
            당신은 친근하고 도움이 되는 가상 비서입니다.
            항상 자연스럽고 명확하게 한국어로 대답하세요.
            원활한 음성 경험을 위해 답변을 간결하게 유지하세요.
            너무 긴 답변은 피하세요 - 한 번에 최대 2-3문장.
            """,
        }
        
        super().__init__(
            instructions=instructions_map.get(lang_code, instructions_map["en"]),
            stt=deepgram.STT(language=deepgram_lang_map.get(lang_code, "en-US")),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=SupertonicTTS(
                service_url=tts_service_url,
                voice=voice,
                lang_code=lang_code,
            ),
        )
        self.lang_code = lang_code

    async def on_enter(self) -> None:
        """Initial greeting when user joins."""
        greetings = {
            "pt": "Olá! Como posso ajudar você hoje?",
            "en": "Hello! How can I help you today?",
            "es": "¡Hola! ¿Cómo puedo ayudarte hoy?",
            "fr": "Bonjour! Comment puis-je vous aider aujourd'hui?",
            "ko": "안녕하세요! 오늘 무엇을 도와드릴까요?",
        }
        await self.session.say(greetings.get(self.lang_code, greetings["en"]))


async def entrypoint(ctx: JobContext) -> None:
    """
    Agent entrypoint.
    
    Called when a new participant joins the room.
    """
    import os
    
    # Get configuration from environment
    tts_url = os.environ.get("SUPERTONIC_TTS_URL", "ws://localhost:8000/ws/tts")
    lang_code = os.environ.get("TTS_LANG_CODE", "pt")
    voice = os.environ.get("TTS_VOICE", "F1")
    
    # Create and start the agent
    agent = MultilingualAssistant(
        tts_service_url=tts_url,
        lang_code=lang_code,
        voice=voice,
    )
    
    await ctx.connect()
    
    # Wait for a participant to join
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")
    
    # Start the agent session
    session = AgentSession()
    await session.start(ctx.room, participant=participant, agent=agent)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

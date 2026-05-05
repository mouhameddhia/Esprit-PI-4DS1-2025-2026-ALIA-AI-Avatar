"""
LLM Service - Ollama Integration
Handles conversational AI using local Ollama models
"""

import asyncio
import logging
from typing import Dict, Optional
import uuid
from datetime import datetime
import httpx

from utils.config import config
from services.rag_service import RAGService
from services.hybrid_medical_service import HybridMedicalService

logger = logging.getLogger(__name__)

class LLMService:
    """Service for LLM inference using Ollama"""
    
    def __init__(self):
        self.ollama_url = config.OLLAMA_URL
        self.model = config.OLLAMA_MODEL
        self.conversations: Dict[str, list] = {}
        self.client = None
        self.rag_service = RAGService()
        self.medical_service = HybridMedicalService()  # Medical rep training agent
        
    async def initialize(self):
        """Initialize Ollama connection"""
        try:
            # Increase timeout for slow LLM responses (especially first run)
            # Using 300s (5 min) for first model load, then responses are faster
            self.client = httpx.AsyncClient(timeout=300.0)
            
            # Check if Ollama is running
            response = await self.client.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                logger.info(f"Connected to Ollama at {self.ollama_url}")
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                logger.info(f"Available models: {model_names}")
                
                if not any(self.model in name for name in model_names):
                    logger.warning(f"Model '{self.model}' not found. Pulling model...")
                    await self.pull_model()
            else:
                logger.error("Failed to connect to Ollama")

            if config.RAG_ENABLED:
                try:
                    logger.info("Warming RAG worker during backend startup")
                    await asyncio.to_thread(self.rag_service.warmup)
                except Exception as warmup_error:
                    logger.error(f"RAG warmup failed: {warmup_error}", exc_info=True)
            
            if config.USE_HYBRID_MEDICAL_AGENT and config.MEDICAL_REP_TRAINING_ENABLED:
                try:
                    logger.info(f"Warming hybrid medical agent at competency level: {config.MEDICAL_COMPETENCY_LEVEL}")
                    await asyncio.to_thread(self.medical_service.warmup)
                except Exception as medical_warmup_error:
                    logger.error(f"Medical agent warmup failed: {medical_warmup_error}", exc_info=True)
                
        except Exception as e:
            logger.error(f"Error initializing LLM service: {e}")
            logger.info("Make sure Ollama is running: ollama serve")
    
    async def pull_model(self):
        """Pull the specified model if not available"""
        try:
            logger.info(f"Pulling model {self.model}... This may take a few minutes.")
            async with self.client.stream(
                'POST',
                f"{self.ollama_url}/api/pull",
                json={"name": self.model}
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        logger.info(f"Pull progress: {line}")
            logger.info(f"Model {self.model} pulled successfully")
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
    
    def is_ready(self) -> bool:
        """Check if service is ready"""
        return self.client is not None

    @staticmethod
    def _resolve_use_rag(use_rag: Optional[object]) -> bool:
        if config.RAG_FORCE_ENABLED:
            return True
        if use_rag is None:
            return bool(config.RAG_DEFAULT_ENABLED)
        if isinstance(use_rag, str):
            normalized = use_rag.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off"}:
                return False
        return bool(use_rag)
    
    async def generate_response(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        use_rag: Optional[bool] = None,
        competency_level: Optional[str] = None,
    ) -> Dict:
        """
        Generate response from LLM
        
        Args:
            message: User input message
            conversation_id: Optional conversation ID for context
            system_prompt: Optional system prompt override
            
        Returns:
            Dictionary with conversation_id, message, and metadata
        """
        try:
            start_time = datetime.now()
            
            # Create or retrieve conversation
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = []
            
            # Build conversation history
            messages = self.conversations[conversation_id].copy()

            # Keep context bounded to avoid latency growth over long sessions.
            # Preserve the system prompt plus the most recent user/assistant turns.
            if len(messages) > 1:
                system_msg = messages[0] if messages[0].get("role") == "system" else None
                recent_turns = messages[1:] if system_msg else messages
                max_turn_messages = max(2, config.MAX_CONVERSATION_HISTORY * 2)
                recent_turns = recent_turns[-max_turn_messages:]
                messages = [system_msg] + recent_turns if system_msg else recent_turns
            
            # Add system prompt if provided or use default
            if not messages:
                default_system = system_prompt or config.SYSTEM_PROMPT
                messages.append({
                    "role": "system",
                    "content": default_system
                })
            
            # Add user message
            messages.append({
                "role": "user",
                "content": message
            })

            rag_requested = self._resolve_use_rag(use_rag)
            rag_payload = None
            logger.info(
                "RAG routing: requested=%s enabled=%s default=%s force=%s",
                use_rag,
                config.RAG_ENABLED,
                config.RAG_DEFAULT_ENABLED,
                config.RAG_FORCE_ENABLED,
            )
            # Route to hybrid medical agent when enabled (same stack used by Streamlit integration).
            if config.USE_HYBRID_MEDICAL_AGENT and config.MEDICAL_REP_TRAINING_ENABLED:
                # Keep per-session medical state and allow runtime level override from Unreal.
                if competency_level:
                    normalized_level = str(competency_level).strip().upper()
                    if normalized_level in {"BEGINNER", "JUNIOR", "CONFIRMED", "EXPERT"}:
                        self.medical_service.set_competency_level(normalized_level)

                mode = "training" if str(getattr(config, "MEDICAL_AGENT_MODE", "training")).lower() == "training" else "commercial"
                preferred_language = str(getattr(config, "MEDICAL_PREFERRED_LANGUAGE", "auto") or "auto").strip().lower()
                if preferred_language not in {"en", "fr"}:
                    preferred_language = None

                preferred_drug = str(getattr(config, "MEDICAL_PREFERRED_DRUG", "") or "").strip() or None
                if preferred_drug and preferred_drug.lower() in {"auto", "auto-detect", "(auto-detect)"}:
                    preferred_drug = None

                # Use non-system conversation turns for medical context.
                prior_turns = messages[:-1]
                history = [m for m in prior_turns if m.get("role") in {"user", "assistant"}]
                max_turn_messages = max(2, config.MAX_CONVERSATION_HISTORY * 2)
                history = history[-max_turn_messages:]

                medical_response = await asyncio.to_thread(
                    self.medical_service.handle_query,
                    message,
                    mode,
                    conversation_id,
                    preferred_drug,
                    preferred_language,
                    history,
                    None,
                    False,
                )
                assistant_message = str(medical_response.get("answer", "")).strip()
                if not assistant_message:
                    assistant_message = "Information not available in the provided data"
                tokens = 0
                rag_payload = {
                    "hybrid": True,
                    "source": medical_response.get("source"),
                    "topic": medical_response.get("topic"),
                    "drug_name": medical_response.get("drug_name"),
                }
            elif rag_requested:
                if not config.RAG_ENABLED:
                    raise RuntimeError("RAG requested but RAG_ENABLED is false")
                rag_payload = await asyncio.to_thread(self.rag_service.query, message)
                assistant_message = str(rag_payload.get("answer", "")).strip()
                if not assistant_message:
                    raise RuntimeError("RAG returned an empty answer")
                tokens = 0
            else:
                # Call Ollama API (with automatic CPU fallback on CUDA OOM)
                response = await self._post_chat_with_fallback(messages)
                result = response.json()
                assistant_message = result["message"]["content"]
                tokens = result.get("eval_count", 0)
            
            # Update conversation history
            messages.append({
                "role": "assistant",
                "content": assistant_message
            })

            # Persist bounded history.
            system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
            recent_turns = messages[1:] if system_msg else messages
            max_turn_messages = max(2, config.MAX_CONVERSATION_HISTORY * 2)
            recent_turns = recent_turns[-max_turn_messages:]
            self.conversations[conversation_id] = [system_msg] + recent_turns if system_msg else recent_turns
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"LLM response generated in {duration:.2f}s")

            rag_meta = None
            if rag_payload:
                rag_meta = {
                    "citations": rag_payload.get("citations"),
                    "answer_confidence": rag_payload.get("answer_confidence"),
                    "uncertainty": rag_payload.get("uncertainty"),
                    "conflict_notes": rag_payload.get("conflict_notes"),
                    "diagnostics": rag_payload.get("diagnostics"),
                    "latency_ms": rag_payload.get("latency_ms"),
                }

            return {
                "conversation_id": conversation_id,
                "message": assistant_message,
                "duration": duration,
                "model": config.RAG_OLLAMA_MODEL if rag_requested else self.model,
                "tokens": tokens,
                "rag": rag_meta,
            }
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}", exc_info=True)
            raise

    async def _post_chat_with_fallback(self, messages: list):
        """
        Send chat request to Ollama. If the GPU runner fails with CUDA OOM,
        retry once with CPU-only settings.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": config.TEMPERATURE,
                "top_p": config.TOP_P,
                # Ollama uses num_predict; keep max_tokens for backward compatibility.
                "num_predict": config.MAX_TOKENS,
                "max_tokens": config.MAX_TOKENS,
                "num_ctx": 1024,
            },
        }

        response = await self.client.post(f"{self.ollama_url}/api/chat", json=payload)
        if response.status_code < 400:
            return response

        error_text = ""
        try:
            error_text = response.text
        except Exception:
            error_text = ""

        if "cudaMalloc failed" in error_text or "out of memory" in error_text.lower():
            logger.warning("Ollama GPU OOM detected. Retrying on CPU (num_gpu=0).")
            payload["options"]["num_gpu"] = 0
            payload["options"]["num_ctx"] = 1024
            retry = await self.client.post(f"{self.ollama_url}/api/chat", json=payload)
            retry.raise_for_status()
            return retry

        response.raise_for_status()
        return response
    
    async def generate_streaming_response(
        self,
        message: str,
        conversation_id: Optional[str] = None
    ):
        """
        Generate streaming response from LLM (for future implementation)
        
        Yields chunks of text as they're generated
        """
        try:
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = []
            
            messages = self.conversations[conversation_id].copy()
            
            if not messages:
                messages.append({
                    "role": "system",
                    "content": config.SYSTEM_PROMPT
                })
            
            messages.append({
                "role": "user",
                "content": message
            })
            
            full_response = ""
            
            async with self.client.stream(
                'POST',
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True
                }
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        if "message" in data:
                            chunk = data["message"]["content"]
                            full_response += chunk
                            yield {
                                "type": "chunk",
                                "content": chunk,
                                "done": data.get("done", False)
                            }
            
            # Update conversation
            messages.append({
                "role": "assistant",
                "content": full_response
            })
            self.conversations[conversation_id] = messages
            
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            raise
    
    def clear_conversation(self, conversation_id: str):
        """Clear conversation history"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Cleared conversation {conversation_id}")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.client:
            await self.client.aclose()
        logger.info("LLM service cleaned up")

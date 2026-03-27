#!/usr/bin/env python3
"""
Simple interactive dialog to test Agent 1 (Conversation Manager).

Usage:
  python backend/agents/conversation_manager/quick_test.py
"""

from agent1 import ConversationManagerAgent
from schemas import ConversationState
import sys
from pathlib import Path
import socket

# Allow importing the Knowledge Agent when running this file directly.
BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
	sys.path.append(str(BACKEND_DIR))

from agents.Knowledge_agent.agent2 import KnowledgeComplianceAgent


def _is_ollama_running(host: str = "localhost", port: int = 11434) -> bool:
	"""Quick socket check to avoid hanging when Ollama is down."""
	try:
		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		sock.settimeout(2)
		result = sock.connect_ex((host, port))
		sock.close()
		return result == 0
	except OSError:
		return False


def run_dialog() -> None:
	print("=" * 60)
	print("Agent 1 Dialog Test")
	print("=" * 60)
	print("Type 'exit' to quit, 'reset' to reset conversation state.\n")

	if not _is_ollama_running():
		print("Ollama is not running on localhost:11434.")
		print("Start it first with: ollama serve")
		return

	try:
		mode = input("Choose mode (training/doctor) [training]: ").strip().lower() or "training"
	except (EOFError, KeyboardInterrupt):
		print("\nStopped before starting dialog.")
		return

	if mode not in {"training", "doctor"}:
		print("Invalid mode, defaulting to 'training'.")
		mode = "training"

	try:
		agent = ConversationManagerAgent(model_name="llama3:8b")
	except Exception as e:
		print(f"Failed to initialize agent: {e}")
		return

	try:
		knowledge_agent = KnowledgeComplianceAgent(model_name="llama3:8b")
		sync_info = knowledge_agent.sync_knowledge_base("backend/data/rag_docs", rebuild=False)
		print(f"Knowledge Agent ready. Synced files: {sync_info.get('files_synced', 0)}, chunks: {sync_info.get('chunks_indexed', 0)}")
	except Exception as e:
		print(f"Knowledge Agent init failed: {e}")
		knowledge_agent = None

	state = ConversationState()

	print(f"\nRunning in '{mode}' mode. Start chatting:\n")

	while True:
		try:
			user_input = input("You: ").strip()
		except EOFError:
			print("\nInput stream ended. Exiting.")
			break
		except KeyboardInterrupt:
			print("\nInterrupted. Exiting.")
			break

		if not user_input:
			continue
		if user_input.lower() == "exit":
			print("Goodbye.")
			break
		if user_input.lower() == "reset":
			state = ConversationState()
			print("Conversation state reset.\n")
			continue

		try:
			result = agent.process(
				user_input=user_input,
				system_mode=mode,
				conversation_state=state.to_dict(),
			)
		except Exception as e:
			print(f"Agent processing failed: {e}")
			print("Check Ollama status and model availability (llama3:8b).\n")
			continue

		state.update(result.get("intent", "unclear"), result.get("entities", []))

		print("\nAgent 1:")
		print(f"  intent: {result.get('intent')}")
		print(f"  entities: {result.get('entities')}")
		print(f"  dialogue_state: {result.get('dialogue_state')}")
		print(f"  next_agent: {result.get('next_agent')}\n")

		if result.get("next_agent") == "knowledge_retrieval" and knowledge_agent is not None:
			try:
				rag_result = knowledge_agent.process(
					user_input=user_input,
					intent=result.get("intent", "general_question"),
					entities=result.get("entities", []),
				)
				print("Knowledge Agent:")
				print(f"  answer: {rag_result.get('answer')}")
				print(f"  citations: {rag_result.get('citations')}")
				print(f"  confidence: {rag_result.get('confidence')}\n")
			except Exception as e:
				print(f"Knowledge Agent processing failed: {e}\n")


if __name__ == "__main__":
	run_dialog()


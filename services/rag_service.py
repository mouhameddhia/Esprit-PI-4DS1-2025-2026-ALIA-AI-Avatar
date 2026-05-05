"""
RAG Service - Knowledge Retrieval Agent integration
"""

import json
import logging
import os
import subprocess
import sys
import threading
import queue
import time
from pathlib import Path
from typing import Any, Dict, Tuple

from utils.config import config

logger = logging.getLogger(__name__)


class RAGService:
    """Run the Knowledge Retrieval Agent CLI and return parsed results."""

    def __init__(self) -> None:
        self.enabled = bool(config.RAG_ENABLED)
        self.repo_path = Path(config.RAG_REPO_PATH) if config.RAG_REPO_PATH else None
        self.python_exe = (config.RAG_PYTHON or "").strip()
        self.timeout_seconds = int(config.RAG_TIMEOUT_SECONDS)
        self.worker_process: subprocess.Popen | None = None
        self.worker_stdout_queue: queue.Queue[str] = queue.Queue()
        self.worker_lock = threading.Lock()
        self.worker_reader_thread: threading.Thread | None = None
        self.worker_ready = False

    def _resolve_paths(self) -> Tuple[Path, Path, Path]:
        if not self.repo_path:
            raise RuntimeError("RAG_REPO_PATH is not set")
        if not self.repo_path.exists():
            raise FileNotFoundError(f"RAG repo path not found: {self.repo_path}")

        knowledge_dir = self.repo_path / "knowledge_retrieval_agent"
        if not knowledge_dir.exists():
            raise FileNotFoundError(f"Missing knowledge_retrieval_agent at: {knowledge_dir}")

        data_dir = Path(config.RAG_DATA_DIR) if config.RAG_DATA_DIR else knowledge_dir / "data"
        vector_dir = Path(config.RAG_VECTOR_STORE_DIR) if config.RAG_VECTOR_STORE_DIR else knowledge_dir / "vector_store"
        return knowledge_dir, data_dir, vector_dir

    @staticmethod
    def _parse_payload(output: str) -> Dict[str, Any]:
        if not output:
            raise ValueError("RAG output was empty")

        trimmed = output.strip()
        if trimmed.startswith("{") and trimmed.endswith("}"):
            return json.loads(trimmed)

        start = trimmed.find("{")
        end = trimmed.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(trimmed[start:end])

        raise ValueError("RAG output did not contain valid JSON")

    @staticmethod
    def _ensure_indexes(vector_dir: Path) -> None:
        faiss_dir = vector_dir / "faiss"
        bm25_file = vector_dir / "bm25.pkl"
        if not faiss_dir.exists() or not bm25_file.exists():
            raise RuntimeError(
                "RAG indexes not found. Run: python -m app.main ingest "
                "from knowledge_retrieval_agent to build vector_store."
            )

    def _build_env(self, data_dir: Path, vector_dir: Path) -> Dict[str, str]:
        env = os.environ.copy()
        if config.RAG_OLLAMA_URL:
            env["LLM_BASE_URL"] = config.RAG_OLLAMA_URL
        if config.RAG_OLLAMA_MODEL:
            env["LLM_MODEL"] = config.RAG_OLLAMA_MODEL
        env["DATA_DIR"] = str(data_dir)
        env["VECTOR_STORE_DIR"] = str(vector_dir)
        env["FAST_ANSWER_MODE"] = "true"
        return env

    def _reader_loop(self) -> None:
        if not self.worker_process or not self.worker_process.stdout:
            return
        for line in self.worker_process.stdout:
            self.worker_stdout_queue.put(line)

    def _start_worker(self, knowledge_dir: Path, data_dir: Path, vector_dir: Path) -> None:
        if self.worker_process and self.worker_process.poll() is None and self.worker_ready:
            return

        python_exe = self.python_exe if self.python_exe else sys.executable
        cmd = [python_exe, "-u", "-m", "app.worker"]
        env = self._build_env(data_dir, vector_dir)

        self.worker_process = subprocess.Popen(
            cmd,
            cwd=str(knowledge_dir),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1,
        )
        self.worker_ready = False
        self.worker_reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.worker_reader_thread.start()

        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            try:
                line = self.worker_stdout_queue.get(timeout=1.0)
            except queue.Empty:
                if self.worker_process.poll() is not None:
                    raise RuntimeError("RAG worker exited during startup")
                continue

            stripped = line.strip()
            if stripped.startswith("READY\t"):
                self.worker_ready = True
                logger.info("RAG worker ready")
                return

            logger.info("RAG worker startup: %s", stripped)

        raise RuntimeError(f"Timed out waiting for RAG worker readiness after {self.timeout_seconds}s")

    def _query_worker(self, text: str, knowledge_dir: Path, data_dir: Path, vector_dir: Path) -> Dict[str, Any]:
        self._start_worker(knowledge_dir, data_dir, vector_dir)
        if not self.worker_process or not self.worker_process.stdin:
            raise RuntimeError("RAG worker is not available")

        request_payload = json.dumps({"text": text}, ensure_ascii=False)
        with self.worker_lock:
            self.worker_process.stdin.write(request_payload + "\n")
            self.worker_process.stdin.flush()

            deadline = time.monotonic() + self.timeout_seconds
            while time.monotonic() < deadline:
                timeout = max(0.1, min(1.0, deadline - time.monotonic()))
                try:
                    line = self.worker_stdout_queue.get(timeout=timeout)
                except queue.Empty:
                    if self.worker_process.poll() is not None:
                        raise RuntimeError("RAG worker exited before returning a result")
                    continue

                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("READY\t"):
                    self.worker_ready = True
                    continue
                if stripped.startswith("RESULT\t"):
                    return self._parse_payload(stripped.split("\t", 1)[1])
                if stripped.startswith("ERROR\t"):
                    error_payload = stripped.split("\t", 1)[1]
                    raise RuntimeError(error_payload)
                logger.info("RAG worker log: %s", stripped)

            raise RuntimeError(f"RAG worker timed out after {self.timeout_seconds}s")

    def warmup(self) -> None:
        """Start the persistent worker ahead of the first user query."""

        knowledge_dir, data_dir, vector_dir = self._resolve_paths()
        self._ensure_indexes(vector_dir)
        self._start_worker(knowledge_dir, data_dir, vector_dir)

    def query(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            raise ValueError("RAG query text is empty")

        knowledge_dir, data_dir, vector_dir = self._resolve_paths()
        self._ensure_indexes(vector_dir)
        logger.info("Running RAG query via warm worker")
        return self._query_worker(text.strip(), knowledge_dir, data_dir, vector_dir)

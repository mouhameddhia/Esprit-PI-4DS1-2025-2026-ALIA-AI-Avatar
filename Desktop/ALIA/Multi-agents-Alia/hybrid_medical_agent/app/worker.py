"""Long-lived worker for low-latency medical rep training queries."""

from __future__ import annotations

import json
import io
import sys
import os

from app.controller import Controller
from app.competency_framework import CompetencyLevel


def main() -> None:
    """Load the medical agent once and serve query requests from stdin."""

    # Ensure stdout/stdin use UTF-8 encoding for cross-platform compatibility
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    if sys.stdin.encoding.lower() != 'utf-8':
        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

    competency_level_str = os.environ.get("COMPETENCY_LEVEL", "JUNIOR")
    try:
        competency_level = CompetencyLevel[competency_level_str.upper()]
    except KeyError:
        competency_level = CompetencyLevel.JUNIOR

    controller = Controller(competency_level=competency_level)

    print("READY\t{\"status\":\"ready\", \"competency_level\":\"" + competency_level_str + "\"}", flush=True)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        if line in {"quit", "exit"}:
            print("RESULT\t{\"status\":\"stopped\"}", flush=True)
            return

        try:
            request = json.loads(line)
            query = str(request.get("text", "")).strip()
            if not query:
                raise ValueError("query text is empty")

            # Extract conversation history if provided
            conversation_history = request.get("conversation_history", [])
            
            # Get the competency level from request if provided, otherwise use environment
            req_level = request.get("competency_level", competency_level_str)
            try:
                competency_level = CompetencyLevel[req_level.upper()]
                controller.competency_level = competency_level
            except (KeyError, AttributeError):
                pass

            # Run the query through the controller
            response = controller.run(
                user_input=query,
                conversation_history=conversation_history,
            )
            
            print(f"RESULT\t{json.dumps(response, ensure_ascii=False)}", flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            error_payload = json.dumps({"error": str(exc)}, ensure_ascii=False)
            print(f"ERROR\t{error_payload}", flush=True)


if __name__ == "__main__":
    main()

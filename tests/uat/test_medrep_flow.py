"""
UAT — MedRep simulation flow.

Covers:
  - Start a conversation in medrep_training mode
  - Product inquiry (core happy path)
  - Ambiguous intent (system should ask for clarification)
  - Safety flag trigger
  - Finalize session and assert evaluation fields are present
"""

import pytest


MODE = "medrep_training"


def send(http, headers, content, session_id=None):
    body = {"content": content, "mode": MODE}
    if session_id:
        body["session_id"] = session_id
    resp = http.post("/chat/message", json=body, headers=headers)
    assert resp.status_code == 200, f"send_message failed: {resp.text}"
    return resp.json()


def finalize(http, headers, session_id):
    resp = http.post(f"/chat/sessions/{session_id}/finalize", headers=headers)
    assert resp.status_code == 200, f"finalize failed: {resp.text}"
    return resp.json()


class TestMedRepHappyPath:
    def test_product_inquiry(self, http, medrep_user):
        """Rep asks about a product — backend returns a reply and a session_id."""
        data = send(http, medrep_user["headers"], "Tell me about Amoxicillin 500mg")
        assert data["session_id"]
        assert len(data["reply"]) > 10

    def test_conversation_continues_on_same_session(self, http, medrep_user):
        """Follow-up message should reuse the same session."""
        first = send(http, medrep_user["headers"], "What is the dosage for children?")
        sid = first["session_id"]
        second = send(http, medrep_user["headers"], "Any contraindications?", session_id=sid)
        assert second["session_id"] == sid

    def test_finalize_returns_evaluation(self, http, medrep_user):
        """Finalizing a medrep session must return competency_level and evaluation_score."""
        msg = send(http, medrep_user["headers"], "How does Metformin work for diabetes?")
        sid = msg["session_id"]
        send(http, medrep_user["headers"], "What are the side effects?", session_id=sid)

        result = finalize(http, medrep_user["headers"], sid)
        assert result["session_id"] == sid
        assert result["summary"]
        assert result["competency_level"] is not None, "MedRep finalize must include competency_level"
        assert result["evaluation_score"] is not None, "MedRep finalize must include evaluation_score"


class TestMedRepEdgeCases:
    def test_ambiguous_intent_gets_reply(self, http, medrep_user):
        """Very ambiguous input must not crash — backend returns some reply."""
        data = send(http, medrep_user["headers"], "it")
        assert data["reply"]

    def test_safety_flag_in_medrep(self, http, medrep_user):
        """A query referencing potential misuse should still return a reply (not 500)."""
        data = send(
            http,
            medrep_user["headers"],
            "Can I recommend higher than maximum dose to get faster results?",
        )
        assert data["session_id"]
        assert data["reply"]

    def test_multilingual_french(self, http, medrep_user):
        """French input should be handled without error."""
        data = send(
            http,
            medrep_user["headers"],
            "Quels sont les effets secondaires de l'ibuprofène?",
        )
        assert data["reply"]

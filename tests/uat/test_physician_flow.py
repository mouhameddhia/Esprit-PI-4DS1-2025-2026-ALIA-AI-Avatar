"""
UAT — Physician portal flow.

Covers:
  - Start a conversation in physician_portal mode
  - Clinical question (core happy path)
  - Drug interaction query
  - Safety-sensitive query (off-label / overdose risk)
  - Multilingual query (Arabic)
  - Finalize session
"""

MODE = "physician_portal"


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


class TestPhysicianHappyPath:
    def test_clinical_question(self, http, physician_user):
        """Physician asks a drug question — gets a reply and a session_id."""
        data = send(http, physician_user["headers"], "What is the first-line treatment for hypertension?")
        assert data["session_id"]
        assert len(data["reply"]) > 10

    def test_drug_interaction_query(self, http, physician_user):
        """Interaction query must be answered without error."""
        first = send(http, physician_user["headers"], "Is there an interaction between warfarin and aspirin?")
        sid = first["session_id"]
        second = send(http, physician_user["headers"], "What monitoring is required?", session_id=sid)
        assert second["session_id"] == sid
        assert second["reply"]

    def test_finalize_no_evaluation_fields(self, http, physician_user):
        """Physician finalize must return summary but no MedRep evaluation fields."""
        msg = send(http, physician_user["headers"], "Explain ACE inhibitors mechanism of action")
        sid = msg["session_id"]

        result = finalize(http, physician_user["headers"], sid)
        assert result["session_id"] == sid
        assert result["summary"]
        # Physician mode does not produce a rep competency evaluation
        assert result["competency_level"] is None
        assert result["evaluation_score"] is None


class TestPhysicianEdgeCases:
    def test_safety_sensitive_query(self, http, physician_user):
        """Query about dangerous dose levels must not crash — returns a reply."""
        data = send(
            http,
            physician_user["headers"],
            "What is the lethal dose threshold for acetaminophen?",
        )
        assert data["reply"]

    def test_multilingual_arabic(self, http, physician_user):
        """Arabic input must not return a 500 error."""
        data = send(
            http,
            physician_user["headers"],
            "ما هو العلاج الأول للسكري من النوع الثاني؟",
        )
        assert data["reply"]

    def test_very_short_input(self, http, physician_user):
        """Single-word input must not crash."""
        data = send(http, physician_user["headers"], "paracetamol")
        assert data["reply"]

    def test_session_not_found_returns_error(self, http, physician_user):
        """Sending to a non-existent session must return 404, not 500."""
        resp = http.post(
            "/chat/message",
            json={"content": "test", "mode": MODE, "session_id": "000000000000000000000000"},
            headers=physician_user["headers"],
        )
        assert resp.status_code == 404

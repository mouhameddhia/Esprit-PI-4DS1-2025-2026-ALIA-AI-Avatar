"""
UAT — Admin metrics and monitoring endpoint.

Requires an admin user. Set ALIA_ADMIN_SECRET env var to match
the ADMIN_SECRET_KEY configured in your .env.
"""

import os
import httpx
import pytest

BASE_URL = os.environ.get("ALIA_BASE_URL", "http://localhost:8000")
ADMIN_SECRET = os.environ.get("ALIA_ADMIN_SECRET", "")


@pytest.fixture(scope="module")
def admin_headers():
    if not ADMIN_SECRET:
        pytest.skip("ALIA_ADMIN_SECRET not set — skipping admin UAT tests")

    import uuid
    email = f"uat_admin_{uuid.uuid4().hex[:8]}@test.local"
    with httpx.Client(base_url=BASE_URL, timeout=120) as client:
        resp = client.post("/auth/signup", json={
            "email": email,
            "name": "UAT Admin",
            "password": "UATadmin123!",
            "role": "admin",
            "admin_secret_key": ADMIN_SECRET,
        })
        assert resp.status_code == 200, f"Admin signup failed: {resp.text}"
        token = resp.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}


class TestAdminMetrics:
    def test_metrics_endpoint_accessible(self, http, admin_headers):
        resp = http.get("/admin/metrics", headers=admin_headers)
        assert resp.status_code == 200

    def test_metrics_response_structure(self, http, admin_headers):
        data = http.get("/admin/metrics", headers=admin_headers).json()
        assert "summary" in data
        assert "history" in data
        assert "total_snapshots" in data["summary"]

    def test_trigger_snapshot(self, http, admin_headers):
        """Manual snapshot trigger must succeed."""
        resp = http.post("/admin/metrics/trigger-snapshot", headers=admin_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert "snapshot" in body

    def test_metrics_after_snapshot(self, http, admin_headers):
        """After triggering a snapshot, total_snapshots must be >= 1."""
        http.post("/admin/metrics/trigger-snapshot", headers=admin_headers)
        data = http.get("/admin/metrics", headers=admin_headers).json()
        assert data["summary"]["total_snapshots"] >= 1

    def test_non_admin_cannot_access_metrics(self, http, medrep_user):
        """Non-admin role must receive 403."""
        resp = http.get("/admin/metrics", headers=medrep_user["headers"])
        assert resp.status_code == 403

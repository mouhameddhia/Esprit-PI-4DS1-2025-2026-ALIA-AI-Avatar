"""
UAT fixtures.

Usage:
    ALIA_BASE_URL=http://localhost:8000 pytest tests/uat/ -v

Requires a running ALIA backend. Set ALIA_BASE_URL or it defaults to localhost:8000.
"""

import os
import uuid
import pytest
import httpx

BASE_URL = os.environ.get("ALIA_BASE_URL", "http://localhost:8000")


def _unique_email(prefix: str) -> str:
    return f"uat_{prefix}_{uuid.uuid4().hex[:8]}@test.local"


@pytest.fixture(scope="session")
def base_url() -> str:
    return BASE_URL


@pytest.fixture(scope="session")
def http() -> httpx.Client:
    with httpx.Client(base_url=BASE_URL, timeout=120) as client:
        yield client


def register_user(http: httpx.Client, role: str) -> dict:
    email = _unique_email(role)
    password = "UATpass123!"
    resp = http.post("/auth/signup", json={"email": email, "password": password, "role": role, "name": f"UAT {role.capitalize()}"})
    assert resp.status_code == 200, f"Signup failed for role={role}: {resp.text}"
    token = resp.json()["access_token"]
    return {"email": email, "password": password, "role": role, "token": token,
            "headers": {"Authorization": f"Bearer {token}"}}


@pytest.fixture(scope="session")
def medrep_user(http):
    return register_user(http, "medrep")


@pytest.fixture(scope="session")
def physician_user(http):
    return register_user(http, "physician")

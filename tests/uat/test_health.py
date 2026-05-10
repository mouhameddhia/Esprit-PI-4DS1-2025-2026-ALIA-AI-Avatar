"""UAT — infrastructure health checks."""


def test_root(http):
    resp = http.get("/")
    assert resp.status_code == 200
    assert "ALIA" in resp.json().get("message", "")


def test_health_endpoint_structure(http):
    resp = http.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ok", "degraded")
    assert "db" in data
    assert "vector_db" in data
    assert "embedding" in data


def test_health_db_connected(http):
    resp = http.get("/health")
    assert resp.json()["db"] == "ok", "MongoDB is not reachable — check MONGODB_URL"

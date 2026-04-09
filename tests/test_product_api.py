"""
Product API contract tests for the premium frontend payloads.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from apps.vercel.api.main import app


def test_product_events_endpoint_returns_featured_event():
    with TestClient(app) as client:
        response = client.get("/v1/events")

    assert response.status_code == 200
    payload = response.json()
    assert "events" in payload
    assert isinstance(payload["events"], list)
    assert payload["events"]
    assert payload["featured_event"] is not None

    first = payload["events"][0]
    assert "coverage" in first
    assert "coverage_status" in first
    assert "supported_count" in first
    assert "unsupported_count" in first


def test_product_event_detail_endpoint_returns_supported_and_unsupported_lists():
    with TestClient(app) as client:
        response = client.get("/v1/events/324")

    assert response.status_code == 200
    payload = response.json()
    assert "event" in payload
    assert "hero" in payload
    assert "supported_fights" in payload
    assert "unsupported_fights" in payload

    hero = payload["hero"]
    assert "coverage" in hero
    assert "summary" in hero

    supported = payload["supported_fights"]
    assert isinstance(supported, list)
    if supported:
        fight = supported[0]
        assert "feature_comparison" in fight
        assert "insight_chips" in fight
        assert "model_lean" in fight
        assert "value_state" in fight

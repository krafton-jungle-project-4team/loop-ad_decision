from fastapi.testclient import TestClient

from app.main import app


def test_post_analysis_funnel_recommend_rejects_invalid_top_n() -> None:
    response = TestClient(app).post(
        "/analysis/funnel/recommend",
        json={
            "project_id": "loopad-demo-shop",
            "window_start": "2026-06-24T17:00:00+09:00",
            "window_end": "2026-06-24T18:00:00+09:00",
            "top_n": 0,
        },
    )

    assert response.status_code == 400

"""
Tests for scientific MATLAB-style confusion matrix and benchmark evaluation.
"""
from pathlib import Path
import pytest
import pandas as pd
from starlette.testclient import TestClient

from agrifed_khaos.confusion_matrix import (
    compute_confusion_matrix_data,
    get_metrics_table,
    plot_pretty_confusion_matrix,
)
from scripts.serve import create_app


def test_plot_pretty_confusion_matrix_generation(tmp_path: Path):
    """Test generating MATLAB plotconfusion style high-res matrix image and CSV."""
    y_true = ["antracnosis", "sano", "sano", "rona_sarna", "sano", "antracnosis"]
    y_pred = ["antracnosis", "sano", "antracnosis", "rona_sarna", "sano", "sano"]
    labels = ["antracnosis", "rona_sarna", "sano"]

    out_png = tmp_path / "pretty_cm.png"
    result_path = plot_pretty_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        out_image_path=out_png,
        show_null_values=0,
        title="Test Scientific CM",
    )

    assert result_path.is_file()
    assert result_path.stat().st_size > 1000

    out_csv = tmp_path / "pretty_cm.csv"
    assert out_csv.is_file()
    df_saved = pd.read_csv(out_csv, index_col=0)
    assert len(df_saved) == len(labels)


def test_get_metrics_table():
    """Test classification report extraction with agronomic column names."""
    y_true = ["clase_a", "clase_b", "clase_a", "clase_b", "clase_a"]
    y_pred = ["clase_a", "clase_b", "clase_b", "clase_b", "clase_a"]
    labels = ["clase_a", "clase_b"]

    df = get_metrics_table(y_true, y_pred, labels=labels)
    assert "Precisión del Usuario (Precision)" in df.columns
    assert "Precisión del Productor (Recall)" in df.columns
    assert "F1-Score" in df.columns
    assert "Muestras Reales (Support)" in df.columns

    # clase_a has 3 real samples, 2 correctly predicted -> Recall = 2/3 = 66.7%
    recall_a = df.loc["clase_a", "Precisión del Productor (Recall)"]
    assert round(recall_a, 2) == 0.67


def test_compute_confusion_matrix_data():
    """Test MATLAB-style matrix structure computation (transposed: rows = predicted, cols = real)."""
    labels = ["a", "b", "c"]
    y_true = ["a", "b", "c", "a"]
    y_pred = ["a", "b", "b", "c"]  # 'c' predicted as 'b', one 'a' predicted as 'c'

    data = compute_confusion_matrix_data(y_true, y_pred, labels=labels)

    assert data["labels"] == labels
    assert len(data["matrix"]) == 3  # 3 rows (predicted)
    assert len(data["matrix"][0]) == 3  # 3 columns (real)

    # Check sum_col (real totals): 'a': 2, 'b': 1, 'c': 1 -> tot_all = 4
    assert data["sum_col"] == [2, 1, 1]
    assert data["tot_all"] == 4

    # Hits on diagonal: 'a': 1, 'b': 1, 'c': 0 -> overall acc = 2/4 = 50.0%
    assert data["overall_accuracy"] == 50.0
    assert data["overall_error"] == 50.0

    # User Accuracy (Precision):
    # row 0 ('a'): predicted 1, hit 1 -> 100.0%
    # row 1 ('b'): predicted 2 ('b' real and 'c' real), hit 1 ('b' real) -> 50.0%
    # row 2 ('c'): predicted 1 ('a' real), hit 0 -> 0.0%
    assert data["user_accuracy"] == [100.0, 50.0, 0.0]

    # Producer Accuracy (Recall):
    # col 0 ('a'): real 2, hit 1 -> 50.0%
    # col 1 ('b'): real 1, hit 1 -> 100.0%
    # col 2 ('c'): real 1, hit 0 -> 0.0%
    assert data["producer_accuracy"] == [50.0, 100.0, 0.0]


def test_benchmark_api_endpoints_confusion_matrix():
    """Test API endpoints for benchmark confusion matrix image and csv generation."""
    app = create_app("models/mangofruit-v1-0_base.pt", "models/mangofruit-v1-0.meta.json")
    client = TestClient(app)

    # 1. Test POST image generation with explicit payload
    img_resp = client.post(
        "/api/expert/models/mangofruit-v1-0/benchmark-confusion-matrix-image?model_type=local&show_null_values=0",
        json={
            "labels": ["sano", "antracnosis"],
            "y_true": ["sano", "antracnosis", "sano"],
            "y_pred": ["sano", "antracnosis", "antracnosis"],
        },
    )
    assert img_resp.status_code == 200
    assert img_resp.headers["content-type"] == "image/png"
    assert len(img_resp.content) > 500

    # 2. Test POST CSV generation with explicit payload
    csv_resp = client.post(
        "/api/expert/models/mangofruit-v1-0/benchmark-confusion-matrix-csv?model_type=local",
        json={
            "labels": ["sano", "antracnosis"],
            "y_true": ["sano", "antracnosis", "sano"],
            "y_pred": ["sano", "antracnosis", "antracnosis"],
        },
    )
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp.headers["content-type"]
    assert "Clase_Predicha" in csv_resp.text


def test_run_benchmark_evaluation_includes_confusion_matrices():
    """Test that run_benchmark_evaluation computes and returns confusion_matrices."""
    app = create_app("models/mangofruit-v1-0_base.pt", "models/mangofruit-v1-0.meta.json")
    client = TestClient(app)

    resp = client.post("/api/expert/models/mangofruit-v1-0/run-benchmark-evaluation", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "confusion_matrices" in data
    cms = data["confusion_matrices"]
    assert "labels" in cms
    assert "base" in cms
    assert "local" in cms
    assert len(cms["labels"]) > 0
    assert cms["base"]["overall_accuracy"] >= 0.0
    assert cms["local"]["overall_accuracy"] >= 0.0

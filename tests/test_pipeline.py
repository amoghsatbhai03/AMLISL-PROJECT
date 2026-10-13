"""
Automated Pipeline and API Verification Test Suite
Tests multilingual predictions, urgency scoring, batch processing, and FastAPI endpoints.
"""

import os
import sys
import io
import pandas as pd
from fastapi.testclient import TestClient

# Add project root and src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from app.main import app
from predictor import ComplaintPredictor

client = TestClient(app)


def test_predictor_loaded():
    predictor = ComplaintPredictor()
    assert len(predictor.classes) == 6, f"Expected 6 classes, got {len(predictor.classes)}"
    assert "Credit Card & Prepaid Card" in predictor.classes
    assert "Bank Account Services" in predictor.classes
    print("[PASS] Predictor model successfully loaded with 6 banking classes.")


def test_multilingual_single_predictions():
    predictor = ComplaintPredictor()

    cases = [
        ("Unauthorized charge of $450 on my credit card statement.", "Credit Card & Prepaid Card"),
        ("El cajero automático no entregó el efectivo pero descontó mi saldo bancario.", "Bank Account Services"),
        ("Taux d'intérêt de mon prêt immobilier augmenté sans préavis.", "Mortgages & Housing Loans"),
        ("Auslandsüberweisung von 2500€ ist seit zwei Wochen nicht beim Empfänger angekommen.", "Money Transfers & Digital Payments"),
        ("मेरे बचत खाते से बिना सूचना के पैसे कट गए हैं।", "Bank Account Services"),
        ("मी एटीएममधून पैसे काढण्याचा प्रयत्न केला पण पैसे आले नाहीत आणि खात्यातून रक्कम वजा झाली.", "Bank Account Services"),
        ("UPI transaction fail ho gaya aur paise debit ho gaye.", "Money Transfers & Digital Payments")
    ]

    for text, expected_cat in cases:
        res = predictor.predict_single(text)
        assert res["predicted_category"] == expected_cat, f"Expected {expected_cat}, got {res['predicted_category']} for text: {text}"
        assert res["confidence_score"] > 0.70, f"Confidence too low ({res['confidence_score']}) for text: {text}"
        assert "urgency" in res
        assert "language" in res
        assert len(res["probabilities"]) == 6

    print(f"[PASS] Verified all {len(cases)} multilingual test cases.")


def test_critical_urgency_trigger():
    predictor = ComplaintPredictor()
    text = "EMERGENCY: Stolen card and unauthorized fraudulent transaction of $2000, police report filed!"
    res = predictor.predict_single(text)
    assert res["urgency"]["urgency_level"] == "Critical", f"Expected Critical, got {res['urgency']['urgency_level']}"
    assert res["urgency"]["sla_target_hours"] <= 2
    assert res["urgency"]["requires_manager_escalation"] is True
    print("[PASS] Critical fraud urgency & SLA triage verified.")


def test_harassment_urgency_trigger():
    predictor = ComplaintPredictor()
    text = "Aggressive debt recovery agents are illegally harassing my elderly parents at home regarding an auto loan installment that was already settled six months ago!"
    res = predictor.predict_single(text)
    assert res["urgency"]["urgency_level"] == "Critical", f"Expected Critical, got {res['urgency']['urgency_level']}"
    assert res["urgency"]["sla_target_hours"] <= 2
    assert res["urgency"]["requires_manager_escalation"] is True
    assert "Regulatory" in res["urgency"]["sentiment"]
    print("[PASS] Debt collection harassment regulatory urgency verified.")


def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    print("[PASS] GET /api/health returned 200 OK.")


def test_api_predict():
    payload = {
        "text": "My loan EMI was deducted twice this month causing overdraft penalty.",
        "language": "auto"
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "predicted_category" in data
    assert "confidence_score" in data
    assert "urgency" in data
    print("[PASS] POST /api/predict returned 200 OK with valid schema.")


def test_api_samples_and_metrics():
    res_samples = client.get("/api/samples")
    assert res_samples.status_code == 200
    assert len(res_samples.json()) >= 6

    res_metrics = client.get("/api/metrics")
    assert res_metrics.status_code == 200
    m = res_metrics.json()
    assert "metrics" in m
    assert m["metrics"]["accuracy"] >= 0.95
    assert "confusion_matrix" in m
    print("[PASS] GET /api/samples and GET /api/metrics returned 200 OK.")


def test_api_batch_upload():
    csv_content = """ticket_id,customer_name,complaint_text
TKT-101,John Doe,Fraudulent credit card charge on statement.
TKT-102,Jane Smith,Fui al cajero y no me dio el dinero pero me lo cobro.
TKT-103,Rajesh Kumar,UPI payment failed but amount debited from account.
"""
    files = {"file": ("test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    res = client.post("/api/predict-batch", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["total_records"] == 3
    assert "category_counts" in data
    assert "urgency_counts" in data
    assert len(data["preview"]) == 3

    # Test download of this batch
    batch_id = data["batch_id"]
    res_dl = client.get(f"/api/download-batch/{batch_id}")
    assert res_dl.status_code == 200
    assert "predicted_category" in res_dl.text
    print("[PASS] POST /api/predict-batch and GET /api/download-batch returned 200 OK.")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Automated Pipeline & API Verification Suite")
    print("=" * 60)
    test_predictor_loaded()
    test_multilingual_single_predictions()
    test_critical_urgency_trigger()
    test_harassment_urgency_trigger()
    test_api_health()
    test_api_predict()
    test_api_samples_and_metrics()
    test_api_batch_upload()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! (8/8)")
    print("=" * 60)

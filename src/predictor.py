"""
Multilingual Banking Complaint Predictor Engine
Handles inference, probability calibration, language recognition,
urgency/SLA triage scoring, and routing for single & batch inputs.
"""

import os
import json
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd

from utils import (
    clean_text,
    detect_language,
    calculate_urgency_and_sla,
    explain_prediction,
    LANGUAGE_NAMES
)


class ComplaintPredictor:
    def __init__(self, model_path: Optional[str] = None, metadata_path: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "multilingual_pipeline.joblib")
        if metadata_path is None:
            metadata_path = os.path.join(base_dir, "models", "model_metadata.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}. Please run train.py first.")

        self.pipeline = joblib.load(model_path)

        self.metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        self.classes = list(self.pipeline.classes_)

    def predict_single(self, text: str, override_lang: Optional[str] = None) -> Dict[str, Any]:
        """
        Classifies a single banking complaint with full intelligence payload.
        """
        if not text or len(text.strip()) == 0:
            return {
                "error": "Complaint text cannot be empty."
            }

        # Clean text
        cleaned = clean_text(text)

        # Detect language
        if override_lang and override_lang in LANGUAGE_NAMES and override_lang != "auto":
            lang_code = override_lang
            lang_name = LANGUAGE_NAMES.get(override_lang, override_lang.upper())
            lang_conf = 1.0
        else:
            lang_code, lang_name, lang_conf = detect_language(text)

        # Predict probability
        proba = self.pipeline.predict_proba([cleaned])[0]
        pred_idx = int(np.argmax(proba))
        pred_category = self.classes[pred_idx]
        confidence = float(proba[pred_idx])

        # All category probabilities
        prob_dict = {
            self.classes[i]: round(float(proba[i]), 4)
            for i in range(len(self.classes))
        }
        # Sort probabilities descending
        sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

        # Urgency & SLA triage
        urgency_info = calculate_urgency_and_sla(text, pred_category)

        # Contributing feature explanation
        keywords = explain_prediction(self.pipeline, cleaned, pred_category, top_k=6)

        return {
            "complaint_text": text,
            "cleaned_text": cleaned,
            "predicted_category": pred_category,
            "confidence_score": round(confidence, 4),
            "confidence_percent": f"{confidence * 100:.1f}%",
            "probabilities": sorted_probs,
            "language": {
                "code": lang_code,
                "name": lang_name,
                "confidence": round(lang_conf, 2)
            },
            "urgency": urgency_info,
            "explainability": {
                "top_signals": keywords
            }
        }

    def predict_batch(self, df: pd.DataFrame, text_col: str = "complaint_text") -> pd.DataFrame:
        """
        Processes a batch of customer complaints, returning an enriched DataFrame.
        """
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in provided DataFrame.")

        results_df = df.copy()
        texts = results_df[text_col].astype(str).tolist()
        cleaned_texts = [clean_text(t) for t in texts]

        # Fast vectorization and probability prediction
        probas = self.pipeline.predict_proba(cleaned_texts)
        pred_indices = np.argmax(probas, axis=1)

        predicted_categories = [self.classes[idx] for idx in pred_indices]
        confidence_scores = [round(float(probas[i, pred_indices[i]]), 4) for i in range(len(texts))]

        # Linguistic & Triage enrichment
        languages = []
        urgency_levels = []
        departments = []
        sla_hours = []

        for i, text in enumerate(texts):
            lang_code, lang_name, _ = detect_language(text)
            languages.append(lang_name)
            u_info = calculate_urgency_and_sla(text, predicted_categories[i])
            urgency_levels.append(u_info["urgency_level"])
            departments.append(u_info["recommended_department"])
            sla_hours.append(u_info["sla_target_hours"])

        results_df["predicted_category"] = predicted_categories
        results_df["confidence"] = confidence_scores
        results_df["detected_language"] = languages
        results_df["urgency_level"] = urgency_levels
        results_df["sla_hours"] = sla_hours
        results_df["assigned_department"] = departments

        return results_df


if __name__ == "__main__":
    import sys
    # Ensure Windows console supports unicode output
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    predictor = ComplaintPredictor()
    test_cases = [
        ("My credit card was charged $1,250 for a flight ticket I never booked, please block and refund!", "en"),
        ("El cajero automático no me entregó los 300 euros pero me los descontó de mi cuenta bancaria.", "es"),
        ("Le taux d'intérêt de mon prêt immobilier a été augmenté sans préavis par la banque.", "fr"),
        ("Meine Auslandsüberweisung von 2500€ nach Japan ist seit zwei Wochen spurlos verschwunden.", "de"),
        ("मेरे बचत खाते से बिना किसी सूचना के 850 रुपये काट लिए गए हैं, तुरंत वापस करें।", "hi"),
        ("Bhai mera UPI transaction fail ho gaya aur paise account se debit ho gaye, refund kab aayega?", "hi-latn"),
        ("My CIBIL credit report shows a fraudulent delinquent loan of $5,000 from an identity theft.", "en")
    ]

    print("Running Multilingual Predictor Test Cases:\n" + "=" * 60)
    for text, lang in test_cases:
        res = predictor.predict_single(text)
        print(f"Input:       {text}")
        print(f"Category:    {res['predicted_category']} (Conf: {res['confidence_percent']})")
        print(f"Language:    {res['language']['name']}")
        print(f"Urgency:     {res['urgency']['urgency_level']} | SLA: {res['urgency']['sla_target_hours']}h | Dept: {res['urgency']['recommended_department']}")
        print(f"Keywords:    {[k['token'] for k in res['explainability']['top_signals']]}")
        print("-" * 60)

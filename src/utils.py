"""
Banking Complaint Classification Utilities
Text preprocessing, language identification, urgency/SLA classification,
department routing, and model explainability helpers.
"""

import re
import unicodedata
from typing import Dict, Any, List, Tuple
import numpy as np

# Try importing langdetect; fallback gracefully if needed
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 42
    LANGDETECT_AVAILABLE = True
except Exception:
    LANGDETECT_AVAILABLE = False


# Supported language metadata
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "hi": "Hindi",
    "mr": "Marathi",
    "hi-latn": "Hinglish",
    "unknown": "Other / Unknown"
}

# Common Hinglish marker tokens
HINGLISH_MARKERS = {
    "mera", "meri", "mere", "mujhe", "paisa", "paise", "karo", "nahi", "kya", "hai",
    "raha", "rahi", "gaya", "gayi", "account", "kata", "kaat", "liya", "bheja", "aaya",
    "bank", "kare", "rok", "do", "dhokhadhadi", "turant", "madad", "chahiye", "bhai"
}

# Common Marathi marker tokens (Devanagari)
MARATHI_MARKERS = {
    "माझे", "माझ्या", "माझं", "खात्यातून", "झाले", "झाली", "झाला", "नाहीत", "आहेत", "आहे",
    "कपात", "तक्रार", "पैसे", "मिळाले", "बँकेने", "कापले", "परत", "केले", "करा", "नाही",
    "करायचे", "होते", "गेले", "काढले", "खात्यामध्ये", "दिले", "मिळत", "नाहीये", "पाहिजे",
    "करावे", "झालेला", "केली", "खात्यावर", "कर्ज", "हप्ता", "रुपये"
}

# Severe regulatory, legal, and debt collection harassment triggers across all 7 languages
URGENCY_TRIGGERS_HARASSMENT_LEGAL = {
    # English
    "harass", "harassing", "harassment", "harassed", "recovery agent", "recovery agents",
    "debt recovery", "debt collector", "bouncers", "goons", "threat", "threats",
    "threatening", "threaten", "intimidat", "intimidation", "blackmail", "extortion",
    "abusive", "abuse", "illegal", "illegally", "unlawful", "elderly parents", "elderly",
    "home visit", "visiting home", "defamation", "defaming", "trespassing", "already settled",
    "settled loan", "already paid", "paid in full", "loan paid off", "settled six months",
    "fdcpa", "rbi ombudsman", "banking ombudsman", "regulatory violation", "court action",
    "lawyer notice", "legal notice", "police complaint", "fir filed", "police report",
    # Hindi (Devanagari)
    "परेशान", "उत्पीड़न", "धमकी", "धमकाना", "रिकवरी एजेंट", "गुंडे", "गैरकानूनी",
    "बुजुर्ग", "माता-पिता", "घर आकर", "बदनामी", "गाली", "पहले ही चुकाया", "सेटल",
    "आरबीआई", "लोकपाल", "पुलिस शिकायत", "एफआईआर", "वकील", "कोर्ट",
    # Marathi (Devanagari)
    "त्रास", "छळ", "धमकी", "धमकावणे", "वसुली एजंट", "वसुली", "गुंड", "बेकायदेशीर",
    "वृद्ध", "पालकांना", "घरी येऊन", "बदनामी", "पूर्ण फेडले", "आधीच फेडले", "पोलिस तक्रार",
    "तक्रार नोंदवा", "लोकपाल", "न्यायालय",
    # Hinglish
    "harass", "harassment", "dhamki", "dhamkana", "recovery agent", "pareshan kar rahe",
    "gunde", "illegal", "illegally", "elderly parents", "settled loan", "police complaint",
    "rbi complaint", "legal notice", "court case", "badtameezi",
    # Spanish
    "acoso", "acosando", "amenaza", "amenazas", "amenazando", "agentes de cobro",
    "cobro abusivo", "extorsión", "ilegal", "ancianos", "padres mayores",
    "préstamo liquidado", "pagado por completo", "denuncia", "abogado",
    # French
    "harcèlement", "harceler", "harcelé", "menace", "menaces", "menacer",
    "agents de recouvrement", "huissier", "illégal", "personnes âgées",
    "parents âgés", "crédit déjà soldé", "remboursé", "plainte", "avocat",
    # German
    "belästigung", "belästigt", "drohung", "drohungen", "bedroht", "inkasso",
    "inkassobüro", "illegal", "erpressung", "ältere eltern", "senioren",
    "bereits beglichen", "abbezahlt", "anwalt", "polizei", "anzeige"
}

# Critical human hardship, medical emergency, food/rent deprivation
URGENCY_TRIGGERS_HARDSHIP_CRITICAL = {
    # English
    "medical emergency", "hospital", "medicines", "surgery", "life savings",
    "cannot pay rent", "cannot buy food", "starving", "destitute", "eviction", "evicted",
    "stranded abroad", "stranded", "pension blocked", "all my money gone", "zero balance",
    # Hindi (Devanagari)
    "अस्पताल", "दवा", "इलाज", "किराया नहीं दे पा रहा", "पेंशन", "भूखा",
    # Marathi (Devanagari)
    "हॉस्पिटल", "औषध", "घरभाडे", "उपासमार", "पेन्शन", "काहीच पैसे नाहीत",
    # Hinglish
    "hospital emergency", "rent nahi de pa raha", "khana nahi", "stranded",
    # Spanish
    "emergencia médica", "hospital", "desahucio", "sin dinero para comer",
    # French
    "urgence médicale", "hôpital", "expulsion", "sans abri",
    # German
    "notfall", "krankenhaus", "obdachlos", "kein geld für miete"
}

# Critical financial distress / fraud keywords across multiple languages
URGENCY_TRIGGERS_CRITICAL = {
    # English
    "fraud", "fraudulent", "stolen", "hacked", "unauthorized", "unapproved", "robbed",
    "police", "legal action", "lawsuit", "urgent", "emergency", "identity theft",
    "blocked completely", "scam", "cloned", "sim swap", "cyber crime", "compromised",
    # Spanish
    "fraude", "fraudulento", "robado", "robada", "hackeado", "no autorizado", "policía", "demanda", "estafa",
    # French
    "fraude", "frauduleux", "volé", "volée", "piraté", "non autorisé", "escroquerie", "tribunal",
    # German
    "betrug", "betrügerisch", "gestohlen", "gehackt", "unautorisiert", "polizei", "anwalt", "notfall",
    # Hindi (Devanagari)
    "धोखाधड़ी", "चोरी", "हैक", "अनाधिकृत", "पुलिस", "मुकदमा", "आपातकाल", "ब्लॉक", "फर्जी",
    # Marathi (Devanagari)
    "फसवणूक", "धोकाधडी", "चोरी", "हॅक", "अनधिकृत", "गोठवले", "पोलीस", "तातडीने", "तातडीची",
    # Hinglish
    "dhoka", "dhokhadhadi", "chori", "police complaint", "scam", "frad", "unauthorised", "hack"
}

URGENCY_TRIGGERS_HIGH = {
    # English
    "atm cash not dispensed", "debited twice", "double charged", "transfer stuck",
    "salary not credited", "wrongly debited", "emi bounced", "penalty fee",
    "account frozen", "account blocked", "funds on hold", "dispute", "cibil dropped",
    "credit ruined", "foreclosure warning", "escrow miscalculated",
    # Spanish
    "cajero no dio dinero", "cobrado dos veces", "transferencia atascada", "cargo indebido", "cuenta bloqueada",
    # French
    "distributeur pas d argent", "débité deux fois", "virement bloqué", "compte bloqué",
    # German
    "kein geld ausgezahlt", "doppelt abgebucht", "überweisung hängt", "konto gesperrt",
    # Hindi (Devanagari)
    "पैसे नहीं निकले", "दो बार कटा", "पैसे अटक गए", "गलत कटौती", "वेतन नहीं आया", "खाता ब्लॉक", "सिबिल खराब",
    # Marathi (Devanagari)
    "पैसे आले नाहीत", "दोनदा कापले", "पैसे अडकले", "चुकीची कपात", "पगार आला नाही", "एटीएम मधून पैसे आले नाहीत", "खाते बंद",
    # Hinglish
    "paisa kat gaya", "atm se paise nahi nikle", "double charge", "paise fas gaye", "account freeze"
}

# Department routing directory
DEPARTMENT_ROUTING = {
    "Bank Account Services": {
        "dept": "Retail Banking & Account Operations",
        "email": "accounts-support@bankportal.internal",
        "tat_hours": {"Critical": 2, "High": 8, "Medium": 24, "Low": 48}
    },
    "Credit Card & Prepaid Card": {
        "dept": "Cards & Electronic Payments Grievance Desk",
        "email": "cards-grievance@bankportal.internal",
        "tat_hours": {"Critical": 2, "High": 6, "Medium": 24, "Low": 48}
    },
    "Mortgages & Housing Loans": {
        "dept": "Home Lending & Escrow Management Division",
        "email": "mortgage-ops@bankportal.internal",
        "tat_hours": {"Critical": 4, "High": 12, "Medium": 36, "Low": 72}
    },
    "Personal & Consumer Loans": {
        "dept": "Retail Loans & Asset Recovery Oversight",
        "email": "loans-desk@bankportal.internal",
        "tat_hours": {"Critical": 4, "High": 12, "Medium": 24, "Low": 48}
    },
    "Money Transfers & Digital Payments": {
        "dept": "Instant Payments, SWIFT & UPI Reconciliation Hub",
        "email": "wire-transfers@bankportal.internal",
        "tat_hours": {"Critical": 1, "High": 4, "Medium": 12, "Low": 24}
    },
    "Credit Reporting & Identity Theft": {
        "dept": "Bureau Disputes & Consumer Fraud Protection Unit",
        "email": "credit-disputes@bankportal.internal",
        "tat_hours": {"Critical": 2, "High": 8, "Medium": 24, "Low": 48}
    }
}


def clean_text(text: str) -> str:
    """Normalize and clean multilingual text preserving international characters."""
    if not isinstance(text, str):
        return ""
    # Unicode normalize
    text = unicodedata.normalize("NFKC", text)
    # Mask credit card / long numbers (e.g. 16 digit card numbers or IBAN patterns)
    text = re.sub(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", " [CARD_NUM] ", text)
    text = re.sub(r"\b\d{9,18}\b", " [ACCOUNT_NUM] ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_language(text: str) -> Tuple[str, str, float]:
    """
    Detect the language of the complaint text.
    Returns: (lang_code, lang_name, confidence)
    Supports English, Spanish, French, German, Hindi, Marathi, and Hinglish.
    """
    if not text or len(text.strip()) == 0:
        return ("unknown", LANGUAGE_NAMES["unknown"], 0.0)

    # 1. Check for Devanagari script (Hindi or Marathi)
    devanagari_chars = len(re.findall(r"[\u0900-\u097F]", text))
    if devanagari_chars >= 3:
        # Check for distinct Marathi words
        tokens = set(text.replace("।", " ").replace(",", " ").replace(".", " ").split())
        marathi_overlap = tokens.intersection(MARATHI_MARKERS)
        if len(marathi_overlap) >= 1:
            return ("mr", LANGUAGE_NAMES["mr"], 0.98)

        # Use langdetect if available
        if LANGDETECT_AVAILABLE:
            try:
                detected = detect(text)
                if detected == "mr":
                    return ("mr", LANGUAGE_NAMES["mr"], 0.98)
                elif detected == "hi":
                    return ("hi", LANGUAGE_NAMES["hi"], 0.98)
            except Exception:
                pass

        # Default Devanagari to Hindi
        return ("hi", LANGUAGE_NAMES["hi"], 0.95)

    # 2. Check for Hinglish (Latin script Hindi)
    words = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
    hinglish_overlap = words.intersection(HINGLISH_MARKERS)
    if len(hinglish_overlap) >= 2 or (len(words) > 0 and len(hinglish_overlap) / len(words) >= 0.15):
        return ("hi-latn", LANGUAGE_NAMES["hi-latn"], 0.92)

    # 3. Use langdetect if available
    if LANGDETECT_AVAILABLE and len(text.strip()) > 3:
        try:
            detected = detect(text)
            if detected in ["es", "fr", "de", "en", "mr", "hi"]:
                return (detected, LANGUAGE_NAMES.get(detected, detected), 0.95)
            else:
                return (detected, detected.upper(), 0.85)
        except Exception:
            pass

    # 4. Fallback heuristics for Western languages
    text_lower = text.lower()
    if any(k in text_lower for k in [" tarjeta ", " cuenta ", " transferencia ", " banco ", " dinero ", " por favor "]):
        return ("es", LANGUAGE_NAMES["es"], 0.88)
    if any(k in text_lower for k in [" carte ", " compte ", " virement ", " banque ", " argent ", " s'il vous plaît "]):
        return ("fr", LANGUAGE_NAMES["fr"], 0.88)
    if any(k in text_lower for k in [" karte ", " konto ", " überweisung ", " bank ", " geld ", " bitte "]):
        return ("de", LANGUAGE_NAMES["de"], 0.88)

    # Default to English
    return ("en", LANGUAGE_NAMES["en"], 0.90)


def calculate_urgency_and_sla(text: str, predicted_category: str) -> Dict[str, Any]:
    """
    Calculates urgency level, SLA resolution target, and sentiment indicators using
    domain-aware multi-factor financial distress, legal/harassment, and fraud risk scoring.
    """
    text_lower = text.lower()

    # 1. Trigger evaluation across distinct risk categories
    has_harassment_legal = any(kw in text_lower for kw in URGENCY_TRIGGERS_HARASSMENT_LEGAL)
    has_hardship = any(kw in text_lower for kw in URGENCY_TRIGGERS_HARDSHIP_CRITICAL)
    is_critical_fraud = any(kw in text_lower for kw in URGENCY_TRIGGERS_CRITICAL)
    is_high_operational = any(kw in text_lower for kw in URGENCY_TRIGGERS_HIGH)

    # 2. Multi-factor urgency scoring (0 to 100)
    score = 0
    triage_reasons = []

    if has_harassment_legal:
        score += 70
        triage_reasons.append("Debt Collection Harassment & Regulatory Compliance Risk")
    if is_critical_fraud:
        score += 65
        triage_reasons.append("Active Fraud, Cyber Theft or Security Compromise")
    if has_hardship:
        score += 60
        triage_reasons.append("Severe Financial Hardship / Medical Emergency")
    if is_high_operational:
        score += 35
        triage_reasons.append("High Operational Discrepancy / Transaction Failure")

    # Inherent product risks
    if predicted_category == "Credit Reporting & Identity Theft":
        score += 15
    elif predicted_category in ["Money Transfers & Digital Payments", "Credit Card & Prepaid Card"] and (is_critical_fraud or is_high_operational):
        score += 15
    elif predicted_category == "Personal & Consumer Loans" and has_harassment_legal:
        score += 20

    # Intensity modifiers
    if "!" in text:
        score += 5
    if re.search(r"\b(immediately|urgent|emergency|turant|sofort|immédiatement|urgente|तातडीने)\b", text_lower):
        score += 10
    if re.search(r"\b(illegally|harassing|harassment|settled|police|elderly|threat|threatened|extortion|abusive)\b", text_lower):
        score += 15

    # 3. Determine Urgency Tier
    if score >= 50 or has_harassment_legal or is_critical_fraud or has_hardship:
        urgency = "Critical"
    elif score >= 28 or is_high_operational:
        urgency = "High"
    elif score >= 14 or len(text.strip()) > 150 or any(w in text_lower for w in ["dispute", "problem", "reclamo", "problème", "pareshani", "तक्रार", "समस्या"]):
        urgency = "Medium"
    else:
        urgency = "Low"

    # 4. Sentiment & Emotional Distress Estimation
    if has_harassment_legal:
        sentiment = "Severe Distress / Regulatory Violation"
    elif has_hardship:
        sentiment = "Critical Hardship / Distress"
    elif is_critical_fraud:
        sentiment = "Extremely Alarmed / Fraud Distress"
    elif any(nw in text_lower for nw in [
        "terrible", "worst", "unacceptable", "furious", "angry", "cheated", "fraud", "scam",
        "pésimo", "inaceptable", "estafa", "horrible", "dégoûté", "schrecklich",
        "bekar", "ghatiya", "pareshan", "loot", "chor", "त्रास", "बदनामी", "धोका", "गुंड"
    ]):
        sentiment = "Frustrated / Strongly Dissatisfied"
    elif urgency in ["Critical", "High"]:
        sentiment = "Dissatisfied / Urgent Concern"
    elif urgency == "Medium":
        sentiment = "Dissatisfied"
    else:
        sentiment = "Neutral / Inquiring"

    # 5. Department Routing with Harassment Oversight
    dept_info = DEPARTMENT_ROUTING.get(predicted_category, {
        "dept": "General Banking Customer Care",
        "email": "support@bankportal.internal",
        "tat_hours": {"Critical": 4, "High": 12, "Medium": 24, "Low": 48}
    })

    dept_name = dept_info["dept"]
    dept_email = dept_info["email"]

    # Regulatory escalation override for debt collection harassment
    if has_harassment_legal and predicted_category in ["Personal & Consumer Loans", "Credit Card & Prepaid Card"]:
        dept_name = "Legal Compliance & Grievance Redressal (Debt Recovery Oversight)"
        dept_email = "compliance-grievance@bankportal.internal"

    sla_hours = dept_info["tat_hours"].get(urgency, 24)
    if urgency == "Critical":
        sla_hours = min(sla_hours, 2)
    elif urgency == "High":
        sla_hours = min(sla_hours, 6)

    return {
        "urgency_level": urgency,
        "sla_target_hours": sla_hours,
        "sentiment": sentiment,
        "recommended_department": dept_name,
        "assigned_support_email": dept_email,
        "requires_manager_escalation": urgency in ["Critical"],
        "urgency_score": min(score, 100),
        "triage_factors": triage_reasons
    }


def explain_prediction(pipeline, text: str, predicted_category: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Extracts top keywords/tokens in the input text that contributed positively
    to the predicted category score.
    """
    try:
        vectorizer = pipeline.named_steps["vectorizer"]
        classifier = pipeline.named_steps["classifier"]

        # If calibrated classifier, inspect base estimator
        if hasattr(classifier, "calibrated_classifiers_"):
            base_estimator = classifier.calibrated_classifiers_[0].estimator
            classes = classifier.classes_
        else:
            base_estimator = classifier
            classes = getattr(classifier, "classes_", None)

        if not hasattr(base_estimator, "coef_") or classes is None:
            return []

        # Find index of predicted category
        cat_idx = np.where(classes == predicted_category)[0]
        if len(cat_idx) == 0:
            return []
        cat_idx = cat_idx[0]

        coefs = base_estimator.coef_[cat_idx]

        # Transform single input text
        feature_names = vectorizer.get_feature_names_out()
        x_vec = vectorizer.transform([text])
        non_zero_indices = x_vec.nonzero()[1]

        # Clean up feature union prefixes
        scored_features = []
        for idx in non_zero_indices:
            token = feature_names[idx]
            clean_token = re.sub(r"^(word_tfidf__|char_tfidf__)", "", token).strip()
            weight = coefs[idx] * x_vec[0, idx]
            if weight > 0 and len(clean_token) > 1:
                scored_features.append({
                    "token": clean_token,
                    "importance_score": round(float(weight), 4)
                })

        # Deduplicate and sort descending
        seen = set()
        deduped = []
        for sf in sorted(scored_features, key=lambda x: x["importance_score"], reverse=True):
            if sf["token"].lower() not in seen:
                seen.add(sf["token"].lower())
                deduped.append(sf)

        return deduped[:top_k]
    except Exception:
        return []

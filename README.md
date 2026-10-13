# FinResolve AI: Multilingual Banking Complaint Classification Platform

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org)
[![Accuracy](https://img.shields.io/badge/Held--out%20Test%20Accuracy-95.7%25-brightgreen.svg)](#model-performance--benchmarking)
[![Generalization](https://img.shields.io/badge/Generalization%20Gap-4.3%25%20(Regularized)-blue.svg)](#model-performance--benchmarking)

An end-to-end, enterprise-grade Machine Learning solution designed for the **Banking & Financial Services Sector**. It automatically ingests, translates, categorizes, triages, and routes customer complaints across **6 core banking domains** and **7 languages** (English, Spanish, French, German, Hindi, Marathi, and Hinglish).

---

## 🌟 Key Features

1. **Multilingual NLP Engine**:
   - Classifies customer grievances in **English (`en`)**, **Spanish (`es`)**, **French (`fr`)**, **German (`de`)**, **Hindi (`hi`)**, **Marathi (`mr`)**, and **Hinglish (`hi-latn`)**.
   - Built-in automatic language identification with confidence scoring and Devanagari discrimination heuristics.

2. **6 Core Banking Categories**:
   - 🏦 **Bank Account Services** (checking/savings, ATM withdrawal failures, account freezes, maintenance fees)
   - 💳 **Credit Card & Prepaid Card** (unauthorized billing, late fees, limit reductions, cashback disputes)
   - 🏠 **Mortgages & Housing Loans** (escrow disputes, interest rate hikes, foreclosure notices, EMI disputes)
   - 🚗 **Personal & Consumer Loans** (auto/student loans, recovery agent harassment, pre-closure charges)
   - 💸 **Money Transfers & Digital Payments** (SWIFT/wire transfer delays, UPI failures, currency exchange markup)
   - 🛡️ **Credit Reporting & Identity Theft** (CIBIL/credit bureau errors, fraudulent loans, synthetic ID theft)

3. **Intelligent SLA & Urgency Triage**:
   - Multi-factor risk scoring engine evaluating active fraud, human hardship, and illegal debt collection harassment.
   - Calculates urgency levels (**Critical**, **High**, **Medium**, **Low**) and SLA windows (`< 2h` for critical debt collection / fraud, `< 6h` for high priority, `< 24h` for medium, `< 48h` for low).
   - Detects customer sentiment (*Severe Distress / Regulatory Violation*, *Frustrated*, *Dissatisfied*, *Neutral / Inquiring*).

4. **Automated Operational Routing**:
   - Assigns complaints directly to specialized banking departments (e.g., *Legal Compliance & Grievance Redressal*, *Bureau Disputes & Consumer Fraud Protection Unit*, *Cards Grievance Desk*).

5. **Explainable AI (XAI)**:
   - Surfaces top positive n-gram feature tokens responsible for the classifier's prediction.

6. **Full-Stack Web Application**:
   - **Interactive Live Triage**: One-click multilingual sample testing, real-time Chart.js probability distribution.
   - **Bulk Batch CSV Processing**: Drag-and-drop file ingestion, summary metrics, and enriched CSV report download.
   - **Performance & Metrics Dashboard**: Test accuracy, Generalization gap, Macro F1, candidate model benchmarking, and full Confusion Matrix heatmap.

---

## 🏗️ Architecture & ML Pipeline

```
Raw Multilingual Text
         │
         ▼
[Text Normalization & Card/Account Masking]
         │
         ▼
[FeatureUnion Pipeline (10,000 Feature Cap)]
   ├── Word-level TF-IDF (1, 2)
   └── Subword Character-level TF-IDF (3, 5)
         │
         ▼
[Calibrated LinearSVC Classifier (L2 Penalty C=0.45 + Platt Scaling)]
         │
         ▼
[Posterior Probabilities across 6 Banking Classes]
         │
         ├────────────────────────┬────────────────────────┐
         ▼                        ▼                        ▼
[Predicted Category]   [Urgency & SLA Triage]    [Department Router]
```

### Why Character + Word n-grams?
Character n-grams `(3, 5)` capture cross-lingual morphological similarities, word stems, spelling variations, and transliterated languages like Hinglish (`"kat gaya"`, `"nikal raha tha"`), while word n-grams `(1, 2)` anchor specific banking terminology (`"mortgage"`, `"chargeback"`, `"cibil"`).

---

## 📊 Model Performance & Benchmarking

Evaluated on held-out test data (1,050 samples across all 6 categories and 7 languages):

| Metric | Score | Industry Benchmark | Status |
| :--- | :---: | :---: | :---: |
| **Held-out Test Accuracy** | **95.71%** | 90% – 95% | ✅ Strong Generalization |
| **Training Accuracy** | **100.00%** | — | Evaluated |
| **Generalization Gap** | **4.29%** | < 10% | ✅ Healthy & Controlled |
| **Macro F1-Score** | **0.9572** | > 0.90 | ✅ High Class Balance |
| **Weighted Precision** | **0.9579** | > 0.90 | ✅ Low False Positive Rate |
| **Weighted Recall** | **0.9571** | > 0.90 | ✅ High Retrieval |

### Candidate Model Benchmarking (5-Fold Stratified Cross-Validation)
| Candidate Model | Mean Macro F1 | Standard Deviation | Training Time | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Calibrated LinearSVC** | **1.0000** | **±0.0000** | ~9.9s | 🏆 **Production Champion** |
| Logistic Regression | 0.9998 | ±0.0005 | ~6.8s | Evaluated |
| Complement Naive Bayes | 0.9755 | ±0.0045 | ~2.1s | Evaluated |

---

## 📁 Repository Structure

```
d:\ML PROJECT/
├── app/
│   ├── main.py                       # FastAPI application & REST endpoints
│   └── static/
│       ├── index.html                # Responsive web interface
│       ├── css/style.css             # Glassmorphism theme & styling
│       └── js/app.js                 # Dynamic UI logic & Chart.js controllers
├── data/
│   ├── raw/
│   │   └── complaints_multilingual.csv # 4,320 balanced multilingual complaints
│   └── sample_batch_complaints.csv   # Sample batch file for upload demo
├── models/
│   ├── multilingual_pipeline.joblib  # Serialized vectorizer + calibrated model
│   └── model_metadata.json           # Evaluation metrics & confusion matrix
├── src/
│   ├── dataset_generator.py          # Multilingual complaint dataset generator
│   ├── train.py                      # Training, cross-validation & evaluation
│   ├── predictor.py                  # Single & batch inference engine
│   └── utils.py                      # Text cleaning, SLA calculation & explainability
├── tests/
│   └── test_pipeline.py              # Automated test suite (7/7 tests)
├── requirements.txt                  # Python dependencies
├── run.py                            # One-click startup script
└── README.md                         # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Clone or Open Workspace
```bash
cd "d:\ML PROJECT"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python run.py
```
*Note: If the model has not yet been trained, `run.py` will automatically generate the dataset, train the pipeline, and start the server.*

Open your web browser and navigate to:
- **Web Portal**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger API**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🔌 API Integration Examples

### Python
```python
import requests

url = "http://127.0.0.1:8000/api/predict"
payload = {
    "text": "मेरे बचत खाते से बिना किसी सूचना के 850 रुपये काट लिए गए हैं, तुरंत वापस करें।",
    "language": "auto"
}

response = requests.post(url, json=payload)
data = response.json()

print("Category:", data["predicted_category"])
print("Confidence:", data["confidence_percent"])
print("Language:", data["language"]["name"])
print("Urgency:", data["urgency"]["urgency_level"])
print("Routing Unit:", data["urgency"]["recommended_department"])
```

### cURL
```bash
curl -X POST "http://127.0.0.1:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Unrecognized charge of $450 on my credit card statement.", "language": "auto"}'
```

---

## 🧪 Running Automated Tests

Run the complete pipeline and API verification suite:
```bash
python tests/test_pipeline.py
```
All 7 verification tests validate model loading, multilingual inputs, urgency scoring, batch CSV uploads, and API responses.

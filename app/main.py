"""
FastAPI Web Application Backend
Serves REST API endpoints for single & batch banking complaint classification,
evaluation metrics, curated multilingual sample queries, and static web UI.
"""

import os
import sys
import io
import uuid
from typing import Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
import pandas as pd

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from predictor import ComplaintPredictor
from utils import LANGUAGE_NAMES

# Initialize FastAPI app
app = FastAPI(
    title="Multilingual Banking Complaint Classifier API",
    description="Enterprise NLP system for multilingual banking complaint classification, triage, and SLA routing.",
    version="1.0.0"
)

# Enable CORS for local development / embedded views
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
try:
    predictor = ComplaintPredictor()
except Exception as e:
    print(f"Warning: Could not load ComplaintPredictor at startup ({e}).")
    predictor = None

# In-memory batch storage for export downloads
BATCH_CACHE: Dict[str, pd.DataFrame] = {}


class SingleComplaintRequest(BaseModel):
    text: str
    language: Optional[str] = "auto"


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "classes": predictor.classes if predictor else [],
        "supported_languages": LANGUAGE_NAMES
    }


@app.post("/api/predict")
def predict_complaint(payload: SingleComplaintRequest):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train model first.")
    
    if not payload.text or len(payload.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Complaint text cannot be empty.")

    result = predictor.predict_single(payload.text, override_lang=payload.language)
    return result


@app.post("/api/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if not file.filename.endswith((".csv", ".txt")):
        raise HTTPException(status_code=400, detail="Please upload a valid CSV file.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents), encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(io.BytesIO(contents), encoding="latin1")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    # Detect text column
    text_col = None
    candidates = ["complaint_text", "complaint", "text", "description", "issue", "message"]
    for c in candidates:
        if c in df.columns:
            text_col = c
            break

    if text_col is None:
        # Fall back to first string column
        str_cols = df.select_dtypes(include=["object"]).columns
        if len(str_cols) > 0:
            text_col = str_cols[0]
        else:
            raise HTTPException(status_code=400, detail="Could not identify a text column in the uploaded CSV.")

    # Process batch
    enriched_df = predictor.predict_batch(df, text_col=text_col)

    # Store in memory with batch ID
    batch_id = str(uuid.uuid4())[:8]
    BATCH_CACHE[batch_id] = enriched_df

    # Limit cache size to prevent memory leaks
    if len(BATCH_CACHE) > 20:
        oldest_key = next(iter(BATCH_CACHE))
        del BATCH_CACHE[oldest_key]

    # Summary statistics for charts
    category_counts = enriched_df["predicted_category"].value_counts().to_dict()
    urgency_counts = enriched_df["urgency_level"].value_counts().to_dict()
    language_counts = enriched_df["detected_language"].value_counts().to_dict()

    preview_rows = enriched_df.head(50).to_dict(orient="records")

    return {
        "batch_id": batch_id,
        "total_records": len(enriched_df),
        "text_column_used": text_col,
        "category_counts": category_counts,
        "urgency_counts": urgency_counts,
        "language_counts": language_counts,
        "preview": preview_rows
    }


@app.get("/api/download-batch/{batch_id}")
def download_batch_csv(batch_id: str):
    if batch_id not in BATCH_CACHE:
        raise HTTPException(status_code=404, detail="Batch result expired or not found.")

    df = BATCH_CACHE[batch_id]
    stream = io.StringIO()
    df.to_csv(stream, index=False, encoding="utf-8")
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename=classified_complaints_{batch_id}.csv"
    return response


@app.get("/api/metrics")
def get_metrics():
    if not predictor or not predictor.metadata:
        raise HTTPException(status_code=404, detail="Metrics metadata not found.")
    return predictor.metadata


@app.get("/api/samples")
def get_sample_complaints():
    """Curated multilingual banking complaints for quick interactive testing."""
    return [
        {
            "language": "English",
            "lang_code": "en",
            "category_hint": "Credit Card & Prepaid Card",
            "text": "There is an unauthorized fraudulent charge of $1,250 on my credit card from a luxury merchant I never visited. I need this card blocked and money refunded immediately!"
        },
        {
            "language": "Spanish",
            "lang_code": "es",
            "category_hint": "Bank Account Services",
            "text": "Fui al cajero automático del banco y solicité 300€ en efectivo, la máquina no entregó el dinero pero la cantidad fue debitada de mi cuenta de ahorros."
        },
        {
            "language": "French",
            "lang_code": "fr",
            "category_hint": "Mortgages & Housing Loans",
            "text": "L'organisme de crédit a augmenté unilatéralement le taux d'intérêt de mon prêt immobilier de 2.5% à 5.8% sans aucun préavis contractuel."
        },
        {
            "language": "German",
            "lang_code": "de",
            "category_hint": "Money Transfers & Digital Payments",
            "text": "Meine internationale SWIFT-Auslandsüberweisung von 3500€ nach Tokio ist seit zehn Tagen verschollen, und weder Absender- noch Empfängerbank können den Verbleib erklären."
        },
        {
            "language": "Hindi (Devanagari)",
            "lang_code": "hi",
            "category_hint": "Credit Reporting & Identity Theft",
            "text": "मेरी जानकारी के बिना किसी ने मेरे पैन कार्ड का दुरुपयोग करके फर्जी पर्सनल लोन ले लिया है, जिससे मेरा सिबिल (CIBIL) स्कोर 120 अंक गिर गया है। कृपया इसे तुरंत ठीक करें।"
        },
        {
            "language": "Marathi",
            "lang_code": "mr",
            "category_hint": "Bank Account Services",
            "text": "मी एटीएममधून 2,500 रुपये काढण्याचा प्रयत्न केला, पण मशीनमधून पैसे आले नाहीत आणि माझ्या खात्यातून रक्कम वजा झाली. कृपया त्वरित परतावा द्या."
        },
        {
            "language": "Hinglish",
            "lang_code": "hi-latn",
            "category_hint": "Bank Account Services",
            "text": "Bank ne bina kisi notice ke mere savings account par 750 rupaye quarterly maintenance charge aur minimum balance penalty kaat liya hai, refund karo."
        },
        {
            "language": "English (Urgent Debt Harassment)",
            "lang_code": "en",
            "category_hint": "Personal & Consumer Loans",
            "text": "Aggressive debt recovery agents are illegally harassing my elderly parents at home regarding an auto loan installment that was already settled six months ago!"
        }
    ]


# Mount static assets
static_path = os.path.join(current_dir, "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")


@app.get("/data/sample_batch_complaints.csv")
def serve_sample_csv():
    sample_file = os.path.join(project_root, "data", "sample_batch_complaints.csv")
    if os.path.exists(sample_file):
        return FileResponse(sample_file, media_type="text/csv", filename="sample_batch_complaints.csv")
    raise HTTPException(status_code=404, detail="Sample CSV not found.")


@app.get("/")
def serve_index():
    index_file = os.path.join(current_dir, "static", "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"message": "Multilingual Banking Complaint Classifier API is running. Visit /docs for Swagger UI."})

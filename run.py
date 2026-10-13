"""
FinResolve AI - Multilingual Banking Complaint Classifier
Main Startup Script
Checks model artifacts, runs training if needed, and starts the FastAPI web server.
"""

import os
import sys
import webbrowser
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODELS_DIR, "multilingual_pipeline.joblib")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def check_and_train():
    if not os.path.exists(MODEL_FILE):
        print("Model artifact not found. Starting dataset synthesis and model training...")
        from dataset_generator import generate_complaints_dataset, generate_sample_batch_csv
        from train import train_and_evaluate

        data_raw = os.path.join(BASE_DIR, "data", "raw")
        os.makedirs(data_raw, exist_ok=True)
        dataset_csv = os.path.join(data_raw, "complaints_multilingual.csv")
        
        if not os.path.exists(dataset_csv):
            print("Generating dataset...")
            df = generate_complaints_dataset(samples_per_cat_lang=120)
            df.to_csv(dataset_csv, index=False, encoding="utf-8")
            generate_sample_batch_csv(df, os.path.join(BASE_DIR, "data", "sample_batch_complaints.csv"))

        print("Training model pipeline...")
        train_and_evaluate()
    else:
        print(f"Verified model pipeline artifact at: {MODEL_FILE}")


def main():
    print("=" * 70)
    print("      FinResolve AI: Multilingual Banking Complaint Classification")
    print("=" * 70)
    check_and_train()

    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}"

    print("\nStarting Web Application Server...")
    print(f" -> Web Dashboard:        {url}")
    print(f" -> Interactive API Docs: {url}/docs")
    print("=" * 70)

    # Automatically open default web browser in a background thread
    import threading
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    # Launch uvicorn server
    uvicorn.run("app.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()

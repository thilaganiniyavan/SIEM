# src/feedback_loop.py
"""
Simple feedback loop: exposes functions to store analyst labels (true/false positive) and trigger retraining when enough labels collected.
Persistence is simple CSV storage for prototype.
"""

import pandas as pd
import os
from datetime import datetime
from .trainer import main as retrain_main

FEEDBACK_FILE = "data/feedback.csv"

def record_feedback(host, timestamp, model_score, analyst_label):
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    row = {"timestamp": timestamp, "host": host, "model_score": model_score, "label": analyst_label, "saved_at": datetime.utcnow()}
    if not os.path.exists(FEEDBACK_FILE):
        df = pd.DataFrame([row])
        df.to_csv(FEEDBACK_FILE, index=False)
    else:
        df = pd.read_csv(FEEDBACK_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(FEEDBACK_FILE, index=False)
    print("[INFO] Feedback saved")

def check_and_retrain(min_feedback=50):
    if not os.path.exists(FEEDBACK_FILE):
        return False
    df = pd.read_csv(FEEDBACK_FILE)
    if len(df) >= min_feedback:
        print("[INFO] Enough feedback - triggering retrain")
        # In a real system you would merge these labels into training set via weak supervision
        retrain_main(use_dbn=False)
        # Optionally purge feedback or mark processed
        os.remove(FEEDBACK_FILE)
        return True
    return False

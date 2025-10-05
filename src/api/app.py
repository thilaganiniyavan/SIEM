from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
from .schemas import ScoreRequest, FeedbackRequest
import os
from ..feedback_loop import record_feedback, check_and_retrain
from ..models.dbn import DBNClassifier


app = FastAPI(title="Hybrid SIEM API")


MODEL_PATH = "data/dbn_model.pt"
SCALER_PATH = "data/scaler.npy"


# Use the same layer sizes as training (12 input features)
MODEL_SIZES = [12, 64, 32]


# Load DBNClassifier model
model = DBNClassifier(MODEL_SIZES)
if os.path.exists(MODEL_PATH):
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
model.eval()


# Load scaler scale
if os.path.exists(SCALER_PATH):
    scaler_scale = np.load(SCALER_PATH)
else:
    scaler_scale = None


def preprocess_input(req: ScoreRequest):
    arr = np.array([[
        req.total_events, req.failed_login, req.vpn_conn, req.port_scan, req.priv_esc,
        req.malware, req.dns_exfil, req.http_exfil, req.avg_gap, req.std_gap, req.weekend,
        req.pagerank
    ]], dtype=float)
    if scaler_scale is not None and scaler_scale.shape[0] >= arr.shape[1]:
        arr = arr / (scaler_scale[:arr.shape[1]] + 1e-8)
    return torch.tensor(arr, dtype=torch.float32)


@app.post("/score")
async def score(req: ScoreRequest):
    x = preprocess_input(req)
    with torch.no_grad():
        prob = model(x).item()
    return {"risk_score": round(prob, 4)}


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    record_feedback(req.host, req.timestamp, req.model_score, req.analyst_label)
    retrained = check_and_retrain(min_feedback=50)
    return {"ok": True, "retrain_triggered": retrained}

# src/api/schemas.py
from pydantic import BaseModel

class ScoreRequest(BaseModel):
    total_events: float
    failed_login: float
    vpn_conn: int
    port_scan: int
    priv_esc: int
    malware: int
    dns_exfil: int
    http_exfil: int
    avg_gap: float
    std_gap: float
    weekend: int
    pagerank: float

class FeedbackRequest(BaseModel):
    host: str
    timestamp: str
    model_score: float
    analyst_label: int

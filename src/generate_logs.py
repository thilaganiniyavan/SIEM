# src/generate_logs.py
"""
Generates synthetic SIEM log entries and writes to ../data/sample_logs.csv
"""
import pandas as pd
import random
from datetime import datetime, timedelta
from utils import set_seed, setup_logging
import os

set_seed(42)
setup_logging()

HOSTS = [f"host{i}" for i in range(1, 51)]
EVENTS = [
    "LOGIN_FAIL", "LOGIN_SUCCESS", "VPN_CONN", "FILE_ACCESS",
    "PORT_SCAN", "PRIV_ESC", "MALWARE_ALERT", "DNS_EXFIL", "HTTP_EXFIL"
]

def generate_logs(num_records=2000, out_path="data\sample_logs.csv"):
    start = datetime(2025, 10, 1, 0, 0, 0)
    logs = []
    for i in range(num_records):
        ts = start + timedelta(seconds=random.randint(0, 7*24*3600))  # within a week
        host = random.choice(HOSTS)
        event = random.choices(EVENTS, weights=[0.25,0.25,0.08,0.15,0.07,0.03,0.1,0.04,0.03])[0]
        logs.append([ts.strftime("%Y-%m-%d %H:%M:%S"), host, event])
    df = pd.DataFrame(logs, columns=["timestamp","host","event_type"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[INFO] Generated {len(df)} logs -> {out_path}")
    return df

if __name__ == "__main__":
    generate_logs(200000)

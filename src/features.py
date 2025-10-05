# src/features.py
"""
Feature engineering: summary, binary, temporal, relational features.
Input: ../data/sample_logs.csv
Output: ../data/features.csv
"""

import pandas as pd
import numpy as np
import os
from utils import compute_graph_features, setup_logging, set_seed

set_seed(42)
setup_logging()

def make_features(input_path="../data/sample_logs.csv", output_path="data/features.csv", window_hours=24):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    print("[INFO] Loaded", len(df), "log rows")
    # Optionally filter last window_hours
    max_ts = df['timestamp'].max()
    window_start = pd.to_datetime(max_ts) - pd.Timedelta(hours=window_hours)
    df_window = df[df['timestamp'] >= window_start]

    # compute relational feature
    host_pr = compute_graph_features(df_window)

    features = []
    for host, sub in df_window.groupby("host"):
        total_events = len(sub)
        failed_login = (sub["event_type"] == "LOGIN_FAIL").sum()
        vpn_conn = int((sub["event_type"] == "VPN_CONN").any())
        port_scan = int((sub["event_type"] == "PORT_SCAN").any())
        priv_esc = int((sub["event_type"] == "PRIV_ESC").any())
        malware = int((sub["event_type"] == "MALWARE_ALERT").any())
        dns_exfil = int((sub["event_type"] == "DNS_EXFIL").any())
        http_exfil = int((sub["event_type"] == "HTTP_EXFIL").any())

        inter = sub["timestamp"].sort_values().diff().dt.total_seconds().dropna()
        avg_gap = inter.mean() if len(inter) > 0 else 0
        std_gap = inter.std() if len(inter) > 0 else 0
        weekend = int(sub["timestamp"].dt.dayofweek.isin([5,6]).any())

        pagerank = host_pr.get(host, 0.0)

        # label: simple heuristics for synthetic data; in real system, derive from analyst notes
        label = int((failed_login > 10) or port_scan or priv_esc or malware or dns_exfil or http_exfil)

        features.append([
            host, total_events, failed_login, vpn_conn, port_scan, priv_esc,
            malware, dns_exfil, http_exfil, avg_gap, std_gap, weekend, pagerank, label
        ])

    cols = ["host","total_events","failed_login","vpn_conn","port_scan","priv_esc",
            "malware","dns_exfil","http_exfil","avg_gap","std_gap","weekend","pagerank","label"]
    feat_df = pd.DataFrame(features, columns=cols)
    feat_df.to_csv(output_path, index=False)
    print(f"[INFO] Features saved -> {output_path}")
    return feat_df

if __name__ == "__main__":
    make_features()

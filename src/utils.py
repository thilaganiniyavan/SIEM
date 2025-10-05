import torch
import numpy as np
import random
import logging
import networkx as nx
import pandas as pd

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s'
    )

def compute_graph_features(df, host_col='host', event_col='event_type'):
    """
    Build a bipartite host-event graph and compute host PageRank as a relational feature.
    df: DataFrame with host,event_type,timestamp
    returns: dict host -> pagerank
    """
    # Build bipartite graph: hosts <-> events
    G = nx.Graph()
    hosts = df[host_col].unique()
    events = df[event_col].unique()
    for h in hosts:
        G.add_node(f"H_{h}", bipartite=0)
    for e in events:
        G.add_node(f"E_{e}", bipartite=1)
    # Add edges weighted by occurrence count
    grouped = df.groupby([host_col, event_col]).size().reset_index(name='count')
    for _, row in grouped.iterrows():
        G.add_edge(f"H_{row[host_col]}", f"E_{row[event_col]}", weight=int(row['count']))
    pr = nx.pagerank(G, weight='weight')
    # extract host pageranks
    host_pr = {node[2:]: score for node, score in pr.items() if node.startswith("H_")}
    return host_pr

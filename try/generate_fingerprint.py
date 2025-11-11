import os
import time
import json
import networkx as nx
from tqdm import tqdm
from collections import defaultdict

def canonical_fingerprint(adj):
    """Return canonical fingerprint for unlabeled graph."""
    G = nx.Graph()
    for i, neigh in enumerate(adj):
        for j in neigh:
            G.add_edge(i, j)
    # Canonical label: label-invariant
    return nx.to_graph6_bytes(G).decode().strip()

def find_duplicate_graphs(folder_path):
    fingerprints = defaultdict(list)

    # Collect all JSON files in folder
    json_files = [f for f in os.listdir(folder_path) if f.startswith('input_') and f.endswith('.json')]

    for filename in tqdm(json_files, desc="Processing graphs"):
        filepath = os.path.join(folder_path, filename)

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                # Supports either raw adjacency list or dict containing it
                adj = data # ['adjacency_list'] if isinstance(data, dict) and 'adjacency_list' in data else data
                
                t0 = time.time()
                fp = canonical_fingerprint(adj)
                t1 = time.time()
                
                fingerprints[fp].append(filename)
                # print(f"Processed {filename} in {t1 - t0:.4f} seconds.")
        except Exception as e:
            print(f"Error in {filename}: {e}")

    # Find duplicates
    duplicates = {fp: files for fp, files in fingerprints.items() if len(files) > 1}

    print(f"\n✅ Found {len(duplicates)} groups of duplicates.")
    for fp, files in duplicates.items():
        print(f"\nDuplicate group ({len(files)} files):")
        for f in files:
            print(f"  - {f}")

    return duplicates

if __name__ == "__main__":
    folder = "../live_data/results"
    find_duplicate_graphs(folder)
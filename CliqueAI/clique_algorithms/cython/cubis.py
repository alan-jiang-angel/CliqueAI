#!/usr/bin/env python3
"""
max_clique_cubis_parallel.py

Driver that:
- reads JSON adjacency from file or URL,
- runs CUBIS decomposition (pre-prune, core-decomposition),
- calls Cython BK core (bk_core.bron_kerbosch_max_cy) on CUBIS subgraphs,
- optionally runs parallel searches on multiple seeds/CUBIS.
"""

import sys, time, json
from collections import deque
try:
    import requests
except Exception:
    requests = None

import multiprocessing as mp

# Try import Cython module
try:
    import bk_core
except Exception as e:
    bk_core = None

# ------------------ small helpers (same logic as earlier) ------------------
def load_json_path_or_url(path):
    if path.startswith("http://") or path.startswith("https://"):
        if requests:
            r = requests.get(path, timeout=15)
            r.raise_for_status()
            return r.json()
        else:
            from urllib.request import urlopen
            with urlopen(path, timeout=15) as f:
                return json.load(f)
    else:
        with open(path, "r") as f:
            return json.load(f)

def adjacency_list_to_bitsets(adj_list):
    n = len(adj_list)
    adj = [0] * n
    for u, neigh in enumerate(adj_list):
        bits = 0
        for v in neigh:
            if v == u:
                continue
            bits |= (1 << v)
        adj[u] = bits
    return n, adj

def induced_subgraph_bitsets(orig_adj, nodes):
    nodes = list(nodes)
    sub_index = {v:i for i,v in enumerate(nodes)}
    m = len(nodes)
    adj_sub = [0] * m
    for i,v in enumerate(nodes):
        b = orig_adj[v]
        while b:
            lsb = b & -b
            u = lsb.bit_length() - 1
            b &= b - 1
            if u in sub_index:
                adj_sub[i] |= 1 << sub_index[u]
    return m, adj_sub, sub_index, nodes

def degree_from_bitset(b):
    return b.bit_count()

def core_decomposition(n, adj):
    deg = [degree_from_bitset(adj[v]) for v in range(n)]
    maxd = max(deg) if deg else 0
    bins = [deque() for _ in range(maxd+1)]
    for v, d in enumerate(deg):
        bins[d].append(v)
    core = [0]*n
    removed = [False]*n
    cur_deg = 0
    for _ in range(n):
        while cur_deg <= maxd and not bins[cur_deg]:
            cur_deg += 1
        if cur_deg > maxd:
            break
        v = bins[cur_deg].popleft()
        if removed[v]:
            continue
        removed[v] = True
        core[v] = cur_deg
        nb = adj[v]
        while nb:
            lsb = nb & -nb
            u = lsb.bit_length() - 1
            nb &= nb - 1
            if not removed[u]:
                d_old = deg[u]
                deg[u] -= 1
                try:
                    bins[d_old].remove(u)
                except ValueError:
                    pass
                bins[d_old-1].append(u)
    for v in range(n):
        if core[v] == 0 and not removed[v]:
            core[v] = deg[v]
    return core

# ------------------ Pre-prune + CUBIS helpers (lighter) ------------------
def greedy_heuristic_clique(n, adj, iterations=20):
    import random
    best = []
    verts = list(range(n))
    for _ in range(iterations):
        start = max(verts, key=lambda x: degree_from_bitset(adj[x])) if random.random() < 0.8 else random.choice(verts)
        clique = [start]
        cand = adj[start]
        while cand:
            cand_list = []
            b = cand
            while b:
                lsb = b & -b
                v = lsb.bit_length() - 1
                cand_list.append(v)
                b &= b - 1
            if not cand_list:
                break
            v = max(cand_list, key=lambda x: degree_from_bitset(adj[x] & cand))
            clique.append(v)
            cand &= adj[v]
        if len(clique) > len(best):
            best = clique.copy()
    return best

def pre_prune(n, adj, time_limit):
    degs = [degree_from_bitset(adj[v]) for v in range(n)]
    max_v = max(range(n), key=lambda x: degs[x])
    nb_bits = adj[max_v]
    neighbor_vertices = [max_v]
    b = nb_bits
    while b:
        lsb = b & -b
        u = lsb.bit_length() - 1
        neighbor_vertices.append(u)
        b &= b - 1
    # induced
    m, adj_sub, sub_index, nodes = induced_subgraph_bitsets(adj, neighbor_vertices)
    psi = 0
    psi_clique = []
    if m <= 100:
        # use Cython BK if available
        if bk_core:
            try:
                size, clique_local = bk_core.bron_kerbosch_max_cy(m, adj_sub, min(5.0, time_limit))
                psi = size
                psi_clique = [nodes[i] for i in clique_local]
            except Exception:
                psi_clique = greedy_heuristic_clique(n, adj, iterations=50)
                psi = len(psi_clique)
        else:
            psi_clique = greedy_heuristic_clique(n, adj, iterations=50)
            psi = len(psi_clique)
    else:
        psi_clique = greedy_heuristic_clique(n, adj, iterations=50)
        psi = len(psi_clique)

    # iterative pruning
    keep = set(range(n))
    removed = True
    while removed:
        removed = False
        # compute degrees inside keep quickly
        mask_keep = 0
        for u in keep:
            mask_keep |= (1 << u)
        to_remove = []
        for v in list(keep):
            if (adj[v] & mask_keep).bit_count() < (psi - 1):
                to_remove.append(v)
        if to_remove:
            removed = True
            for v in to_remove:
                keep.remove(v)

    n2, adj2, _, nodes_list = induced_subgraph_bitsets(adj, keep)
    return adj2, set(nodes_list), psi, psi_clique

def build_CUBIS_nodes(n, adj, core, c_val):
    nodes = set()
    for v in range(n):
        if core.get(v, -1) == c_val:
            nodes.add(v)
            b = adj[v]
            while b:
                lsb = b & -b
                u = lsb.bit_length() - 1
                b &= b - 1
                if core.get(u, -1) > c_val:
                    nodes.add(u)
    return nodes

# ------------------ Worker wrappers ------------------
def run_bk_on_subgraph(orig_adj, nodes_subset, time_budget):
    """Induce subgraph and call Cython BK if available else fallback to basic python
       Returns (size, clique_original_indices)
    """
    m, adj_sub, sub_index, nodes = induced_subgraph_bitsets(orig_adj, nodes_subset)
    if m == 0:
        return 0, []
    if bk_core:
        try:
            size, clique_local = bk_core.bron_kerbosch_max_cy(m, adj_sub, time_budget)
            clique_orig = [nodes[i] for i in clique_local]
            return size, clique_orig
        except Exception:
            # fallback to Python simple BK (not included to keep code concise)
            pass
    # Last-resort heuristic if no C extension
    clique = greedy_heuristic_clique(m, adj_sub, iterations=50)
    clique_orig = [nodes[i] for i in clique]
    return len(clique_orig), clique_orig

# ------------------ Top-level CUBIS driver similar to earlier script ------------------
def cubis_driver(adj_list, global_time_limit=30.0, parallel=True):
    t0 = time.time()
    n, adj = adjacency_list_to_bitsets(adj_list)
    if n == 0:
        return 0, []
    rem_time = max(0.1, global_time_limit - (time.time() - t0))
    adj_pruned, nodes_pruned_set, psi, psi_clique = pre_prune(n, adj, rem_time)
    orig_nodes = sorted(nodes_pruned_set)
    if not orig_nodes:
        return psi, psi_clique
    m, adj_sub, orig_to_sub, sub_to_orig = induced_subgraph_bitsets(adj, orig_nodes)
    core_list = core_decomposition(m, adj_sub)
    core_orig = {}
    for i, orig_v in enumerate(sub_to_orig):
        core_orig[orig_v] = core_list[i]
    core_vals = sorted(set(core_list), reverse=True)
    if not core_vals:
        return psi, psi_clique
    c_max = core_vals[0]
    c2nd = core_vals[1] if len(core_vals) > 1 else c_max

    nodes_Gcmax = build_CUBIS_nodes(n, adj, core_orig, c_max).intersection(nodes_pruned_set)
    best_candidates = [(len(psi_clique), psi_clique)]

    # prepare tasks: always evaluate G_cmax, and optionally G_c2nd
    tasks = []
    if nodes_Gcmax:
        tasks.append(("Gcmax", nodes_Gcmax))

    # compute c_min per paper idea (we'll run Gc2 if needed)
    # quick run on Gcmax to get omega estimate
    total_time = global_time_limit
    elapsed = time.time() - t0
    rem_time = max(0.1, total_time - elapsed)
    if tasks:
        # Optionally parallelize: run each task in a separate process
        if parallel and len(tasks) > 1:
            with mp.Pool(processes=min(mp.cpu_count(), len(tasks))) as pool:
                async_res = []
                for name, nodeset in tasks:
                    # give each task a proportion of remaining time
                    async_res.append(pool.apply_async(run_bk_on_subgraph, args=(adj, nodeset, rem_time / len(tasks))))
                for a in async_res:
                    try:
                        size, clique = a.get(timeout=rem_time + 1.0)
                    except Exception:
                        size, clique = 0, []
                    best_candidates.append((size, clique))
        else:
            # serial run
            for name, nodeset in tasks:
                size, clique = run_bk_on_subgraph(adj, nodeset, rem_time)
                best_candidates.append((size, clique))

    # If Gcmax result indicates we need second subgraph, build and run it
    # compute best so far
    best_size, best_clique = max(best_candidates, key=lambda x: x[0])
    # condition to build second CUBIS (simple version)
    if best_size < c2nd:
        # build c_min = best_size -1
        c_min = max(0, best_size - 1)
        nodes_Gc2 = set()
        for v in orig_nodes:
            cv = core_orig.get(v, 0)
            if c_min <= cv <= c2nd:
                nodes_Gc2.add(v)
                b = adj[v]
                while b:
                    lsb = b & -b
                    u = lsb.bit_length() - 1
                    b &= b - 1
                    if core_orig.get(u, -1) > cv:
                        nodes_Gc2.add(u)
        nodes_Gc2 = nodes_Gc2.intersection(nodes_pruned_set)
        if nodes_Gc2:
            elapsed = time.time() - t0
            rem_time = max(0.1, total_time - elapsed)
            size2, clique2 = run_bk_on_subgraph(adj, nodes_Gc2, rem_time)
            if size2 > best_size:
                best_size, best_clique = size2, clique2

    # compare with psi
    if len(psi_clique) > best_size:
        best_size, best_clique = len(psi_clique), psi_clique

    return best_size, sorted(best_clique)

# ------------------ Main ------------------
def main():
    # data = load_json_path_or_url("https://raw.githubusercontent.com/toptensor/CliqueAI/refs/heads/main/test_data/general_0.2.json")
    # adj_list = data.get("adjacency_list")
    data = load_json_path_or_url("../../../results2/input_2025-11-02_17-19-09.json")
    adj_list = data

    if adj_list is None:
        print("JSON must contain 'adjacency_list'.")
        return

    TIME_LIMIT = 30.0
    t0 = time.time()
    size, clique = cubis_driver(adj_list, global_time_limit=TIME_LIMIT, parallel=True)
    t1 = time.time()
    print(f"Elapsed: {t1-t0:.3f}s (limit {TIME_LIMIT}s)")
    print(f"Max clique size (best found): {size}")
    print(f"Clique vertices (0-based): {clique}")

if __name__ == "__main__":
    main()
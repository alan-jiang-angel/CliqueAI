"""
max_clique_cubis.py

Non CPU-Intensive, but, smaller result
"""

import time
import json
import sys
from collections import deque

# optional requests
try:
    import requests
except Exception:
    requests = None

# -------------------------
# Utility: Read JSON (file or URL)
# -------------------------
def load_json_path_or_url(path):
    if path.startswith("http://") or path.startswith("https://"):
        if requests:
            r = requests.get(path, timeout=15)
            r.raise_for_status()
            return r.json()
        else:
            # fallback
            from urllib.request import urlopen
            with urlopen(path, timeout=15) as f:
                return json.load(f)
    else:
        with open(path, "r") as f:
            return json.load(f)

# -------------------------
# Graph helpers (bitset representation)
# -------------------------
def adjacency_list_to_bitsets(adj_list):
    n = len(adj_list)
    adj = [0] * n
    for u, neigh in enumerate(adj_list):
        bits = 0
        for v in neigh:
            if v == u:
                continue
            if v < 0 or v >= n:
                raise ValueError("Invalid vertex index in adjacency list")
            bits |= (1 << v)
        adj[u] = bits
    return n, adj

def induced_subgraph_bitsets(orig_adj, nodes):
    """Return adjacency bitsets and index map for the induced subgraph on 'nodes' (iterable of original indices).
       Returns (m, adj_sub, orig_to_sub, sub_to_orig).
    """
    nodes = list(nodes)
    sub_index = {v:i for i,v in enumerate(nodes)}
    m = len(nodes)
    adj_sub = [0] * m
    for i,v in enumerate(nodes):
        # original bitset intersect nodes
        bits = orig_adj[v]
        # map bits to sub-index bitset
        b = bits
        while b:
            lsb = b & -b
            u = lsb.bit_length() - 1
            b &= b - 1
            if u in sub_index:
                adj_sub[i] |= 1 << sub_index[u]
    return m, adj_sub, sub_index, nodes

def degree_from_bitset(b):
    return b.bit_count()

# -------------------------
# k-core decomposition (returns core number per vertex)
# -------------------------
def core_decomposition(n, adj):
    deg = [degree_from_bitset(adj[v]) for v in range(n)]
    maxd = max(deg) if deg else 0
    bins = [deque() for _ in range(maxd+1)]
    for v, d in enumerate(deg):
        bins[d].append(v)
    core = [0]*n
    removed = [False]*n
    cur_deg = 0
    for k in range(n):
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
                # move u between bins (linear removal from deque acceptable for moderate n)
                try:
                    bins[d_old].remove(u)
                except ValueError:
                    pass
                bins[d_old-1].append(u)
    # For nodes not removed (isolated tail), set their core as deg (remaining)
    for v in range(n):
        if core[v] == 0 and not removed[v]:
            core[v] = deg[v]
    return core

# -------------------------
# Greedy heuristic clique (quick lower bound)
# -------------------------
import random
def greedy_heuristic_clique(n, adj, iterations=20):
    best = []
    verts = list(range(n))
    for _ in range(iterations):
        # start with random vertex among top-degree ones
        start = max(verts, key=lambda x: degree_from_bitset(adj[x])) if random.random() < 0.8 else random.choice(verts)
        clique = [start]
        cand = adj[start]
        while cand:
            # choose the candidate with maximum degree inside cand
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

# -------------------------
# Bron-Kerbosch with pivot + greedy coloring bound (bitset version)
# returns (max_clique_list)
# -------------------------
def greedy_color_order_bits(cand_bits, adj, ordering_pos=None):
    """Return order list and color numbers for vertices in cand_bits (LSB-first)."""
    verts = []
    b = cand_bits
    while b:
        lsb = b & -b
        v = lsb.bit_length() - 1
        verts.append(v)
        b &= b - 1
    if ordering_pos is not None:
        verts.sort(key=lambda x: ordering_pos.get(x,0), reverse=True)
    color_of = {}
    res_order = []
    res_color = []
    uncolored = set(verts)
    color = 0
    while uncolored:
        color += 1
        forbidden = 0
        # iterate over a snapshot to allow removal
        for v in list(uncolored):
            if forbidden & (1 << v):
                continue
            res_order.append(v)
            res_color.append(color)
            forbidden |= adj[v]
            uncolored.remove(v)
    return res_order, res_color

def bron_kerbosch_bitset_max(n, adj, time_limit, initial_bound=0):
    """Find maximum clique in graph (adj bitsets length n).
       Returns (size, clique_list). Stops early if time_limit exceeded (best-so-far returned).
    """
    start = time.time()
    best_clique = []
    best_size = initial_bound

    # Ordering map: simple degeneracy-like order (degree descending)
    degs = [degree_from_bitset(adj[v]) for v in range(n)]
    order = sorted(range(n), key=lambda x: degs[x], reverse=True)
    ordering_pos = {v:i for i,v in enumerate(order)}

    nodes_searched = 0
    TIME_CHECK_INTERVAL = 2048

    def recurse(cand_bits, cur_clique):
        nonlocal best_clique, best_size, nodes_searched
        nodes_searched += 1
        if (nodes_searched & (TIME_CHECK_INTERVAL-1)) == 0:
            if time.time() - start > time_limit:
                raise TimeoutError
        if cand_bits == 0:
            if len(cur_clique) > best_size:
                best_size = len(cur_clique)
                best_clique = cur_clique.copy()
            return

        order_list, colors = greedy_color_order_bits(cand_bits, adj, ordering_pos)
        # process in reverse (highest color first for good pruning)
        for v, col in zip(reversed(order_list), reversed(colors)):
            if len(cur_clique) + col <= best_size:
                return
            # include v
            cur_clique.append(v)
            new_cand = cand_bits & adj[v]
            if new_cand:
                recurse(new_cand, cur_clique)
            else:
                if len(cur_clique) > best_size:
                    best_size = len(cur_clique)
                    best_clique = cur_clique.copy()
            cur_clique.pop()
            cand_bits &= ~(1 << v)
            if cand_bits == 0:
                if len(cur_clique) > best_size:
                    best_size = len(cur_clique)
                    best_clique = cur_clique.copy()
                return

    # initial candidates: all vertices
    all_bits = (1 << n) - 1
    try:
        recurse(all_bits, [])
    except TimeoutError:
        # return best found so far
        pass
    return best_size, best_clique

# -------------------------
# CUBIS construction per paper:
# Given core numbers 'core', for a chosen core value c:
# G_c nodes N = { v | core[v] == c } U { neighbors of those v with core > c }
# -------------------------
def build_CUBIS_nodes(n, adj, core, c_val):
    nodes = set()
    for v in range(n):
        if core[v] == c_val:
            nodes.add(v)
            nb = adj[v]
            b = nb
            while b:
                lsb = b & -b
                u = lsb.bit_length() - 1
                b &= b - 1
                if core[u] > c_val:
                    nodes.add(u)
    return nodes

# -------------------------
# Pre-pruning: neighborhood of max-degree node and iterative degree pruning using heuristic bound psi
# -------------------------
def pre_prune(n, adj, time_limit, max_neighbors_search_limit=500):
    """Return (pruned_adj, pruned_nodes_set, psi) where psi is initial clique lower bound."""
    start = time.time()
    # get degrees
    degs = [degree_from_bitset(adj[v]) for v in range(n)]
    # pick node with max degree
    max_v = max(range(n), key=lambda x: degs[x])
    # build neighborhood subgraph of max_v (including max_v)
    nb_bits = adj[max_v]
    neighbor_vertices = [max_v]
    b = nb_bits
    while b:
        lsb = b & -b
        u = lsb.bit_length() - 1
        neighbor_vertices.append(u)
        b &= b - 1
    # if neighborhood is too large, we can sample top-degree neighbors
    if len(neighbor_vertices) > max_neighbors_search_limit:
        neighbor_vertices = neighbor_vertices[:max_neighbors_search_limit]

    # induced subgraph
    m, adj_sub, sub_index, nodes = induced_subgraph_bitsets(adj, neighbor_vertices)
    # find heuristic clique within neighborhood (use greedy or BK if small)
    psi = 0
    psi_clique = []
    # use BK if small
    if m <= 100:
        size, clique = bron_kerbosch_bitset_max(m, adj_sub, time_limit= min(5.0, max(0.1, time_limit - (time.time() - start))), initial_bound=0)
        # map back to original indices
        psi_clique = [nodes[i] for i in clique]
        psi = size
    else:
        # use greedy heuristic on whole graph
        hc = greedy_heuristic_clique(n, adj, iterations=50)
        psi = len(hc)
        psi_clique = hc

    # iterative pruning: remove nodes with degree < psi-1
    removed = True
    keep = set(range(n))
    while removed:
        removed = False
        to_remove = [v for v in keep if degree_from_bitset(adj[v] & sum((1<<u) for u in keep)) < (psi - 1)]
        if to_remove:
            removed = True
            for v in to_remove:
                keep.remove(v)

    # build pruned adjacency for remaining nodes
    n2, adj2, _, nodes_list = induced_subgraph_bitsets(adj, keep)
    # return pruned adjacency in original indexing (but we also return nodes_list for map)
    return adj2, set(nodes_list), psi, psi_clique

# -------------------------
# Top-level CUBIS algorithm
# -------------------------
def cubis_max_clique(adj_list, global_time_limit=30.0):
    start_time = time.time()
    data_start = start_time
    n, adj = adjacency_list_to_bitsets(adj_list)
    if n == 0:
        return 0, []
    # Pre-prune using neighborhood heuristic
    remaining_time = max(0.1, global_time_limit - (time.time() - start_time))
    adj_pruned, nodes_pruned_set, psi, psi_clique = pre_prune(n, adj, remaining_time)
    # If pre-prune removed nothing, nodes_pruned_set may equal full
    # Build core numbers on the pruned graph (map needed)
    # We need mapping original_index -> new index
    orig_nodes = sorted(nodes_pruned_set)
    if len(orig_nodes) == 0:
        # trivial fallback
        return psi, psi_clique

    m, adj_sub, orig_to_sub, sub_to_orig = induced_subgraph_bitsets(adj, orig_nodes)
    core = core_decomposition(m, adj_sub)  # core numbers for subgraph indices

    # convert to original indexing for convenience
    core_orig = {}
    for i, orig_v in enumerate(sub_to_orig):
        core_orig[orig_v] = core[i]

    # determine unique core values sorted descending
    core_vals = sorted(set(core), reverse=True)
    if not core_vals:
        return psi, psi_clique

    c_max = core_vals[0]
    c2nd = core_vals[1] if len(core_vals) > 1 else c_max

    # Build G_cmax (on original indices) and search
    nodes_Gcmax = build_CUBIS_nodes(n, adj, core_orig, c_max)
    # restrict to pruned nodes only (we only computed cores on pruned graph)
    nodes_Gcmax = nodes_Gcmax.intersection(nodes_pruned_set)
    if not nodes_Gcmax:
        # fallback: run BK on full pruned graph
        rem_time = max(0.1, global_time_limit - (time.time() - start_time))
        size1, clique1 = bron_kerbosch_bitset_max(m, adj_sub, rem_time, initial_bound=len(psi_clique))
        clique1 = [sub_to_orig[v] for v in clique1]
        best_size = max(len(psi_clique), size1)
        best_clique = psi_clique if len(psi_clique) >= size1 else clique1
        return best_size, best_clique

    # Induce G_cmax
    m1, adj1, map1, list1 = induced_subgraph_bitsets(adj, nodes_Gcmax)
    rem_time = max(0.1, global_time_limit - (time.time() - start_time))
    size1, clique1_local = bron_kerbosch_bitset_max(m1, adj1, rem_time, initial_bound=len(psi_clique))
    clique1 = [list1[i] for i in clique1_local]  # map back to original indices

    # If we already beat/meet second-core threshold, may terminate early
    # Per paper: if omega(G_cmax) >= c2nd then we can stop (c2nd is second largest core)
    if size1 >= c2nd:
        best = clique1 if size1 >= len(psi_clique) else psi_clique
        best_size = max(len(best), size1, len(psi_clique))
        return best_size, best

    # Otherwise compute c_min = minimal core number such that c_min + 1 >= omega(G_cmax)
    c_min = max(0, size1 - 1)
    # find smallest core value >= c_min among core_vals (but <= c2nd)
    eligible_cores = [cv for cv in set(core) if c_min <= cv <= c2nd]
    if not eligible_cores:
        # no second CUBIS needed, return best so far
        best = clique1 if size1 >= len(psi_clique) else psi_clique
        best_size = max(size1, len(psi_clique))
        return best_size, best

    # build G_c2nd: nodes with core in [c_min, c2nd] and their neighbors with higher core
    # Note: core_orig contains only nodes we computed (pruned set)
    nodes_Gc2 = set()
    for v in sub_to_orig:  # these are original indices available in pruned graph
        cv = core_orig.get(v, 0)
        if c_min <= cv <= c2nd:
            nodes_Gc2.add(v)
            # add neighbors with higher core
            b = adj[v]
            while b:
                lsb = b & -b
                u = lsb.bit_length() - 1
                b &= b - 1
                if core_orig.get(u, -1) > cv:
                    nodes_Gc2.add(u)
    # restrict to pruned set
    nodes_Gc2 = nodes_Gc2.intersection(nodes_pruned_set)
    if not nodes_Gc2:
        # nothing to do
        best = clique1 if size1 >= len(psi_clique) else psi_clique
        best_size = max(size1, len(psi_clique))
        return best_size, best

    # Induce G_c2 and search
    m2, adj2, map2, list2 = induced_subgraph_bitsets(adj, nodes_Gc2)
    rem_time = max(0.1, global_time_limit - (time.time() - start_time))
    size2, clique2_local = bron_kerbosch_bitset_max(m2, adj2, rem_time, initial_bound=max(len(psi_clique), size1))
    clique2 = [list2[i] for i in clique2_local]

    # choose best among psi_clique, clique1, clique2
    candidates = [(len(psi_clique), psi_clique), (size1, clique1), (size2, clique2)]
    best_size, best_clique = max(candidates, key=lambda x: x[0])
    return best_size, best_clique

# -------------------------
# Main entrypoint
# -------------------------
def main():
    
    data = load_json_path_or_url("https://raw.githubusercontent.com/toptensor/CliqueAI/refs/heads/main/test_data/general_0.2.json")
    adj_list = data.get("adjacency_list")
    
    if adj_list is None:
        print("JSON must contain 'adjacency_list' key.")
        return

    TIME_LIMIT = 30.0
    t0 = time.time()
    size, clique = cubis_max_clique(adj_list, global_time_limit=TIME_LIMIT)
    t1 = time.time()
    print(f"Elapsed: {t1-t0:.3f}s (limit {TIME_LIMIT}s)")
    print(f"Max clique size (best found): {size}")
    print(f"Clique vertices (0-based): {sorted(clique)}")

if __name__ == "__main__":
    main()
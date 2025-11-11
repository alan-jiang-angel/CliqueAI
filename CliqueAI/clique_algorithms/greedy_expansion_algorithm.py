import json
import random
from multiprocessing import Pool, cpu_count


# --------------------------------------------------------
# Load adjacency list from JSON → convert to bitmask only
# --------------------------------------------------------
def load_adjlist(data):
    # normalize dict or list input
    if isinstance(data, dict):
        n = max(int(k) for k in data.keys()) + 1
        tmp = [set() for _ in range(n)]
        for k, vs in data.items():
            u = int(k)
            for v in vs:
                if v != u:
                    tmp[u].add(int(v))
    else:
        n = len(data)
        tmp = [set(v for v in vs if v != i) for i, vs in enumerate(data)]

    # enforce symmetry
    for u in range(n):
        for v in list(tmp[u]):
            tmp[v].add(u)

    # ✅ convert adjacency to bitmask (only integer bitsets)
    adj = [0] * n
    for u in range(n):
        mask = 0
        for v in tmp[u]:
            mask |= 1 << v
        adj[u] = mask

    return adj


# --------------------------------------------------------
# Beam greedy clique finder (bitmask-only version)
# --------------------------------------------------------
def greedy_clique(adj, start_node, beam_width=3):
    n = len(adj)

    # (clique_list, candidate_bitmask)
    beam = [([start_node], adj[start_node])]
    best = [start_node]

    while beam:
        next_level = []

        for clique, candidates in beam:
            if candidates == 0:
                if len(clique) > len(best):
                    best = clique
                continue

            # extract candidate nodes
            candidate_nodes = [v for v in range(n) if (candidates >> v) & 1]

            # ✅ correct ranking: mutual neighbors ONLY
            ranked = sorted(
                candidate_nodes,
                key=lambda v: (adj[v] & candidates).bit_count(),
                reverse=True,
            )

            for v in ranked[:beam_width]:
                new_clique = clique + [v]
                new_candidates = candidates & adj[v]

                next_level.append((new_clique, new_candidates))

                if len(new_clique) > len(best):
                    best = new_clique

        # prune weakest branches
        beam = sorted(
            next_level,
            key=lambda item: item[1].bit_count(),
            reverse=True,
        )[:beam_width]

    return best

# --------------------------------------------------------
# Helper for multiprocessing
# --------------------------------------------------------
def run_clique(args):
    adj, start_node, beam_width = args
    return greedy_clique(adj, start_node, beam_width)


# --------------------------------------------------------
# Multicore max clique (degree ordering, no tqdm)
# --------------------------------------------------------
def approximate_max_clique(adj, top_k=200, beam_width=3):
    n = len(adj)

    # pick top_k by degree (bitcount)
    nodes_by_degree = sorted(range(n), key=lambda v: adj[v].bit_count(), reverse=True)
    search_nodes = nodes_by_degree[:top_k]

    best = []
    # with Pool(cpu_count()) as pool:
    with Pool(12) as pool:
        tasks = [(adj, start, beam_width) for start in search_nodes]

        for clique in pool.imap_unordered(run_clique, tasks):
            if len(clique) > len(best):
                best = clique

    return sorted(best)

def greedy_expansion_algorithm(number_of_nodes, graph):
    adj = load_adjlist(graph)
    clique = approximate_max_clique(adj, top_k=300, beam_width=80)
    return sorted(clique)


# --------------------------------------------------------
# Main
# --------------------------------------------------------
# if __name__ == "__main__":
    
    # adj = load_adjlist(path)
    # clique = approximate_max_clique(adj, top_k=300, beam_width=50)

    # print("\nAPPROX CLIQUE SIZE:", len(clique))
    # print("CLIQUE:", clique)

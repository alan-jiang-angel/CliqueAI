import json
import random
import os
import glob
from multiprocessing import Pool, cpu_count
import sys
import time

# ========================================================
# Load adjacency list
# ========================================================

def load_adjlist(data):
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

    for u in range(n):
        for v in list(tmp[u]):
            tmp[v].add(u)

    adj = [0] * n
    for u in range(n):
        mask = 0
        for v in tmp[u]:
            mask |= 1 << v
        adj[u] = mask

    return adj


# ========================================================
# Load clique result
# ========================================================

def load_clique(path):
    with open(path) as f:
        data = json.load(f)
    return sorted(map(int, data))


# ========================================================
# Bit iteration
# ========================================================

def iter_bits(x):
    while x:
        v = (x & -x).bit_length() - 1
        x &= x - 1
        yield v


# ========================================================
# Randomized beam greedy clique finder
# ========================================================

def greedy_clique(adj, start_node, beam_width=5):

    beam = [([start_node], adj[start_node])]
    best = [start_node]

    while beam:
        next_level = []

        for clique, candidates in beam:

            if candidates == 0:
                if len(clique) > len(best):
                    best = clique
                continue

            candidate_nodes = list(iter_bits(candidates))

            scored = [
                (v, (adj[v] & candidates).bit_count())
                for v in candidate_nodes
            ]

            scored.sort(key=lambda x: x[1], reverse=True)

            TOP = min(len(scored), 10)
            head = scored[:TOP]
            tail = scored[TOP:]

            random.shuffle(head)

            ranked = [v for v, _ in head] + [v for v, _ in tail]

            for v in ranked[:beam_width]:
                new_clique = clique + [v]
                new_candidates = candidates & adj[v]

                next_level.append((new_clique, new_candidates))

                if len(new_clique) > len(best):
                    best = new_clique

        random.shuffle(next_level)

        beam = sorted(
            next_level,
            key=lambda item: item[1].bit_count(),
            reverse=True,
        )[:beam_width]

    return best


# ========================================================
# Multiprocessing helper
# ========================================================

def run_clique(args):
    adj, start_node, beam_width = args
    random.seed(os.getpid() ^ random.randint(0, 10**9))
    return greedy_clique(adj, start_node, beam_width)


# ========================================================
# Multicore approximate max clique (single call)
# ========================================================

def approximate_max_clique(adj, top_k=200, beam_width=5):

    n = len(adj)

    nodes_by_degree = sorted(
        range(n), key=lambda v: adj[v].bit_count(), reverse=True
    )

    elite = nodes_by_degree[: max(10, top_k // 3)]
    others = nodes_by_degree[max(10, top_k // 3): 5 * top_k]

    sampled = elite + random.sample(
        others,
        min(len(others), max(0, top_k - len(elite)))
    )

    random.shuffle(sampled)

    best = []

    with Pool(cpu_count()) as pool:
        tasks = [(adj, start, beam_width) for start in sampled]

        for clique in pool.imap_unordered(run_clique, tasks, chunksize=1):
            if len(clique) > len(best):
                best = clique

    return sorted(best)


# ========================================================
# 30 second adaptive loop for one graph
# ========================================================

def run_30s_solver(adj):

    node_len = len(adj)

    if (node_len > 500):
        top_k = node_len * 2 // 7
    else:
        top_k = node_len * 2 // 5

    if (node_len > 500):
        max_beam = max(4, node_len // 25)
    else:
        max_beam = max(4, node_len // 20)

    TIME_LIMIT = 20.0

    start_time = time.perf_counter()
    deadline = start_time + TIME_LIMIT

    max_len = 0
    best = []
    times = []
    run_id = 0

    while True:

        now = time.perf_counter()

        if times:
            avg = sum(times) / len(times)
            if now + avg >= deadline:
                break
        elif now >= deadline:
            break

        run_id += 1

        bw = random.randint(3, max_beam)

        t0 = time.perf_counter()

        clique = approximate_max_clique(
            adj,
            top_k=top_k,
            beam_width=bw,
        )

        t1 = time.perf_counter()
        elapsed = t1 - t0
        times.append(elapsed)

        if len(clique) > max_len:
            max_len = len(clique)
            best = clique
            # print(
            #     f"  [Run {run_id}] NEW BEST={max_len} "
            #     f"time={elapsed:.2f}s beam={bw}"
            # )
        # else:
            # print(
            #     f"  [Run {run_id}] size={len(clique)} "
            #     f"time={elapsed:.2f}s beam={bw}"
            # )

    return best, sum(times), len(times)


def greedy_exp_rand_algorithm(number_of_nodes, graph):
    adj = load_adjlist(graph)
    clique, total_time, run_count = run_30s_solver(adj)
    print(f"🔹 Greedy Exp Rand: found clique of size {len(clique)} in {total_time:.2f}s over {run_count} runs")
    return sorted(clique)

# import json
import sys
import time
import multiprocessing as mp
from collections import deque
import requests
import time

# CPU-Intensive, 
# Produces 22 Max Clique for 0.2 difficulty graph in 30 seconds
# Produces 24 Max Clique for 0.4 difficulty graph in 30 seconds

# ---------------------------
# Configuration
# ---------------------------
TIME_LIMIT_SECONDS = 25.0   # respond within 25 seconds as requested
MAX_VERTICES = 1000         # safety cap (user said up to 500) - adjust if needed
# ---------------------------


# ---------------------------
# Utilities: bitset representation
# ---------------------------
def graph_from_adjacency_list(adj_list):
    """
    Convert adjacency list (list of lists of ints) to bitset adjacency list (list of ints).
    Assumes vertices are 0..n-1.
    """
    n = len(adj_list)
    if n > MAX_VERTICES:
        raise ValueError(f"Graph has {n} vertices which exceeds MAX_VERTICES={MAX_VERTICES}.")
    adj = [0] * n
    for u, neigh in enumerate(adj_list):
        bits = 0
        for v in neigh:
            if v < 0 or v >= n:
                raise ValueError("Adjacency list contains invalid vertex index.")
            if v == u:
                continue
            bits |= 1 << v
        adj[u] = bits
    return n, adj

def bitcount(x):
    return x.bit_count()

def bit_iter(x):
    """Yield indices of set bits (LSB-first)."""
    while x:
        lsb = x & -x
        idx = lsb.bit_length() - 1
        yield idx
        x &= x - 1

# ---------------------------
# Degeneracy ordering (reduces branching factor)
# ---------------------------
def degeneracy_ordering(n, adj):
    deg = [bitcount(adj[v]) for v in range(n)]
    maxd = max(deg) if deg else 0
    bins = [deque() for _ in range(maxd + 1)]
    for v, d in enumerate(deg):
        bins[d].append(v)
    removed = [False] * n
    order = []
    cur_deg = 0
    for _ in range(n):
        while cur_deg <= maxd and not bins[cur_deg]:
            cur_deg += 1
        if cur_deg > maxd:
            # append remaining
            for v in range(n):
                if not removed[v]:
                    order.append(v)
                    removed[v] = True
            break
        v = bins[cur_deg].popleft()
        if removed[v]:
            continue
        removed[v] = True
        order.append(v)
        nb = adj[v]
        while nb:
            u = (nb & -nb).bit_length() - 1
            nb &= nb - 1
            if not removed[u]:
                d_old = deg[u]
                deg[u] -= 1
                # move u from bins[d_old] to bins[d_old-1]
                # removing from deque by value is O(k) but amortized okay here
                try:
                    bins[d_old].remove(u)
                except ValueError:
                    pass
                bins[d_old - 1].append(u)
    # reverse to get an ordering that tends to put high-core vertices first
    order.reverse()
    pos = {v: i for i, v in enumerate(order)}
    return order, pos

# ---------------------------
# Greedy coloring bound (approx upper bound)
# ---------------------------
def greedy_color_order(cand_bits, adj, ordering_pos=None):
    """
    Color candidate set greedily to produce ordering and color numbers.
    Returns (order_list, color_list) corresponding 1-based colors.
    """
    verts = [v for v in bit_iter(cand_bits)]
    if ordering_pos is not None:
        verts.sort(key=lambda x: ordering_pos.get(x, 0), reverse=True)
    color = 0
    uncolored = set(verts)
    order_res = []
    color_res = []
    while uncolored:
        color += 1
        forbidden = 0
        # iterate over a snapshot to avoid modifying while iterating
        for v in list(uncolored):
            if forbidden & (1 << v):
                continue
            order_res.append(v)
            color_res.append(color)
            forbidden |= adj[v]
            uncolored.remove(v)
    return order_res, color_res

# ---------------------------
# Single-threaded branch and bound solver (uses shared best for pruning)
# ---------------------------
def solve_subproblem(start_vertex, n, adj, ordering_pos, shared_best, end_time, time_check_interval=1024):
    """
    This function runs in a worker process.
    start_vertex: the vertex this process explores first (int)
    shared_best: mp.Manager().Namespace() with attributes: best_size (int), best_clique (list)
    end_time: float (epoch) deadline
    """
    # Local copy for speed
    best_size_local = shared_best.best_size

    nodes_visited = 0

    # Internal recursive search using Python int bitsets
    def search(cand_bits, clique):
        nonlocal best_size_local, nodes_visited
        # periodic time check
        nodes_visited += 1
        if nodes_visited & (time_check_interval - 1) == 0:
            if time.time() >= end_time:
                return False  # signal to stop due to timeout
            # refresh shared best for pruning
            if shared_best.best_size > best_size_local:
                best_size_local = shared_best.best_size

        if cand_bits == 0:
            if len(clique) > best_size_local:
                best_size_local = len(clique)
                # update shared best
                shared_best.best_size = best_size_local
                shared_best.best_clique = clique.copy()
            return True

        # coloring bound
        order, colors = greedy_color_order(cand_bits, adj, ordering_pos)
        # process in reverse order to try larger colors first (stronger pruning)
        for v, col in zip(reversed(order), reversed(colors)):
            if len(clique) + col <= best_size_local:
                return True  # prune this branch
            # include v
            clique.append(v)
            new_cand = cand_bits & adj[v]
            cont = search(new_cand, clique)
            if not cont:
                return False
            clique.pop()
            # remove v from candidates
            cand_bits &= ~(1 << v)
            if cand_bits == 0:
                # leaf
                if len(clique) > best_size_local:
                    best_size_local = len(clique)
                    shared_best.best_size = best_size_local
                    shared_best.best_clique = clique.copy()
                return True
        return True

    # initial candidate set: start_vertex plus neighbors that appear after it in ordering (to avoid duplicates)
    # We restrict to vertices with ordering_pos greater than start to avoid repeated work across workers.
    threshold = ordering_pos[start_vertex]
    # Build candidate bits with vertices whose ordering_pos >= threshold
    cand_bits = 0
    for v in range(n):
        if ordering_pos[v] >= threshold:
            cand_bits |= (1 << v)
    # intersect with neighbors of start_vertex to explore clique containing start_vertex
    cand_bits &= adj[start_vertex]
    clique = [start_vertex]
    # Also consider the trivial clique of single start_vertex
    if 1 > best_size_local:
        best_size_local = 1
        shared_best.best_size = 1
        shared_best.best_clique = [start_vertex]

    # run search
    try:
        ok = search(cand_bits, clique)
    except RecursionError:
        # fall back to iterative / return current best
        ok = True

    return True  # worker finished (or stopped)

# ---------------------------
# Parallel driver
# ---------------------------
def parallel_max_clique(n, adj, time_limit=TIME_LIMIT_SECONDS):
    manager = mp.Manager()
    shared_best = manager.Namespace()
    shared_best.best_size = 0
    shared_best.best_clique = []

    # ordering for pruning and splitting tasks
    order, ordering_pos = degeneracy_ordering(n, adj)

    # create task list: we will spawn tasks starting from high-core vertices first
    # limit tasks to n (one per vertex); processes will be limited by CPU count
    tasks = order[:]  # vertices in degeneracy order

    cpu_count = mp.cpu_count()
    max_workers = max(1, cpu_count)
    # Use a pool of worker processes implemented manually so we can pass shared namespace
    processes = []
    task_iter = iter(tasks)
    end_time = time.time() + time_limit

    # worker launcher: start up to max_workers processes
    def launch_proc(vertex):
        p = mp.Process(target=solve_subproblem, args=(vertex, n, adj, ordering_pos, shared_best, end_time))
        p.start()
        return p

    # start initial batch
    for _ in range(min(max_workers, len(tasks))):
        try:
            v = next(task_iter)
        except StopIteration:
            break
        p = launch_proc(v)
        processes.append((p, v))

    # as processes finish, start new ones until tasks exhausted or time is up
    try:
        while processes and time.time() < end_time:
            for i, (p, v) in enumerate(list(processes)):
                if not p.is_alive():
                    p.join(timeout=0.1)
                    processes.remove((p, v))
                    # launch next task if any
                    try:
                        nv = next(task_iter)
                        np_proc = launch_proc(nv)
                        processes.append((np_proc, nv))
                    except StopIteration:
                        pass
            time.sleep(0.01)
    except KeyboardInterrupt:
        # attempt graceful shutdown
        pass
    finally:
        # time's up or tasks done: terminate any running processes (they'll have checked shared end_time)
        for p, v in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=0.1)

    # return best found
    return shared_best.best_size, list(shared_best.best_clique)

def bron_kerbosch_algorithm(number_of_nodes: int, adjacency_list: list[list[int]], time_limit=TIME_LIMIT_SECONDS) -> list[int]:
    n, adj = graph_from_adjacency_list(adjacency_list)
    size, clique = parallel_max_clique(number_of_nodes, adj, time_limit=TIME_LIMIT_SECONDS)
    clique_sorted = sorted(clique)
    
    return clique_sorted


import json
def load_graph_from_json(path: str) -> dict[int, set[int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data

# ---------------------------
# Main: read input and run
# ---------------------------
def main():
    print(f"Reading Graph Data")
    
    adj_list = load_graph_from_json("../../results2/input_2025-11-02_17-57-20.json")
    # print("Loaded graph with", len(graph), "vertices")
    
    # INPUT_JSON = load_json_from_url("https://raw.githubusercontent.com/toptensor/CliqueAI/refs/heads/main/test_data/general_0.2.json")
    # adj_list = INPUT_JSON.get("adjacency_list")

    start_time = time.time()

    if adj_list is None:
        print("Input JSON must contain 'adjacency_list'. Exiting.")
        sys.exit(1)

    print(f"Calculating...")
    n, adj = graph_from_adjacency_list(adj_list)
    if n == 0:
        print("Empty graph. Max clique size = 0")
        return
    if n == 1:
        print("Single vertex. Max clique size = 1, clique = [0]")
        return

    print(f"Graph: n={n} vertices")
    start_time = time.time()
    size, clique = parallel_max_clique(n, adj, time_limit=TIME_LIMIT_SECONDS)
    duration = time.time() - start_time

    clique_sorted = sorted(clique)
    print(f"Time elapsed: {duration:.3f}s (limit {TIME_LIMIT_SECONDS}s)")
    print(f"Max clique size (best found): {size}")
    print(f"Max clique vertices (0-based indices): {clique_sorted}")

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.6f} seconds")


# def load_json_from_url(url: str):
#     """Download and parse JSON data from a remote URL."""
#     try:
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         data = response.json()
#         return data
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching JSON from {url}: {e}")
#         raise
    
if __name__ == "__main__":
    main()
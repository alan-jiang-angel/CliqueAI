# max_clique_timeout_safe.py
# Hard wall-time via multiprocessing with streaming best-so-far results.

import time, json, multiprocessing as mp
from typing import List, Tuple, Dict, Any, Iterable

# ===== Bit helpers =====
def popcount(x: int) -> int: return x.bit_count()
def bits_iter(x: int):
    while x:
        b = x & -x
        yield b.bit_length() - 1
        x ^= b

# ===== Loaders (adj list or 0/1 matrix) =====
def load_graph_from_json_data(data: Dict[str, Any], key="adjacency") -> List[int]:
    if key not in data or not isinstance(data[key], list):
        raise ValueError("JSON must contain a list under key='adjacency'.")
    adj_raw = data[key]
    n = len(adj_raw)
    is_matrix = all(isinstance(row, list) and len(row) == n and all(v in (0,1) for v in row) for row in adj_raw)
    # label_to_idx = None
    # if labels_key and labels_key in data:
    #     labels = data[labels_key]
    #     label_to_idx = {lab:i for i,lab in enumerate(labels)}
    adj = [0]*n
    if is_matrix:
        for u,row in enumerate(adj_raw):
            bs = 0
            for v,val in enumerate(row):
                if u!=v and val: bs |= 1<<v
            adj[u] = bs
    else:
        for u,nbrs in enumerate(adj_raw):
            bs = 0
            for nb in nbrs:
                v = int(nb)
                if v!=u: bs |= 1<<v
            adj[u] = bs
    # symmetrize
    for u in range(n):
        for v in bits_iter(adj[u]):
            adj[v] |= 1<<u
    return adj

# ===== Heuristics =====
def greedy_clique_lb(adj: List[int]) -> Tuple[int, List[int]]:
    n = len(adj)
    order = sorted(range(n), key=lambda v: popcount(adj[v]), reverse=True)
    best_sz, best = 0, []
    for s in order:
        C = [s]
        cand = adj[s]
        while cand:
            v = max(bits_iter(cand), key=lambda x: popcount(adj[x] & cand))
            C.append(v)
            cand &= adj[v]
        if len(C) > best_sz:
            best_sz, best = len(C), C
    return best_sz, best

def greedy_coloring_ub_complement(adj: List[int]) -> int:
    n = len(adj); all_mask = (1<<n)-1
    comp = [(all_mask ^ adj[v]) & ~(1<<v) for v in range(n)]
    vs = sorted(range(n), key=lambda v: popcount(comp[v]), reverse=True)
    colors = []
    for v in vs:
        for i,cls in enumerate(colors):
            if (comp[v] & cls) == 0:
                colors[i] |= 1<<v
                break
        else:
            colors.append(1<<v)
    return len(colors)

# ===== SAT backend (optional) =====
def _with_pysat():
    try:
        from pysat.formula import CNF
        from pysat.card import CardEnc, EncType
        from pysat.solvers import Minisat22
        return CNF, CardEnc, EncType, Minisat22
    except Exception:
        return None

def _build_cnf_geq(adj: List[int], k: int, CNF, CardEnc, EncType):
    n = len(adj)
    cnf = CNF()
    # non-edges
    for u in range(n):
        non_nbr = (~adj[u]) & ((1<<n)-1) & ~(1<<u)
        for v in bits_iter(non_nbr):
            if v > u:
                cnf.append([-(u+1), -(v+1)])
    if k > 0:
        atleast = CardEnc.atleast(lits=[i+1 for i in range(n)], bound=k, encoding=EncType.seqcounter)
        cnf.extend(atleast.clauses)
    return cnf

def _sat_has_streaming(adj: List[int], k: int, deadline: float):
    """Return (res, verts) with res in {True, False, None}. None = timeout."""
    pysat_pkg = _with_pysat()
    if pysat_pkg is None:
        return None, []
    CNF, CardEnc, EncType, Minisat22 = pysat_pkg
    cnf = _build_cnf_geq(adj, k, CNF, CardEnc, EncType)
    BUDGET = 50_000  # smaller budget = more responsive to timeout
    with Minisat22(bootstrap_with=cnf.clauses) as S:
        while True:
            if time.perf_counter() >= deadline:
                return None, []
            S.conf_budget(BUDGET)
            S.prop_budget(BUDGET)
            res = S.solve_limited(expect_interrupt=True)
            if res is True:
                model = S.get_model()
                chosen = [i-1 for i in model if i > 0 and 1 <= i <= len(adj)]
                chosen_set = set(chosen)
                # prune to a clique (safety)
                bad = True
                while bad:
                    bad = False
                    for u in list(chosen_set):
                        if any(((adj[u] >> v) & 1) == 0 for v in chosen_set if v != u):
                            chosen_set.remove(u); bad = True; break
                return True, sorted(chosen_set)
            elif res is False:
                return False, []
            # else: budget hit, loop again (timeout checked above)

# ===== Exact BnB (deadline-aware) =====
class _LRU:
    def __init__(self, cap=8192): self.cap=cap; self.d={}
    def get(self,k):
        v=self.d.get(k)
        if v is not None: self.d[k]=v
        return v
    def put(self,k,v):
        self.d[k]=v
        if len(self.d)>self.cap: self.d.pop(next(iter(self.d)))

def _color_order(cand:int, adj:List[int]):
    order = sorted(bits_iter(cand), key=lambda v: popcount(adj[v]&cand), reverse=True)
    colors=[]; col_of={}; maxc=0
    for v in order:
        for i,cls in enumerate(colors,1):
            if (adj[v] & cls)==0: colors[i-1]|=1<<v; col_of[v]=i; break
        else:
            colors.append(1<<v); maxc+=1; col_of[v]=maxc
    order.sort(key=lambda v: col_of[v], reverse=True)
    return order, maxc

def _choose_pivot(cand:int, adj:List[int])->int:
    best=-1; bd=-1
    for v in sorted(bits_iter(cand), key=lambda x: popcount(adj[x]&cand), reverse=True)[:32]:
        d=popcount(adj[v]&cand)
        if d>bd: bd, best = d, v
    return best

def _bnb_exact(adj: List[int], deadline: float, progress_cb=None) -> Tuple[int, int]:
    """Returns (size, mask). Calls progress_cb(size, mask) on improvement."""
    n=len(adj)
    if n==0:
        if progress_cb: progress_cb(0, 0)
        return 0,0
    best, best_set = 0, 0
    ic=_LRU()

    def isect(A:int,v:int)->int:
        k=(A,v); r=ic.get(k)
        if r is not None: return r
        r=A & adj[v]; ic.put(k,r); return r

    def expand(R:int,P:int):
        nonlocal best, best_set
        if time.perf_counter() >= deadline: return
        order,ub = _color_order(P,adj)
        if popcount(R)+ub <= best: return
        u=_choose_pivot(P,adj)
        NP=(P & ~adj[u]) if u!=-1 else P
        for v in list(order):
            if time.perf_counter() >= deadline: return
            if ((1<<v)&NP)==0: continue
            R2=R|(1<<v); P2=isect(P,v)
            if P2==0:
                sz=popcount(R2)
                if sz>best:
                    best,best_set=sz,R2
                    if progress_cb: progress_cb(best, best_set)
            else:
                expand(R2,P2)
            P &= ~(1<<v)
            if popcount(R)+popcount(P) <= best: return

    P=(1<<n)-1
    expand(0,P)
    return best, best_set

# ===== Child process worker: streams best-so-far to parent =====
def _worker(adj: List[int], deadline: float, q: mp.Queue):
    try:
        lb, lb_set = greedy_clique_lb(adj)
        q.put(("improve", lb, lb_set))  # stream LB immediately

        ub = max(lb, greedy_coloring_ub_complement(adj))
        best_size, best_verts = lb, lb_set[:]

        # SAT binary search, respecting child deadline
        pysat_ok = _with_pysat() is not None
        if pysat_ok:
            lo, hi = lb, ub
            while lo <= hi and time.perf_counter() < deadline:
                mid = (lo + hi + 1)//2
                res, verts = _sat_has_streaming(adj, mid, deadline)
                if res is None:  # timeout inside SAT
                    break
                if res is True and len(verts) >= mid:
                    best_size, best_verts = len(verts), verts
                    q.put(("improve", best_size, best_verts))
                    lo = mid + 1
                else:
                    hi = mid - 1

        # If time remains, run exact BnB to try to push further
        if time.perf_counter() < deadline:
            def cb(sz, mask):
                verts = [i for i in bits_iter(mask)]
                q.put(("improve", sz, verts))
            _bnb_exact(adj, deadline, progress_cb=cb)

        q.put(("done", best_size, best_verts))
    except Exception as e:
        q.put(("error", str(e), []))

# ===== Public API with hard timeout =====
def max_clique_clisat_style_timeout_safe(adj: List[int], time_limit: float = 30.0) -> Tuple[int, List[int]]:
    if len(adj) == 0:
        return 0, []
    deadline = time.perf_counter() + time_limit
    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_worker, args=(adj, deadline, q), daemon=True)
    p.start()

    best_size, best_verts = 0, []
    try:
        # Parent loop: collect improvements until timeout
        while time.perf_counter() < deadline:
            try:
                msg, a, b = q.get(timeout=max(0.0, deadline - time.perf_counter()))
            except Exception:
                break
            if msg == "improve":
                if a > best_size:
                    best_size, best_verts = a, b
            elif msg == "done":
                if a > best_size:
                    best_size, best_verts = a, b
                break
            elif msg == "error":
                # Return best-so-far even if error
                break
    finally:
        if p.is_alive():
            p.terminate()
        p.join(timeout=1.0)

    return best_size, sorted(best_verts)

def clisat_algorithm(number_of_nodes, graph, timeout=25):
    data_list = { "adjacency": graph }
    adj = load_graph_from_json_data(data_list)
    size, verts = max_clique_clisat_style_timeout_safe(adj, time_limit=timeout)
    return verts

import json
def load_graph_from_json(path: str) -> dict[int, set[int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

if __name__ == "__main__":
    graph = load_graph_from_json("../../results2/input_2025-11-02_17-57-20.json")
    print("Loaded graph with", len(graph), "vertices")

    t0 = time.time()

    verts = clisat_algorithm(len(graph), graph)

    print("Max clique size:", len(verts))
    print("Vertices:", verts)

    t1 = time.time()
    print(f"Elapsed: {t1-t0:.3f}s (limit 30s)")

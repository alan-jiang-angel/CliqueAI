from collections import OrderedDict
from typing import List, Tuple, Dict, Any, Iterable
import time, json

# -------- bit helpers --------
def popcount(x: int) -> int: return x.bit_count()
def bits_iter(x: int) -> Iterable[int]:
    while x:
        b = x & -x
        yield b.bit_length() - 1
        x ^= b

# -------- JSON → bitsets (list or 0/1 matrix; optional labels) --------
def load_graph_from_json_data(data: Dict[str, Any], key="adjacency") -> List[int]:
    if key not in data or not isinstance(data[key], list):
        raise ValueError("JSON must contain a list under key='adjacency'.")
    adj_raw = data[key]
    n = len(adj_raw)
    is_matrix = all(isinstance(row, list) and len(row) == n and all(v in (0,1) for v in row) for row in adj_raw)

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

# -------- induce subgraph --------
def induce(adj: List[int], vertices: List[int]) -> Tuple[List[int], Dict[int,int], List[int]]:
    m = len(vertices)
    idx = {v:i for i,v in enumerate(vertices)}
    mask = 0
    for v in vertices: mask |= 1<<v
    sub = [0]*m
    for i,v in enumerate(vertices):
        av_masked = adj[v] & mask
        bs = 0
        for u in bits_iter(av_masked):
            bs |= 1 << idx[u]
        sub[i] = bs
    return sub, idx, vertices  # vertices == new->old mapping

# -------- core numbers (degeneracy) --------
def core_numbers(adj: List[int]) -> List[int]:
    n = len(adj)
    if n == 0: return []
    deg = [popcount(adj[v]) for v in range(n)]
    maxd = max(deg)
    bins = [0]*(maxd+1)
    for d in deg: bins[d]+=1
    start = [0]*(maxd+1); s=0
    for d in range(maxd+1): start[d], s = s, s+bins[d]
    pos=[0]*n; vert=[0]*n; nxt=start[:]
    for v in range(n):
        d=deg[v]; pos[v]=nxt[d]; vert[pos[v]]=v; nxt[d]+=1
    core=deg[:]
    for i in range(n):
        v=vert[i]
        for u in bits_iter(adj[v]):
            if core[u] > core[v]:
                du=core[u]; pu=pos[u]; pw=start[du]; w=vert[pw]
                if u!=w:
                    vert[pw],vert[pu]=vert[pu],vert[pw]
                    pos[u],pos[w]=pw,pu
                start[du]+=1; core[u]-=1
    return core

# -------- BnB (deadline-based timer) --------
class IntersectCache:
    def __init__(self, cap=8192): self.cap, self._d = cap, OrderedDict()
    def get(self,k):
        v=self._d.get(k)
        if v is not None: self._d.move_to_end(k)
        return v
    def put(self,k,v):
        self._d[k]=v; self._d.move_to_end(k)
        if len(self._d)>self.cap: self._d.popitem(last=False)

def color_order(cand:int, adj:List[int])->Tuple[List[int],int]:
    order = sorted(bits_iter(cand), key=lambda v: popcount(adj[v]&cand), reverse=True)
    colors=[]; col_of={}; maxc=0
    for v in order:
        for i,cls in enumerate(colors,1):
            if (adj[v] & cls)==0:
                colors[i-1] |= 1<<v; col_of[v]=i; break
        else:
            colors.append(1<<v); maxc+=1; col_of[v]=maxc
    order.sort(key=lambda v: col_of[v], reverse=True)
    return order, maxc

def choose_pivot(cand:int, adj:List[int])->int:
    best=-1; bd=-1
    for v in sorted(bits_iter(cand), key=lambda x: popcount(adj[x]&cand), reverse=True)[:32]:
        d=popcount(adj[v]&cand)
        if d>bd: bd, best = d, v
    return best

class CliqueBnB:
    def __init__(self, adj: List[int], deadline: float, lb: int = 1):
        self.adj=adj; self.best=lb; self.best_set=0
        self.deadline=deadline; self.ic=IntersectCache()
    def timeout(self)->bool: return time.perf_counter() >= self.deadline
    def isect(self,A:int,v:int)->int:
        k=(A,v); hit=self.ic.get(k)
        if hit is not None: return hit
        r=A & self.adj[v]; self.ic.put(k,r); return r
    def expand(self,R:int,P:int):
        if self.timeout(): return
        order,ub=color_order(P,self.adj)
        if popcount(R)+ub<=self.best: return
        u=choose_pivot(P,self.adj)
        NP = (P & ~self.adj[u]) if u!=-1 else P
        for v in list(order):
            if self.timeout(): return
            if ((1<<v)&NP)==0: continue
            R2=R|(1<<v); P2=self.isect(P,v)
            if P2==0:
                sz=popcount(R2)
                if sz>self.best: self.best, self.best_set = sz, R2
            else:
                self.expand(R2,P2)
            P &= ~(1<<v)
            if popcount(R)+popcount(P) <= self.best: return

# -------- greedy LB that RETURNS a clique mask --------
def greedy_lb_with_mask(adj: List[int]) -> Tuple[int, int]:
    n=len(adj); best_size=0; best_mask=0
    for start in range(n):
        C=0; cand=(1<<n)-1; v=start
        while True:
            C |= 1<<v
            cand &= adj[v]
            if cand==0: break
            v = max(bits_iter(cand), key=lambda x: popcount(adj[x]&cand))
        sz = popcount(C)
        if sz > best_size:
            best_size, best_mask = sz, C
    return best_size, best_mask

# -------- CUBIS-inspired driver --------
def max_clique_cubis(adj: List[int], time_limit: float = 30.0) -> Tuple[int, List[int]]:
    n=len(adj)
    if n==0: return 0, []
    t_start = time.perf_counter()
    deadline = t_start + time_limit

    # Reserve some time for final extraction to ensure verts are returned
    reserve = max(0.5, 0.1 * time_limit)
    search_deadline = deadline - reserve

    # 1) global greedy LB with mask (on full graph!)
    lb, lb_mask_global = greedy_lb_with_mask(adj)

    # 2) degree pre-prune using LB-1, but keep LB clique we already have
    active = [v for v in range(n) if popcount(adj[v]) >= max(0, lb-1)]
    if len(active) < n:
        adj_sub, _, map_back = induce(adj, active)
    else:
        adj_sub, map_back = adj[:], list(range(n))

    # 3) core numbers + UB
    core = core_numbers(adj_sub)
    degeneracy = max(core) if core else 0
    ub = degeneracy + 1

    # Track best (start from greedy LB)
    best_size = lb
    best_mask_global = lb_mask_global  # already in global index space

    # If little search time remains, jump to extraction
    if time.perf_counter() >= search_deadline or best_size >= ub:
        # ensure we have a concrete clique (we do: lb_mask_global)
        return best_size, [i for i in bits_iter(best_mask_global)]

    # ---- CUBIS search phase (uses search_deadline) ----
    k = ub
    while k > best_size and time.perf_counter() < search_deadline:
        # seeds with core >= k-1
        seeds = [i for i,c in enumerate(core) if c >= (k-1)]
        seeds.sort(key=lambda v: (core[v], popcount(adj_sub[v])), reverse=True)
        budget = min(len(seeds), 64)

        # precompute mask of core≥k-1
        S_mask = 0
        for v in range(len(adj_sub)):
            if core[v] >= (k-1): S_mask |= 1<<v

        def prune_to_kminus1(mask:int)->int:
            changed=True
            while changed:
                changed=False
                for v in list(bits_iter(mask)):
                    if popcount(adj_sub[v] & mask) < (k-1):
                        mask &= ~(1<<v); changed=True
            return mask

        for s in seeds[:budget]:
            if time.perf_counter() >= search_deadline: break
            neigh = adj_sub[s] & S_mask
            cubis1 = (1<<s) | neigh
            cubis2 = prune_to_kminus1(cubis1)

            for cubis_mask in (cubis1, cubis2):
                if time.perf_counter() >= search_deadline: break
                if popcount(cubis_mask) < k: continue
                verts_sub = list(bits_iter(cubis_mask))
                sub_adj, _, back = induce(adj_sub, verts_sub)
                P = (1<<len(verts_sub)) - 1
                bnb = CliqueBnB(sub_adj, search_deadline, lb=best_size)
                bnb.expand(0, P)
                if bnb.best > best_size and bnb.best_set != 0:
                    best_size = bnb.best
                    # map subgraph solution → adj_sub → global
                    mask_global = 0
                    for i in bits_iter(bnb.best_set):
                        v_sub = back[i]          # index in adj_sub
                        v_g   = map_back[v_sub]  # index in original graph
                        mask_global |= 1 << v_g
                    best_mask_global = mask_global
        k -= 1

    # ---- Final extraction (use reserved time) ----
    # If we improved size but somehow didn't capture members, re-run a short BnB on pruned graph
    if best_mask_global == 0 and best_size > 0:
        P=(1<<len(adj_sub))-1
        bnb = CliqueBnB(adj_sub, deadline, lb=best_size-1)
        bnb.expand(0,P)
        mask_global = 0
        for i in bits_iter(bnb.best_set):
            mask_global |= 1 << map_back[i]
        if mask_global != 0:
            best_mask_global = mask_global
        else:
            # As a last resort, return the greedy LB clique we had
            best_mask_global = lb_mask_global

    return best_size, [i for i in bits_iter(best_mask_global)]

def cubis_lb_algorithm(number_of_nodes, graph):
    data_list = { "adjacency": graph }
    adj = load_graph_from_json_data(data_list)
    size, verts = max_clique_cubis(adj, time_limit=25)
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

    verts = cubis_lb_algorithm(len(graph), graph)

    print("Max clique size:", len(verts))
    print("Vertices:", verts)

    t1 = time.time()
    print(f"Elapsed: {t1-t0:.3f}s (limit 30s)")

# bk_core.pyx
# Cython implementation of Bron-Kerbosch with greedy coloring bound (bitset ints).
# Compiles to a Python extension for fast inner search.

from cpython cimport bool as cbool
cimport cython
from libc.time cimport time as c_time, time_t

@cython.boundscheck(False)
@cython.wraparound(False)
def greedy_color_order_cy(long cand_bits, adj):
    """
    Greedy coloring on candidate set cand_bits (Python int).
    adj: Python list of ints.
    Returns (order_list, color_list)
    """
    cdef list verts = []
    cdef long b = cand_bits
    while b:
        lsb = b & -b
        v = lsb.bit_length() - 1
        verts.append(v)
        b &= b - 1

    # greedy color
    cdef list order_res = []
    cdef list color_res = []
    cdef set uncolored = set(verts)
    cdef int color = 0
    while uncolored:
        color += 1
        forbidden = 0
        # iterate snapshot
        for v in list(uncolored):
            if forbidden & (1 << v):
                continue
            order_res.append(v)
            color_res.append(color)
            forbidden |= adj[v]
            uncolored.remove(v)
    return order_res, color_res

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def bron_kerbosch_max_cy(int n, object adj_py, double time_limit, int initial_bound=0):
    """
    Find maximum clique in graph with n vertices and adjacency list adj_py (Python list of int bitsets).
    Returns (best_size:int, best_clique:list_of_ints).
    This function is intentionally implemented with Python ints for bitsets but in Cython for speed.
    """
    cdef time_t now
    cdef double t0 = c_time(&now)
    cdef double deadline = t0 + time_limit
    cdef object adj = adj_py  # list of ints
    cdef list best_clique = []
    cdef int best_size = initial_bound

    # ordering: degree-desc (simple heuristic)
    cdef list degs = [0] * n
    for i in range(n):
        degs[i] = (adj[i]).bit_count()
    order = sorted(range(n), key=lambda x: degs[x], reverse=True)
    ordering_pos = {v:i for i,v in enumerate(order)}

    cdef long nodes_searched = 0
    cdef int TIME_CHECK_INTERVAL = 2048

    # define recursive search
    def search(long cand_bits, list cur_clique):
        cdef time_t now1
        cdef double t10 = c_time(&now1)

        nonlocal best_clique, best_size, nodes_searched
        nodes_searched += 1
        if (nodes_searched & (TIME_CHECK_INTERVAL - 1)) == 0:
            if t10 > deadline:
                raise TimeoutError
        if cand_bits == 0:
            if len(cur_clique) > best_size:
                best_size = len(cur_clique)
                best_clique = cur_clique.copy()
            return

        order_list, colors = greedy_color_order_cy(cand_bits, adj)
        # iterate in reverse for stronger pruning
        idx = len(order_list) - 1
        while idx >= 0:
            v = order_list[idx]
            col = colors[idx]
            idx -= 1
            if len(cur_clique) + col <= best_size:
                return
            # include v
            cur_clique.append(v)
            new_cand = cand_bits & adj[v]
            if new_cand:
                search(new_cand, cur_clique)
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

    try:
        all_bits = (1 << n) - 1
        search(all_bits, [])
    except TimeoutError:
        # timeout: return best-so-far
        pass

    return best_size, best_clique

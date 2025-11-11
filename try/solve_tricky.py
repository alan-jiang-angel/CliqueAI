# Use previous method for <= 100
# Use previous method + 5X new method and return better one <= 300
# Use new method for 10X times and return best one <= 500

import os
import json
from pathlib import Path
import time

from CliqueAI.clique_algorithms.cython.cubis import cubis_driver
from CliqueAI.clique_algorithms.clisat_algorithm import clisat_algorithm
# from CliqueAI.clique_algorithms.bron_kerbosch_algorithm import parallel_max_clique, graph_from_adjacency_list
from CliqueAI.clique_algorithms.greedy_expansion_algorithm import greedy_expansion_algorithm

def read_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def hybrid_algorithm(number_of_nodes, graph):
    # number_of_nodes = len(graph)
    print(f"🔹 Number of vertics in graph: {number_of_nodes}")
    result = []

    t0 = time.time()
    if (number_of_nodes <= 100):
        # Hard Code
        clique = clisat_algorithm(number_of_nodes, graph)
        clique2 = greedy_expansion_algorithm(number_of_nodes, graph)
        
        if (len(clique2) > len(clique)):
            clique = clique2
        # n, adj = graph_from_adjacency_list(graph)
        # size, clique = parallel_max_clique(n, adj, time_limit=25)
    # elif (number_of_nodes <= 300):
        # clique = greedy_expansion_algorithm(number_of_nodes, graph)
        # clique = clisat_algorithm(number_of_nodes, graph, 23)
        # size = len(clique)

        # while time.time() - t0 < 27:
        #     size_new, result_new = cubis_driver(graph, 25, True)
        #     if size_new > size:
        #         size = size_new
        #         clique = result_new
        #         print('found better one from cubis:', size)
    else:
        clique = greedy_expansion_algorithm(number_of_nodes, graph)
        size = len(clique)

        for _ in range(100):
            size_new, result_new = cubis_driver(graph, 25, True)
            if size_new > size:
                size = size_new
                clique = result_new
                print('found better one from cubis:', size)

    result = sorted(clique)
    return result

def main():
    input_file = "../test_data/level-2-41.json"
    print(f"🔹 Processing {input_file}")

    # Read input JSON
    input_data = read_json(input_file)
    
    t0 = time.time()
    # Execute dummy function
    result = hybrid_algorithm(len(input_data), input_data)
    t1 = time.time() - t0
    print(f"⏱️ Time taken: {t1:.2f} seconds")

    print(f"🔹 Result clique size: {len(result)}")


if __name__ == "__main__":
    main()
# Use previous method for <= 100
# Use previous method + 5X new method and return better one <= 300
# Use new method for 10X times and return best one <= 500

import os
import json
from pathlib import Path
import time

# from CliqueAI.clique_algorithms.cython.cubis import cubis_driver
# from CliqueAI.clique_algorithms.clisat_algorithm import clisat_algorithm
# from CliqueAI.clique_algorithms.bron_kerbosch_algorithm import parallel_max_clique, graph_from_adjacency_list
# from CliqueAI.clique_algorithms.greedy_expansion_algorithm import greedy_expansion_algorithm
from CliqueAI.clique_algorithms.greedy_exp_rand_algorithm import greedy_exp_rand_algorithm

def read_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def hybrid_algorithm(number_of_nodes, graph):
    # number_of_nodes = len(graph)
    print(f"🔹 Number of vertics in graph: {number_of_nodes}")
    result = []

    # t0 = time.time()
    # if (number_of_nodes <= 100):
    #     # Hard Code
    #     clique = clisat_algorithm(number_of_nodes, graph)
    #     clique2 = greedy_expansion_algorithm(number_of_nodes, graph)
        
    #     if (len(clique2) > len(clique)):
    #         clique = clique2
    #     # n, adj = graph_from_adjacency_list(graph)
    #     # size, clique = parallel_max_clique(n, adj, time_limit=25)
    # # elif (number_of_nodes <= 300):
    #     # clique = greedy_expansion_algorithm(number_of_nodes, graph)
    #     # clique = clisat_algorithm(number_of_nodes, graph, 23)
    #     # size = len(clique)

    #     # while time.time() - t0 < 27:
    #     #     size_new, result_new = cubis_driver(graph, 25, True)
    #     #     if size_new > size:
    #     #         size = size_new
    #     #         clique = result_new
    #     #         print('found better one from cubis:', size)
    # else:
    #     clique = greedy_expansion_algorithm(number_of_nodes, graph)
    #     size = len(clique)

    #     # while time.time() - t0 < 25:
    #     for _ in range(10):
    #         size_new, result_new = cubis_driver(graph, 25, True)
    #         if size_new > size:
    #             size = size_new
    #             clique = result_new
    #             print('found better one from cubis:', size)

    # clique = 
    # result = sorted(clique)
    return greedy_exp_rand_algorithm(number_of_nodes, graph)

def main():
    base_path = Path("../../results0208")
    input_files = sorted(base_path.glob("input_*.json"))

    better_vertics = []
    poor_vertics = []
    
    better_count = 0
    poor_count = 0
    
    for input_file in input_files:
        # Extract timestamp suffix (everything after 'input')
        suffix = input_file.stem.replace("input_", "", 1)
        output_file = base_path / f"result_{suffix}.json"

        print(f"✅ Match for {suffix} ---")

        if not output_file.exists():
            print(f"⚠️ No matching output file for: {input_file.name}")
            continue

        print(f"🔹 Processing {input_file.name} ↔ {output_file.name}")

        # Read input JSON
        input_data = read_json(input_file)
        
        if (len(input_data) <= 100):
            continue
        # elif (len(input_data) > 300):
        #     continue

        t0 = time.time()
        # Execute dummy function
        result = hybrid_algorithm(len(input_data), input_data)
        t1 = time.time() - t0
        print(f"⏱️ Time taken: {t1:.2f} seconds")
        
        # Read expected output JSON
        expected_output = read_json(output_file)
        
        if (len(result) > len(expected_output)):
            print(f"🔥 Better result! New: {len(result)}, Expected: {len(expected_output)}")
            better_count += 1
            better_vertics.append(len(input_data))
        elif (len(result) < len(expected_output)):
            print(f"❌ Poor! New: {len(result)}, Expected: {len(expected_output)}")
            poor_count += 1
            poor_vertics.append(len(input_data))
        else:
            print(f"✅ Equal: {len(result)}")

        # time.sleep(1)
        print("--------------------------------------------------\n")
        
    print(f"Summary: Better: {better_count}, Poor: {poor_count}")
    print(better_vertics)
    print(poor_vertics)

if __name__ == "__main__":
    main()
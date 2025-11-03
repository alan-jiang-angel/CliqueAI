# Use previous method for <= 100
# Use previous method + 5X new method and return better one <= 300
# Use new method for 10X times and return best one <= 500

import os
import json
from pathlib import Path
import time

from CliqueAI.clique_algorithms.cython.cubis import cubis_driver
from CliqueAI.clique_algorithms.bron_kerbosch_algorithm import parallel_max_clique, graph_from_adjacency_list

def read_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def hybrid_algorithm(number_of_nodes, graph):
    # number_of_nodes = len(graph)
    print(f"🔹 Number of vertics in graph: {number_of_nodes}")
    result = []

    if (number_of_nodes <= 100):
        # Hard Code
        n, adj = graph_from_adjacency_list(graph)
        size, clique = parallel_max_clique(n, adj, time_limit=25)
        result = sorted(clique)
    elif (number_of_nodes <= 300):
        n, adj = graph_from_adjacency_list(graph)
        size, clique = parallel_max_clique(n, adj, time_limit=20)
        result = sorted(clique)

        for i in range(5):
            size_cubis, result_cubis = cubis_driver(graph, 25, True)
            if size_cubis > size:
                size = size_cubis
                result = result_cubis
    else:
        size, result = cubis_driver(graph, 25, True)
        for i in range(20):
            # CUBIS
            size_new, result_new = cubis_driver(graph, 25, True)
            if size_new > size:
                size = size_new
                result = result_new

    return result

def main():
    base_path = Path("../../results")
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
            print("Skip --------------------------------------------------\n")
            continue

        # Execute dummy function
        result = hybrid_algorithm(input_data)

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
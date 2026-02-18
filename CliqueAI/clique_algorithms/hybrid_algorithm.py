# Use previous method for <= 100
# Use previous method + 5X new method and return better one <= 300
# Use new method for 10X times and return best one <= 500

import os
import json
import time
import subprocess
import redis
from pathlib import Path
from CliqueAI.clique_algorithms.greedy_exp_rand_algorithm import greedy_exp_rand_algorithm
import redis

INDEXER_PATH = "./graph-iso/indexer"    # Your compiled bliss indexer
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
BASE_PATH = Path("./results")

r = redis.Redis()

def read_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def run_indexer(filepath):
    try:
        result = subprocess.run(
            [INDEXER_PATH, filepath],
            capture_output=True,
            text=True,
            check=True,
            timeout=60
        )

        lines = result.stdout.strip().splitlines()

        if len(lines) < 2:
            raise ValueError("Unexpected indexer output format")

        # First line = SHA256
        hash_val = lines[0].strip()

        # Second line = PERM: ...
        if not lines[1].startswith("PERM:"):
            raise ValueError("Missing PERM line")

        perm_str = lines[1].split("PERM:", 1)[1].strip()

        # Convert hex → decimal integers
        perm_list = [int(x, 16) for x in perm_str.split()]

        # Store permutation as space-separated decimal string
        perm_serialized = " ".join(map(str, perm_list))

        return hash_val, perm_serialized

    except Exception as e:
        print(f"[ERROR] {filepath}: {e}")
        return None


def hybrid_algorithm(number_of_nodes, graph, timestamp):
    print(f"🔹 Number of vertics in graph: {number_of_nodes}")

    # 1. Read Input Graph
    # 2. Calculate Hash and CP (Canonical Permutation)
    filepath = BASE_PATH / f"input_{timestamp}.json"
    result = run_indexer(filepath)
    if not result:
        print(f"❌ Failed to run indexer on {filepath}")
        return greedy_exp_rand_algorithm(number_of_nodes, graph)

    hash_val, perm = result

    graph_key = f"graph:{hash_val}"
    dup_key = f"dups:{hash_val}"
    file_key = f"file:{filepath}"
    file_name = f"{filepath}"

    # 3. Hash exists on Redis
    graph_info = r.hgetall(graph_key)  # Check if graph exists in Redis

    if not graph_info:
        print(f"❌ Graph not found in Redis for hash: {hash_val}")
        clique = greedy_exp_rand_algorithm(number_of_nodes, graph)

        pipe = r.pipeline()

        # Store canonical permutation only once
        pipe.hsetnx(graph_key, "perm", perm)
        pipe.hsetnx(graph_key, "cliquelen", len(clique))

        # Add filename to duplicate set
        pipe.sadd(dup_key, file_name)

        # Reverse lookup
        pipe.set(file_key, hash_val)

        pipe.execute()

        return clique

    print(f"✅ Graph found in Redis for hash: {hash_val}")
    existing_perm = graph_info.get(b'perm', b'').decode('utf-8')
    mapping = compute_mapping(list(map(int, perm.split())), list(map(int, existing_perm.split())))
    print(f"🔹 Mapping calculated: {mapping}")
    
    output_file = BASE_PATH / f"result_{timestamp}.json"
    output_data = read_json(output_file)
    
    if not output_data:
        print(f"❌ Failed to read output data from {output_file}")
        return greedy_exp_rand_algorithm(number_of_nodes, graph)
    
    mapped_clique = [mapping[node] for node in output_data]
    print(f"🔹 Original clique: {output_data}")
    print(f"🔹 Mapped clique: {mapped_clique}")

    return mapped_clique


def compute_mapping(permA, permB):
    n = len(permA)

    invA = [0] * n
    for i in range(n):
        invA[permA[i]] = i

    mapping = [0] * n
    for original_node in range(n):
        canonical_index = invA[original_node]
        mapping[original_node] = permB[canonical_index]

    return mapping


def main():
    input_files = sorted(BASE_PATH.glob("input_*.json"))

    better_vertics = []
    poor_vertics = []

    better_count = 0
    poor_count = 0

    for input_file in input_files:
        # Extract timestamp suffix (everything after 'input')
        suffix = input_file.stem.replace("input_", "", 1)
        output_file = BASE_PATH / f"result_{suffix}.json"

        print(f"✅ Match for {suffix} ---")

        if not output_file.exists():
            print(f"⚠️ No matching output file for: {input_file.name}")
            continue

        print(f"🔹 Processing {input_file.name} ↔ {output_file.name}")

        # Read input JSON
        input_data = read_json(input_file)

        # if (len(input_data) <= 500):
            # continue
        # elif (len(input_data) > 300):
            # continue

        t0 = time.time()
        # Execute dummy function
        result = hybrid_algorithm(len(input_data), input_data, suffix)
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
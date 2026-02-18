# Same with Hybrid_algorithm, except main function that runs through all input files and compares results with expected output. This is for testing the whole pipeline (indexer + hybrid algorithm) on all test cases.

import os
import json
import time
import subprocess
import redis
from pathlib import Path
from CliqueAI.clique_algorithms.greedy_exp_rand_algorithm import greedy_exp_rand_algorithm

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
        # if not lines[1].startswith("PERM:"):
        #     raise ValueError("Missing PERM line")

        # perm_str = lines[1].split("PERM:", 1)[1].strip()

        # # Convert hex → decimal integers
        # perm_list = [int(x, 16) for x in perm_str.split()]

        # # Store permutation as space-separated decimal string
        # perm_serialized = " ".join(map(str, perm_list))

        return hash_val # , perm_serialized

    except Exception as e:
        print(f"[ERROR] {filepath}: {e}")
        return None


def try_become_leader(hash):
    LOCK_KEY = f"job:{hash}"
    DONE_KEY = f"job:{hash}:done"
    return r.set(LOCK_KEY, hash, nx=True, ex=30)


def do_work(hash, number_of_nodes, graph, timestamp):
    LOCK_KEY = f"job:{hash}"
    DONE_KEY = f"job:{hash}:done"
    print(f"[{hash}] Acting as leader...")

    clique = greedy_exp_rand_algorithm(number_of_nodes, graph)

    output_file = BASE_PATH / f"result_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(clique, f, indent=2)

    r.publish(DONE_KEY, "done")
    r.delete(LOCK_KEY)
    print(f"[{hash}] Done.")


def wait_for_result(hash, timestamp):
    LOCK_KEY = f"job:{hash}"
    DONE_KEY = f"job:{hash}:done"
    print(f"[{hash}] Waiting for result...")

    # if file already exists, no need to wait
    output_file = BASE_PATH / f"result_{timestamp}.json"
    if os.path.exists(output_file):
        return

    pubsub = r.pubsub()
    pubsub.subscribe(DONE_KEY)

    # double-check after subscribe (avoid race condition)
    if os.path.exists(output_file):
        return

    for message in pubsub.listen():
        if message["type"] == "message":
            break


def hybrid_algorithm(number_of_nodes, graph, timestamp):
    print(f"🔹 Number of vertics in graph: {number_of_nodes}")

    filepath = BASE_PATH / f"input_{timestamp}.json"
    result = run_indexer(filepath)
    if not result:
        print(f"❌ Failed to run indexer on {filepath}")
        return greedy_exp_rand_algorithm(number_of_nodes, graph)

    hash_val = result

    if try_become_leader(hash_val):
        do_work(hash_val, number_of_nodes, graph, timestamp)
    else:
        wait_for_result(hash_val, timestamp)

    output_file = BASE_PATH / f"result_{timestamp}.json"
    clique = read_json(output_file)
    
    print(f"Read and returning clique from {output_file}")

    return clique


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
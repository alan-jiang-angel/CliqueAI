import os
import re
import subprocess
import redis
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================

DATA_DIR = "/root/works/CliqueAI/old_data/"          # Folder containing json files
INDEXER_PATH = "./indexer"    # Your compiled bliss indexer
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

FILENAME_PATTERN = re.compile(
    r"^input_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.json$"
)

# =========================
# REDIS
# =========================

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

# =========================
# RUN INDEXER
# =========================
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

# =========================
# MAIN LOOP
# =========================

def main():
    files = sorted([
        f for f in os.listdir(DATA_DIR)
        if FILENAME_PATTERN.match(f)
    ])

    print(f"Found {len(files)} files")

    for filename in tqdm(files):
        filepath = os.path.join(DATA_DIR, filename)

        result = run_indexer(filepath)
        if not result:
            continue

        hash_val, perm = result

        graph_key = f"graph:{hash_val}"
        dup_key = f"dups:{hash_val}"
        file_key = f"file:{filename}"

        pipe = r.pipeline()

        # Store canonical permutation only once
        pipe.hsetnx(graph_key, "perm", perm)

        # Add filename to duplicate set
        pipe.sadd(dup_key, filename)

        # Reverse lookup
        pipe.set(file_key, hash_val)

        pipe.execute()

    print("Ingestion complete.")

if __name__ == "__main__":
    main()

import wandb
import pandas as pd
import datetime
import time

UIDS = [177, 172, 161, 181, 107]

# ff3cc6ddf11ad82a38910e0750e8a803014cf51d
# authenticate
wandb.login()  # or set WANDB_API_KEY env var

api = wandb.Api()

def getRunInfo(run_id):
    run = api.run(f"{entity}/{project}/{run_id}")

    # --- BASIC RUN INFO ---
    # print("🔹 Basic Info")
    # print("Name:", run.name)
    # print("ID:", run.id)
    # print("State:", run.state)

    if run.state == 'running':
        for uid in UIDS:
            try:
                idx = run.summary.miner_uids.index(uid)
            except:
                continue

            my_answer = len(run.summary.miner_ans[idx])
            dt = datetime.datetime.fromtimestamp(run.summary.timestamp)   # local time

            better_count = 0
            better_answer = []
            
            for a in run.summary.miner_ans:
                if len(a) > my_answer:
                    better_count += 1
                    better_answer = a
            
            if better_count > 0:
                print(f"❌ {dt} Uid: {run.summary.miner_uids[idx]}, difficulty: {run.summary.difficulty}, Answer: {my_answer}, Reward: {run.summary.miner_rewards[idx]}, RID: {run.id}")
                print(f"- {better_count} are better than me out of {len(run.summary.miner_ans)} miners")
                print(run.summary.adjacency_list.value)
                print(my_answer, run.summary.miner_ans[idx])
                better_answer.sort()
                print(len(better_answer), better_answer)
                print()
            else:
                print(f"✅ {dt} Uid: {run.summary.miner_uids[idx]}, difficulty: {run.summary.difficulty}, Answer: {my_answer}, Reward: {run.summary.miner_rewards[idx]}")
                print()

# specify entity/project/run
entity = "toptensor-ai"
project = "CliqueAI"
# run_id = "5Fh1YDPcJ2Bs6JHfE5Ck1gCwvPxgkK32ZpHy94oy9XKMEyh9"
while(True):
    runs = api.runs(f"{entity}/{project}")

    # Print basic info
    for run in runs:
        # print(f"Run ID: {run.id}")
        # print(f"Name: {run.name}")
        # print(f"State: {run.state}")
        # print(f"Created at: {run.created_at}")
        getRunInfo(run.id)
        
    print("-" * 60)
    time.sleep(10)
    
# # list artifacts for that run
# artifacts = run.logged_artifacts()
# for art in artifacts:
#     print(art.name, art.type, art.version)

# # assuming you find an artifact with the table
# artifact = api.artifact(f"{entity}/{project}/{artifact_name}:{version}")
# artifact_dir = artifact.download()

# # locate the table file (e.g., table_name.table.json)
# import json
# with open(f"{artifact_dir}/{table_name}.table.json") as f:
#     table_json = json.load(f)

# df = pd.DataFrame(table_json["data"], columns=table_json["columns"])
# print(df.head())
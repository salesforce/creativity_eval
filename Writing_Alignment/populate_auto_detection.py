from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools, tqdm, json, argparse, os, fcntl, random
from utils_generate_edits import run_error_detection


# dataset_fn = "data/span_detection_experiment_v0.3.json"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_fn", type=str, default="all_finegrained_clean.json")
parser.add_argument("--model", type=str, default="all")
parser.add_argument("--N_workers", type=int, default=5)
args = parser.parse_args()

dataset_fn = args.dataset_fn

prompts = {"v2-fs25": "prompts/detection_v2_fs25.txt", "v2-fs5": "prompts/detection_v2_fs5.txt", "v2-fs2": "prompts/detection_v2_fs2.txt"} # "v2-fs5": "prompts/detection_v2_fs5.txt", "v1-fs5": "prompts/detection_v1_fs5.txt", "v1-fs25": "prompts/detection_v1_fs25.txt"
llms = [args.model]
if args.model == "all":
    llms = ["gpt-4o", "gpt-4o-mini"] # , "llama3.1-70b", "llama3.1-8b", "claude3-haiku", "claude3.5-sonnet"

with open(dataset_fn, "r") as f:
    data = json.load(f)

data = [d for d in data if d["split"] == "test"]

def thread_safe_write(file_path, data):
    with open(file_path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(data) + "\n")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

prompt_ids = list(prompts.keys())
random.shuffle(prompt_ids)

for prompt_id, llm in itertools.product(prompt_ids, llms):
    out_fn = f"data/detection_preds/{llm}_{prompt_id}.jsonl"

    already_samples = []
    if os.path.exists(out_fn):
        with open(out_fn, "r") as f:
            for line in f:
                already_samples.append(json.loads(line))

    already_ids = set([d["id"] for d in already_samples])

    # pop_key = f"pred_{llm}_{prompt_id}"

    todos = [d for d in data if d["id"] not in already_ids]
    if len(todos) == 0:
        continue

    random.shuffle(todos)

    # for d in tqdm.tqdm(todos):
    #     output = run_error_detection(d["preedit"], llm_engine=llm, temperature=0.0, prompt_fn=prompts[prompt_id])
    #     thread_safe_write(out_fn, {"id": d["id"], "detection": output})

    N_workers = 1 if "gemini-1.5-pro" in llm else args.N_workers

    with ThreadPoolExecutor(max_workers=N_workers) as executor:
        futures = {executor.submit(run_error_detection, d["preedit"], llm_engine=llm, temperature=0.0, prompt_fn=prompts[prompt_id]): d for d in todos}
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc=f"{llm} {prompt_id}"):
            d = futures[future]
            output = future.result()
            thread_safe_write(out_fn, {"id": d["id"], "detection": output})

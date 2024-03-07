import random, os, tqdm, json, sys
import argparse

# Note: Replace this with any path to your favorite LLM API
sys.path.insert(0, "/export/home/model_utils")
from model_bank import get_model_fns


# get the model through argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="claudev21")
args = parser.parse_args()

model = args.model

model_fn, model_kwargs = get_model_fns(model)

already_ids = set()
annotation_fn = "data/annotations_%s.jsonl" % model
if os.path.exists(annotation_fn):
    with open(annotation_fn, "r") as f:
        already_annotations = [json.loads(line) for line in f]
        already_ids = set([a["id"] for a in already_annotations])

with open("data/crea_eval_dataset.json", "r") as f:
    dataset = json.load(f)

todos = [s for s in dataset if s["id"] not in already_ids]
random.shuffle(todos)

with open("prompts/with_background.txt", "r") as f:
    prompt = f.read().strip()

ite = tqdm.tqdm(todos)
for sample in ite:
    prompt_populated = prompt.replace("[STORY]", sample["story"]).replace("[BACKGROUND]", sample["expanded_context"]).replace("[QUESTION]", sample["question"])
    response = model_fn([{"role": "user", "content": prompt_populated}], **model_kwargs)

    with open(annotation_fn, "a") as f:
        f.write(json.dumps({"id": sample["id"], "response": response}) + "\n")
    ite.set_description("Label: %s; Response: %s" % (sample["label"], response[:3]))

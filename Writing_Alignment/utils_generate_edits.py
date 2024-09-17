from collections import Counter
import anyllm, json, utils_diff

categories = ['Awkward Word Choice and Phrasing', 'Cliche', 'Lack of Specificity and Detail', 'Poor Sentence Structure', 'Punctuation', 'Purple Prose', 'Tense Inconsistency', 'Unnecessary/Redundant Exposition', 'Capitalization', "Factuality"]
cat2clean = {k: k.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "") for k in categories}
cat2clean["all"] = "all"

def run_error_detection(paragraph, llm_engine="claude3.5-sonnet", prompt_fn="prompts/detection_v1.txt", temperature=0.0, printing=False):
    with open(prompt_fn, "r") as f:
        prompt = f.read()

    populated_prompt = prompt.replace("[[PARAGRAPH]]", paragraph)
    messages = [{"role": "user", "content": populated_prompt}]

    kwargs = {"prefix": '{"problematic_spans":'} if "claude" in llm_engine else {}

    response = anyllm.generate_json(messages, model=llm_engine, temperature=temperature, try_strict_extraction=True, **kwargs)
    if response is None:
        return []
    problematic_spans = response["problematic_spans"]
    if printing:
        for span in problematic_spans:
            print(span)
    return problematic_spans


def run_edit_generation(paragraph, llm_engine="claude3.5-sonnet", printing=False):
    # 1. Run the detection
    problematic_spans = run_error_detection(paragraph, llm_engine=llm_engine, printing=printing)

    # 2. Run the revision
    all_edits = []
    categories = Counter([span["category"] for span in problematic_spans])
    for cat, _ in categories.most_common():
        cat_spans = [span for span in problematic_spans if span["category"] == cat]
        edits = [{"span_id": i+1, "span": span["span"], "category": cat} for i, span in enumerate(cat_spans)]
        
        span_id2span = {span["span_id"]: span for span in edits}
        clean_cat = cat.replace(' ', '_').replace("/", "_")
        with open(f"prompts/revision_{clean_cat}_v1.txt", "r") as f:
            prompt = f.read()
        
        kwargs = {"prefix": '{"revisions":'} if "claude" in llm_engine else {}

        populated_prompt = prompt.replace("[[PARAGRAPH]]", paragraph).replace("[[SPANS]]", json.dumps(edits, indent=2))
        response_revisions = anyllm.generate_json([{"role": "user", "content": populated_prompt}], model=llm_engine, **kwargs)
        
        for revision in response_revisions["revisions"]:
            span_id = int(revision["span_id"])
            span = span_id2span[span_id]
            span["revision"] = revision["revision"]
        if printing:
            print("----", cat)
            for span in edits:
                print(span)
        all_edits += edits
    return all_edits
    
def build_revised_paragraph(paragraph, edits):
    # Piecing it together
    revised_paragraph_diff = paragraph
    revised_paragraph = paragraph
    for edit in edits:
        if "revision" in edit and edit["span"] in revised_paragraph:
            revision = edit["revision"]
            revision_rich = edit["revision"]+f" \033[94m[{edit['category']}]\033[0m"
            revised_paragraph = revised_paragraph.replace(edit["span"], revision)
            revised_paragraph_diff = revised_paragraph_diff.replace(edit["span"], revision_rich)
    revised_paragraph_diff = utils_diff.make_colored_text(paragraph, revised_paragraph_diff)
    return revised_paragraph, revised_paragraph_diff

def prep_sample_indices(sample):
    paragraph = sample["preedit"]

    sample["gold_indices"] = {"all": []}
    for cat in categories:
        sample["gold_indices"][cat2clean[cat]] = []

    for span in sample["fine_grained_edits"]:
        categorization = span["categorization"].replace(" (Unnecessary ornamental and overly verbose)", "")
        if categorization == "Word Choice and Phrasing":
            categorization = "Awkward Word Choice and Phrasing"
        categorization = categorization.replace("Unnecessary/ Redundant", "Unnecessary/Redundant")
        if categorization not in cat2clean:
            # print in red color
            print("\033[91mWARNING: category not found\033[0m", categorization)
            continue
        clean_cat = cat2clean[categorization]
        if span["originalText"] in paragraph:
            idx = paragraph.index(span["originalText"])
            span["indices"] = list(range(idx, idx + len(span["originalText"])))
        else:
            span["indices"] = []
        sample["gold_indices"]["all"] += span["indices"]
        sample["gold_indices"][clean_cat] += span["indices"]
        
        for pred_key in list(sample.keys()):
            if not pred_key.startswith("pred_"):
                continue
            idx_key = pred_key.replace("pred_", "idx_")
            sample[idx_key] = {"all": []}
            sample[idx_key].update({cat2clean[cat]: [] for cat in categories})
            for span in sample[pred_key]:
                span["category"] = span["category"].replace("Redundant/Unnecessary", "Unnecessary/Redundant").replace(' (Unnecessary ornamental and overly verbose)', "")
                if span["category"] not in cat2clean:
                    # check if it is a substring of any of the categories
                    found = False
                    for cat in categories:
                        if span["category"] in cat:
                            span["category"] = cat
                            found = True
                            break
                    if not found:
                        print(f"WARNING: category not found ({span['category']})")
                        continue
                clean_cat = cat2clean[span["category"]]
                if span["span"] in paragraph:
                    idx = paragraph.index(span["span"])
                    span["indices"] = list(range(idx, idx + len(span["span"])))
                else:
                    span["indices"] = []
                sample[idx_key]["all"] += span["indices"]
                sample[idx_key][clean_cat] += span["indices"]

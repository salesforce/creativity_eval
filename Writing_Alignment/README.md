# Idiosyncrasies in LLM writing

Code repository for the paper `Can AI writing be salvaged? Mitigating Idiosyncrasies and Improving Human-AI Alignment in the Writing Process through Edits`

<p align="center" style="width: 750px;">
  <img width="750" style='vertical-align: middle;' src="images/intro.png">
</p>

## 1. Data Release

In this repository, we release (1) The LAMP corpus 1057 instruction,response pair with finegrained_edits, (2) 50 samples that are reannoted by 3 writers, (3) the 600 preference annotations we collected on to judge alignment:

- The LAMP folder contains LAMP.json and a reannotation folder. The schema for each sample is as follow
    ```
        {
        "instruction": "What happens when she goes outside to smoke a menthol cigarette and starts thinking about Shirley?",
        "preedit": "She steps out into the crisp evening air, [......]",
        "postedit": "The first drag fills her lungs, [.....]",
        "id": "W1_batch1",
        "source": "claude3.5-sonnet",
        "type": "Literary Fiction",
        "fine_grained_edits": [
                        {
                            "originalText": "She steps out into the crisp evening air, cigarette in hand.",
                            "editedText": "",
                            "categorization": "Unnecessary/Redundant Exposition"
                        },
                        {
                            "originalText": "damp chill.",
                            "editedText": "damp evening chill, or maybe it's just the menthols.",
                            "categorization": "Lack of Specificity and Detail"
                        },
                        {
                            "originalText": "like they always seem to lately",
                            "editedText": "as they often have in these difficult months",
                            "categorization": "Awkward Word Choice and Phrasing"
                        },
                        {
                            "originalText": "as",
                            "editedText": "and",
                            "categorization": "Awkward Word Choice and Phrasing"
                        },
                        {
                            "originalText": "But the good memories are fleeting, replaced by their last fight",
                            "editedText": "Yet again, the good memories subsumed the wound of their last fight",
                            "categorization": "Poor Sentence Structure"
                        },
                        {
                            "originalText": "biting",
                            "editedText": "vicious",
                            "categorization": "Awkward Word Choice and Phrasing"
                        },
                        {
                            "originalText": "She",
                            "editedText": "Through the dense, pungent smoke, she",
                            "categorization": "Lack of Specificity and Detail"
                        },
                        {
                            "originalText": "now. She stubs it out and",
                            "editedText": "and",
                            "categorization": "Unnecessary/Redundant Exposition"
                        },
                        {
                            "originalText": "cigarette",
                            "editedText": "stub",
                            "categorization": "Awkward Word Choice and Phrasing"
                        }
                ],
        "url": "https://www.newyorker.com/magazine/2012/12/24/shirley-temple-three",
        "creativity_scores": [
            "3",
            "5"
        ],
        "split": "test",
    },
  ```
- The corpus of 48 short stories is included in the `stories/` folder. 12 stories are original pieces published on the New Yorker website: we do not include the full-text version of these stories, and instead, provide a link to the original stories. For the other 36 LLM-generated stories in the corpus, we include the stories in plain text in the corpus release.
- For each of the 48 stories, we obtained annotations from three independent experts for each of the 14 TTCW, amounting to a total of (48x3x14) 2,016 test outcomes. Each test consists of a binary verdict and a plain-text explanation from the expert.

The [Data_Inspection.ipynb](https://github.com/salesforce/creativity_eval/blob/main/Data_Inspection.ipynb) notebook shows how to open all three of the files, to obtain the judgments on any given story for any given test.

For convenience, we've also put the dataset on HuggingFace: https://huggingface.co/datasets/Salesforce/ttcw_creativity_eval

## 2. LLM Creativity Benchmark (Update March 2024)

The expert judgments we collected can be used to benchmark LLMs' ability at creative writing evaluation (see Section 6 of the paper).
As new LLMs get released, we release code to facilitate benchmarking, as well as model assessments for an initial set of LLMs (GPT3.5-turbo, GPT4-Turbo, Gemini-Pro, Claude {1.3,2.0,2.1,3-opus}).

The [Evaluating_LLM.ipynb](https://github.com/salesforce/creativity_eval/blob/main/Evaluating_LLM.ipynb) notebook provides the process to (1) create benchmark files, (2) benchmark a new LLM using the `run_llm_eval.py` script (3) analyze the results.

## 3. Citing work

If you use this code or data please cite
```
@article{chakrabarty2024alignment,
  title={Can AI writing be salvaged? Mitigating Idiosyncrasies and Improving Human-AI
Alignment in the Writing Process through Edits},
  author={Chakrabarty, Tuhin and Laban, Philippe and and Wu, Chien-Sheng},
  journal={arXiv preprint arXiv:2309.14556},
  year={2024}
}
```

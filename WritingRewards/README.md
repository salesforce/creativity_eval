# AI-Slop to AI-Polish

Welcome to the repository for "AI-Slop to AI-Polish? Aligning Language Models through Edit-Based Writing Rewards and Test-time Computation"

## Paper and Models

- **Paper**: [AI-Slop to AI-Polish? Aligning Language Models through Edit-Based Writing Rewards and Test-time Computation](https://arxiv.org/pdf/2504.07532)

- **Download WQRM models from Hugging Face**:
  - [WQRM](https://huggingface.co/Salesforce/WQRM) (Just trained on LAMP PR)
  - [WQRM-PRE](https://huggingface.co/Salesforce/WQRM-PRE) (Trained on LAMP PR+ extended data of 100 Expert and 83 MFA paragraphs)

## Repository Structure

The data for the Writing Quality Benchmark can be found in the respective folder. We have provided individual splits as well as combined splits.

Code for training can be found inside training folders. It's organized as:

- **Llama3**
  - sft.py
  - inference.py
  - data
    - lamp-P
    - lamp-PR
    - lamp-R
    - lamp-P-exp
    - lamp-PR-exp
    - lamp-P-exp-predict
    - lamp-PR-exp-predict
- **ModernBert**
  - train_wqrm_mbert.py

To train an Edit COT model, you can use the same sft.py code. Data for training it can be found in the edit folder.

The file WQRM_annotations.json contains the human annotations vs WQRM on the `<first draft, random cot, best cot>` experiment.

The final calibration experiment for "How does content affect writing quality?" is in content_quality_experiment:
- lamp_PRE_train.json (Contains MFA-written 83 and Expert-written 100 paragraphs with scores 7.5 and 10 to train a new WQRM)
- lamp_PRE_val.json
- reward_calibration_less_content.json
- reward_calibration_nore_content.json

## Citation

If you use our code and data, please cite us:

```
@article{chakrabarty2025ai,
      title={AI-Slop to AI-Polish? Aligning Language Models through Edit-Based Writing Rewards and Test-time Computation},
      author={Chakrabarty, Tuhin and Laban, Philippe and Wu, Chien-Sheng},
      journal={arXiv preprint arXiv:2504.07532},
      year={2025}
}
```

Welcome to the repo for "AI-Slop to AI-Polish? Aligning Language Models through Edit-Based Writing Rewards and Test-time Computation"
Read our paper https://arxiv.org/pdf/2504.07532

The data for the Writing Qualiity Benchmark can be found in the respective folder. We have given individual splits as well as combined splits

Code for training can be found inside training folders. Its organized as
- Llama3

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

- ModernBert

          - train_wqrm_mbert.py


To train an Edit COT model, you can use same sft.py code. Data for training it can be found in edit folder

The file WQRM_annotations.json is the human annotations vs WQRM on <first draft, random cot, best cot> experiment

The final calibration experiment for "How does content affect writing quality?" is in content_quality_experiment.

          - lamp_PRE_train.json ( Contains MFA written 83 and Exper written 100 paragraphs with scores 7.5 and 10 to train a new WQRM)
          - lamp_PRE_val.json



If you use our code and data please cite us

            @article{chakrabarty2025ai,
                  title={AI-Slop to AI-Polish? Aligning Language Models through Edit-Based Writing Rewards and Test-time Computation},
                  author={Chakrabarty, Tuhin and Laban, Philippe and Wu, Chien-Sheng},
                  journal={arXiv preprint arXiv:2504.07532},
                  year={2025}
            }


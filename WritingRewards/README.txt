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


To train an Edit COT model, you can use same sft.py code. Data for training it can be found in edit folder

If you use our code and data please cite us

                        @article{chakrabarty2025ai,
                          title={AI-Slop to AI-Polish? Aligning Language Models through Edit-Based Writing Rewards and Test-time Computation},
                          author={Chakrabarty, Tuhin and Laban, Philippe and Wu, Chien-Sheng},
                          journal={arXiv preprint arXiv:2504.07532},
                          year={2025}
                        }

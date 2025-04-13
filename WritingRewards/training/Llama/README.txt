To train a Llama3.1-70B model. Run this command. sft.py and llama_3_70b_fsdp_qlora.yaml needs to be in same directory

        ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=8 ./sft.py --config llama_3_70b_fsdp_qlora.yaml

The requirements to train Llama models were this 

                                tensorboard==2.19.0
                                accelerate==0.29.3
                                bitsandbytes==0.45.2
                                datasets==2.18.0
                                trl==0.8.6
                                peft==0.10.0
                                pillow==11.1.0
                                numpy==1.24.3
                                torch==2.2.2
                                torchaudio==2.2.2
                                torchvision==0.17.2
                                transformers==4.43.2

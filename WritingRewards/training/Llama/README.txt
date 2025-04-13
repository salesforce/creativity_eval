To train a Llama3.1-70B model

ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=8 ./sft.py --config llama_3_70b_fsdp_qlora.yaml

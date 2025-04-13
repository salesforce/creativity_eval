import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset
from random import randint
import json
 
peft_model_id = "your_checkpoint"
 
# Load Model with PEFT adapter
model = AutoPeftModelForCausalLM.from_pretrained(
  peft_model_id,
  token = "your_token",
  torch_dtype=torch.float16,
  quantization_config= {"load_in_4bit": True, "bnb_4bit_compute_dtype" : torch.float16},
  device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
tokenizer.pad_token = tokenizer.eos_token


# Load our test dataset
eval_dataset = load_dataset("json", data_files="sample.json", split="train")
predictions = []
for rand_idx in range(len(eval_dataset)):
    print("Processing..",rand_idx)
    messages = eval_dataset[rand_idx]["messages"][:2]

# Test on sample
    input_ids = tokenizer.apply_chat_template(messages,add_generation_prompt=True,return_tensors="pt").to(model.device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=30,
        eos_token_id= tokenizer.eos_token_id,
        do_sample=False)#,
        #temperature=0.6,
        #top_p=0.9)

    response = outputs[0][input_ids.shape[-1]:]
    x = tokenizer.decode(response,skip_special_tokens=True)
    print(x)
    predictions.append({'gold' : eval_dataset[rand_idx]['messages'][2]['content'], 'predicted': x})

with open('./pred.json','w') as f:
    f.write(json.dumps(predictions, indent=4, ensure_ascii=False))

#print(f"**Query:**\n{eval_dataset[rand_idx]['messages'][1]['content']}\n")
#print(f"**Original Answer:**\n{eval_dataset[rand_idx]['messages'][2]['content']}\n")
#print(f"**Generated Answer:**\n{tokenizer.decode(response,skip_special_tokens=True)}")

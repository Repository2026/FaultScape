from unsloth import FastLanguageModel
import torch
import random
import pickle
import sys

import torch
from datasets import Dataset

import numpy as np

from transformers import Trainer, TrainingArguments

import os

import json
from datasets import Dataset
from sklearn.model_selection import train_test_split

import torch.nn.functional as F
import random

from transformers import AutoTokenizer, AutoModelForCausalLM


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# step1. ===========load model and dataset==========
print(" load model and dataset")
max_seq_length = 2048 
dtype = None 
load_in_4bit = True 

fourbit_models = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",      
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",          
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",       
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",            
] # More models at https://huggingface.co/unsloth


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...",
)

model = FastLanguageModel.get_peft_model(  
    model,
    r = 16,

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",], 

    lora_alpha = 16, 
    lora_dropout = 0, 
    bias = "none",   
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
    use_rslora = False,  
    loftq_config = None, 
)



class BinaryClassificationHead(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, hidden_states):
        last_token = hidden_states[:, -1, :]
        return self.classifier(last_token)

def add_classification_head(model):
    hidden_size = model.config.hidden_size  # Llama-3-8B: 4096
    model.binary_head = BinaryClassificationHead(hidden_size)
    
    original_forward = model.forward
    def custom_forward(**kwargs):
        outputs = original_forward(**kwargs)
        
        cls_logits = model.binary_head(outputs.hidden_states[-1])
        return {
            "lm_logits": outputs.logits, 
            "cls_logits": cls_logits.squeeze() 
        }
    model.forward = custom_forward
    return model

model = add_classification_head(model)


# load data
with open('./Multi-view_Datasets/cause_fl_data.json', 'r', encoding='utf-8') as f:
    combined_data = json.load(f)

random.shuffle(combined_data)

# 1-5 item data
for i, item in enumerate(combined_data[:5]):
    print(f"no {i+1} data is: {item}") 


stringInstruction = "Based on the code snippet and its analysis below, determine if the line marked with 'rank2fixstart' and 'rank2fixend' has defects. Output '1' for defects and '0' for no defects. "

dataset = Dataset.from_dict({
                            "instruction": [stringInstruction for item in combined_data],
                             "input": [item["code_analyze_content"]  for item in combined_data],   
                             "output": [item["label"] for item in combined_data]})

df = dataset.to_pandas()

train_df, remaining_df = train_test_split(df, test_size=0.3, random_state=42)
valid_df, test_df = train_test_split(remaining_df, test_size=0.5, random_state=42)

train_data = Dataset.from_pandas(train_df)
valid_data = Dataset.from_pandas(valid_df)
test_data = Dataset.from_pandas(test_df)


alpaca_prompt = """### Instruction:{}, ### Input:{}, ### Response:{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip( instructions, inputs, outputs ):
        text = alpaca_prompt.format( instruction, input, output ) + EOS_TOKEN
        texts.append(text)
    return {"text": texts, }
pass

def formatting_prompts_func_test(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = [''] * len(instructions)
    texts = []
    for instruction, input, output in zip( instructions, inputs, outputs ):
        text = alpaca_prompt.format( instruction, input, output )
        texts.append(text)
    return {"text": texts, "label":examples["output"] }
pass

train_data = train_data.map(formatting_prompts_func, batched=True)
valid_data = valid_data.map(formatting_prompts_func, batched=True)

test_data = test_data.select(range(1000))
test_data = test_data.map(formatting_prompts_func_test, batched=True)



# step 2.===========fine-tune==========
print("fine-tune ing...")
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    # train_dataset = dataset,

    train_dataset=train_data,
    eval_dataset= valid_data.select(random.sample(range(len(valid_data)), 1000)) ,  

    dataset_text_field = "text", 
    max_seq_length = max_seq_length, 
    dataset_num_proc = 2,
    packing = False,  

    args = TrainingArguments(
        per_device_train_batch_size = 2, 
        per_device_eval_batch_size=2, 
        gradient_accumulation_steps = 4,  
        warmup_steps = 5, 
        max_steps = 40000,  
        learning_rate = 2e-4, 
        fp16 = not is_bfloat16_supported(), 
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "./outputs_Analyze_FL",
        report_to=[],
        eval_strategy="steps", 
        eval_steps=1000,  
        save_steps=2000, 
        load_best_model_at_end=True,   

    ),
   
)

trainer_stats = trainer.train()


print("save best model....")
model.save_pretrained("./outputs_Analyze_FL/best_model_code_FL") 
tokenizer.save_pretrained("./outputs_Analyze_FL/best_model_code_FL")


# step3. ============Infer defect4j data for prediction  ==========
print("Infer defect4j data! ")


with open("./src_code.pkl", "rb") as file1:  
    src_code = pickle.load(file1)
with open("./statements.pkl", "rb") as file2: 
    statements = pickle.load(file2)
with open("./faulty_statement_index.pkl",  "rb") as file_index: 
    faulty_statement_number = pickle.load(file_index)

scr_code_total_code_snippet = 0
statement_total_code_snippet = 0

all_project_names = sorted( src_code.keys() )


start_project = sys.argv[1]
end_project = sys.argv[2]

start_project = int(start_project)
end_project = int(end_project)
print(start_project,end_project)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_instruct_id = "unsloth/llama-3-8b-Instruct"
tokenizer_instruct = AutoTokenizer.from_pretrained(model_instruct_id)
model_instruct = AutoModelForCausalLM.from_pretrained(
    model_instruct_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

d4j_Analyze_Inference_Results = {}


for project in all_project_names[start_project:end_project]:

    
    project_results = []

    for t, code_snippet in enumerate(src_code[project]):

        
        prompt_analyzer =  "First, identify the code statement tagged with 'rank2fixstart' and 'rank2fixend' in the snippet above. Then, assess the likelihood that this tagged statement contains a bug."

        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt_analyzer + "\nThis code snippet is: \n" +code_snippet },
        ]

        input_ids = tokenizer_instruct.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model_instruct.device)

        terminators = [
            tokenizer_instruct.eos_token_id,
            tokenizer_instruct.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = model_instruct.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        prediction_analyze = tokenizer_instruct.decode(response, skip_special_tokens=True)


        code_Analyze_data = "\nThis code snippet is: \n" + code_snippet + "\n****This code snippet is analyzed as follows:****\n" +  prediction_analyze 
        
        project_results.append(code_Analyze_data)

    d4j_Analyze_Inference_Results[project] = project_results



result_file_name = f"../d4j_Cause_Inference_Results.pkl"

with open(result_file_name, "wb") as result_file:
    pickle.dump(d4j_Analyze_Inference_Results, result_file)

print(" the inference results are saved!")



# step4. ============predict defect4j LLM code_Fl data  ==========
print("Begin......!")

with open("./statements.pkl", "rb") as file2:    
    statements = pickle.load(file2)
with open("./faulty_statement_index.pkl",  "rb") as file_index: 
    faulty_statement_number = pickle.load(file_index)
with open("./d4j_Cause_Inference_Results.pkl",  "rb") as file_index: 
    src_code = pickle.load(file_index)


scr_code_total_code_snippet = 0
statement_total_code_snippet = 0

all_project_names = sorted( src_code.keys() )

start_project = sys.argv[1]
end_project = sys.argv[2]

start_project = int(start_project)
end_project = int(end_project)
print(start_project,end_project)


max_seq_length = 2048 
dtype = None 
load_in_4bit = True 

from unsloth import FastLanguageModel
finetune_model, finetune_tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./best_model_code_FL", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(finetune_model) # Enable native 2x faster inference


for project in all_project_names[start_project:end_project]:

    every_project_result = {}
    every_project_result_list = []

    for t, code_snippet in enumerate(src_code):

        finetune_model.eval()

        pad_token_id = finetune_tokenizer.pad_token_id
        eos_token_id = finetune_tokenizer.eos_token_id

        id_for_1 = finetune_tokenizer.convert_tokens_to_ids('1')
        id_for_0 = finetune_tokenizer.convert_tokens_to_ids('0')        

        code_Analyze_data = code_snippet

       
        alpaca_prompt = """### Instruction:{}, ### Input:{}, ### Response:{}"""
        instruction = "Based on the code snippet and its analysis below, determine if the line marked with 'rank2fixstart' and 'rank2fixend' has defects. Output '1' for defects and '0' for no defects. "

        inputs = finetune_tokenizer(
                [
                    alpaca_prompt.format(
                        instruction, 
                        code_Analyze_data , 
                        "",
                    )
                ], return_tensors = "pt").to("cuda")


        outputs = finetune_model.generate(**inputs, max_new_tokens=10, output_scores=True, use_cache=True, pad_token_id=pad_token_id, return_dict_in_generate=True)


        generated_tokens = outputs['sequences'][0]
        scores = outputs['scores'] 

        num_input_tokens = inputs['input_ids'].shape[1]
        generated_tokens = generated_tokens[num_input_tokens:]

    
        prediction = finetune_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    
        probabilities = F.softmax(scores[0], dim=-1)
        top_probs, top_ids = torch.topk(probabilities, 5)
        top_probs = top_probs.cpu().tolist()
        top_ids = top_ids.cpu().tolist()
      

        prob_for_1 = probabilities[0, id_for_1].item() 
        prob_for_0 = probabilities[0, id_for_0].item() 

      
        pred_response = (prediction[0])

    
        every_project_result[t] = prob_for_1
        every_project_result_list.append(  prob_for_1 )


    # output every project the LLM_cause_FL value
    print("{}=={}".format( project, every_project_result) )


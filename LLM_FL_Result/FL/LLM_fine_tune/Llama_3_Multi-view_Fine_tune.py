import os, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import set_seed

from unsloth import FastLanguageModel
from peft import LoraConfig, get_peft_model

# -------------------------
# Config
# -------------------------
VIEWS = ["code", "type", "cause", "fix"]
DATA_PATHS = {
    "code":  "./Multi-view_Datasets/code_fl_data.json",
    "type":  "./Multi-view_Datasets/type_fl_data.json",
    "cause": "./Multi-view_Datasets/cause_fl_data.json",
    "fix":   "./Multi-view_Datasets/fix_fl_data.json",
}

SAVE_DIRS = {
    "code":  "./best_model_code_FL",
    "type":  "./best_model_type_FL",
    "cause": "./best_model_cause_FL",
    "fix":   "./best_model_fix_FL",
}

EXTRA_SAVE = "./heads_and_gate.pt"  # heads + gating
MODEL_NAME = "/unsloth/llama-3-8b-bnb-4bit"
MAX_LEN = 2048

LAMBDA_CONTRAST = 0.1
TAU = 0.07
SEED = 3407

TRAIN_ARGS = dict(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    warmup_ratio=0.03,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    logging_steps=100,
    save_steps=5000,
    eval_steps=5000,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    report_to=[],
    output_dir="./outputs_joint_multiview",
    optim="adamw_8bit",
    fp16=True,
)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def detect_key_field(items):
    candidate_keys = ["uid", "id", "sample_id", "stmt_id", "statement_id", "bug_id", "pair_id"]
    for k in candidate_keys:
        if isinstance(items, list) and len(items) > 0 and isinstance(items[0], dict) and k in items[0]:
            return k
    return None

def get_text_field(item):
    for k in ["code_analyze_content", "input", "content", "text", "prompt"]:
        if k in item:
            return item[k]
    return json.dumps(item, ensure_ascii=False)

def get_label_field(item):
    for k in ["label", "output", "y", "target"]:
        if k in item:
            return item[k]
    return None

def build_multiview_samples(data_by_view):
    """
    list[dict]: {
      "x_code": ..., "x_type": ..., "x_cause": ..., "x_fix": ...,
      "label": 0/1
    }
    """
    key = None
    for v in VIEWS:
        k = detect_key_field(data_by_view[v])
        if k is not None:
            key = k
            break

    samples = []

    if key is not None:
        maps = {}
        for v in VIEWS:
            maps[v] = {item[key]: item for item in data_by_view[v] if key in item}

        common_ids = set(maps[VIEWS[0]].keys())
        for v in VIEWS[1:]:
            common_ids &= set(maps[v].keys())

        common_ids = sorted(list(common_ids))
        for sid in common_ids:
            items = {v: maps[v][sid] for v in VIEWS}
            label = get_label_field(items["code"])
            if label is None:
                for v in VIEWS[1:]:
                    label = get_label_field(items[v])
                    if label is not None:
                        break
            if label is None:
                continue
            samples.append({
                "x_code":  get_text_field(items["code"]),
                "x_type":  get_text_field(items["type"]),
                "x_cause": get_text_field(items["cause"]),
                "x_fix":   get_text_field(items["fix"]),
                "label": int(label),
            })
    else:
        n = min(len(data_by_view[v]) for v in VIEWS)
        for i in range(n):
            items = {v: data_by_view[v][i] for v in VIEWS}
            label = get_label_field(items["code"])
            if label is None:
                for v in VIEWS[1:]:
                    label = get_label_field(items[v])
                    if label is not None:
                        break
            if label is None:
                continue
            samples.append({
                "x_code":  get_text_field(items["code"]),
                "x_type":  get_text_field(items["type"]),
                "x_cause": get_text_field(items["cause"]),
                "x_fix":   get_text_field(items["fix"]),
                "label": int(label),
            })

    random.shuffle(samples)
    return samples

def build_prompt(view_name, content):
    inst = (
        "Based on the code snippet and its analysis below, determine if the line marked with "
        "'rank2fixstart' and 'rank2fixend' has defects. Output 1 for defects and 0 for no defects."
    )
    return f"### Instruction:{inst}\n### Input:{content}\n### Response:"

# -------------------------
# Data Collator (tokenize 4 views per batch)
# -------------------------
class MultiViewCollator:
    def __init__(self, tokenizer, max_len=2048):
        self.tok = tokenizer
        self.max_len = max_len

    def __call__(self, features):
        batch = {}
        labels = torch.tensor([f["label"] for f in features], dtype=torch.float)

        for v in VIEWS:
            texts = [build_prompt(v, f[f"x_{v}"]) for f in features]
            enc = self.tok(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )
            batch[f"{v}_input_ids"] = enc["input_ids"]
            batch[f"{v}_attention_mask"] = enc["attention_mask"]

        batch["label"] = labels
        return batch

# -------------------------
# Model: shared base + 4 adapters + 4 heads + 1 gate
# -------------------------
class MultiViewModel(nn.Module):
    def __init__(self, peft_model):
        super().__init__()
        self.base = peft_model
        h = peft_model.config.hidden_size

        self.head = nn.ModuleDict({
            v: nn.Sequential(
                nn.Linear(h, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            ) for v in VIEWS
        })

        # shared gating MLP: input [h^v; mean(h^{-v})] -> g^v in (0,1)^H
        self.gate = nn.Sequential(
            nn.Linear(2*h, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.Sigmoid()
        )

    def _encode(self, input_ids, attention_mask, adapter_name):
        self.base.set_adapter(adapter_name)
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hs = out.hidden_states[-1]  # [B,T,H]
        last_idx = attention_mask.sum(dim=1) - 1
        emb = hs[torch.arange(hs.size(0)), last_idx]  # [B,H]
        return emb

    def forward(self, batch): # encode each view with its adapter -> get h^v for each view
        h = {}
        for v in VIEWS:
            h[v] = self._encode(
                batch[f"{v}_input_ids"],
                batch[f"{v}_attention_mask"],
                v
            )
        return h   # h:[B,H]

    def classify(self, h):  #
        logits = {v: self.head[v](h[v]).squeeze(-1) for v in VIEWS}  # [B]
        return logits

    def align(self, h): 
        h_tilde = {}
        for v in VIEWS:
            others = [h[u] for u in VIEWS if u != v]
            h_bar = torch.stack(others, dim=0).mean(dim=0)  # [B,H]
            g = self.gate(torch.cat([h[v], h_bar], dim=-1))  # [B,H]
            h_tilde[v] = g * h[v] + (1 - g) * h_bar   # core alignment step
        return h_tilde

def info_nce_contrast(h_tilde, tau=0.07): #  contrastive loss learning
    B = next(iter(h_tilde.values())).size(0)
    V = len(VIEWS)

    z = torch.stack([F.normalize(h_tilde[v], dim=-1) for v in VIEWS], dim=1)  # [B,V,H]  [batch, view, hidden]
    sim = torch.einsum("bvh,kuh->bvku", z, z) / tau  # [B,V,B,V]

    loss = 0.0
    count = 0

    for vi in range(V):
        for vj in range(V):
            if vj == vi:
                continue
            num = torch.exp(sim[:, vi, torch.arange(B), vj])  # [B]
            den = torch.exp(sim[:, vi].reshape(B, B*V))       # [B, B*V]
            self_index = torch.arange(B) * V + vi
            den.scatter_(1, self_index.unsqueeze(1), 0.0)     # exclude self (i,vi)
            den = den.sum(dim=1)                               # [B]
            loss += (-torch.log(num / (den + 1e-9))).mean()
            count += 1

    return loss / count

# -------------------------
# Trainer
# -------------------------
class JointTrainer(Trainer):
    def __init__(self, lambda_contrast=0.1, tau=0.07, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_contrast = lambda_contrast
        self.tau = tau

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["label"].float()  # [B]

        h = model(inputs)  
        logits = model.classify(h)
        h_tilde = model.align(h)

        # BCE averaged over views
        loss_ce = 0.0
        for v in VIEWS:
            loss_ce += F.binary_cross_entropy_with_logits(logits[v], labels)
        loss_ce = loss_ce / len(VIEWS)

        loss_con = info_nce_contrast(h_tilde, tau=self.tau)
        loss = loss_ce + self.lambda_contrast * loss_con

        if return_outputs:
            return loss, {"logits": logits}
        return loss



# step============ 1. load dataset 2.fine-tune ==========

# 1) Load data
data_by_view = {v: load_json(DATA_PATHS[v]) for v in VIEWS}
samples = build_multiview_samples(data_by_view)  # list of dict: {"x_code":..., "x_type":..., "x_cause":..., "x_fix":..., "label":0/1}
print(f"[Data] multiview samples: {len(samples)}")
print("[Data] example keys:", samples[0].keys())

# 2) Split
df = Dataset.from_list(samples).to_pandas()
train_df, remaining_df = train_test_split(df, test_size=0.3, random_state=SEED)
valid_df, test_df = train_test_split(remaining_df, test_size=0.5, random_state=SEED)

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
valid_ds = Dataset.from_pandas(valid_df.reset_index(drop=True))

# 3) Load base model + tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_LEN,
    dtype=None,
    load_in_4bit=True,
)

# 4) Add 4 LoRA adapters on shared base
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_cfg, adapter_name="code")
model.add_adapter("type",  lora_cfg)
model.add_adapter("cause", lora_cfg)
model.add_adapter("fix",   lora_cfg)
model.set_adapter("code")

mv_model = MultiViewModel(model)

# 5) Trainer
collator = MultiViewCollator(tokenizer, max_len=MAX_LEN)

args = TrainingArguments(
    seed=SEED,
    **TRAIN_ARGS
)

trainer = JointTrainer(
    model=mv_model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds.select(range(min(len(valid_ds), 1000))),
    data_collator=collator,
    lambda_contrast=LAMBDA_CONTRAST,
    tau=TAU,
)

# 6) Train
trainer.train()

# 7) Save 4 "models" (adapters) into your paths
os.makedirs(os.path.dirname(EXTRA_SAVE) or ".", exist_ok=True)

for v in VIEWS:
    os.makedirs(SAVE_DIRS[v], exist_ok=True)
    mv_model.base.set_adapter(v)
    # here save the LoRA adapter for this view (lightweight), dir name is specified best_model_xxx_FL
    mv_model.base.save_pretrained(SAVE_DIRS[v])
    tokenizer.save_pretrained(SAVE_DIRS[v])
    print(f"[Save] adapter saved to: {SAVE_DIRS[v]}")

# Save heads + gate
torch.save(
    {
        "head": mv_model.head.state_dict(),
        "gate": mv_model.gate.state_dict(),
    },
    EXTRA_SAVE
)
print(f"[Save] heads+gate saved to: {EXTRA_SAVE}")






# step3. ============Infer defect4j data for prediction (Type/Cause/Fix ) ==========
print("Infer defect4j data ")

import sys
import json
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------
# Load inputs
# -----------------------
with open("./src_code.pkl", "rb") as f:
    src_code = pickle.load(f)
with open("./statements.pkl", "rb") as f:
    statements = pickle.load(f)
with open("./faulty_statement_index.pkl", "rb") as f:
    faulty_statement_number = pickle.load(f)

test_fail_path = r"./LLM_FL_Result/FL/pairwise_rerank/test_failure_info.json"
with open(test_fail_path, "r", encoding="utf-8") as f:
    test_failure_info = json.load(f)

# filename -> content
fail_info_map = {
    item["filename"]: item["content"]
    for item in test_failure_info
    if isinstance(item, dict) and "filename" in item and "content" in item
}

all_project_names = sorted(src_code.keys())

# start_project = int(sys.argv[1])
# end_project = int(sys.argv[2])
start_project = 0
end_project = len(all_project_names)
print("Range:", start_project, end_project)

# -----------------------
# Load LLM (instruct)
# -----------------------
model_instruct_id = "unsloth/llama-3-8b-Instruct" # or llama-3-70b-Instruct
tokenizer_instruct = AutoTokenizer.from_pretrained(model_instruct_id)
model_instruct = AutoModelForCausalLM.from_pretrained(
    model_instruct_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# -----------------------
# Helpers
# -----------------------
def truncate_text(s: str, max_chars: int = 4000) -> str:
    if not s:
        return ""
    s = str(s)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n...\n(TRUNCATED)\n"

def build_fail_prefix(project_name: str) -> str:
    fail_ctx = truncate_text(fail_info_map.get(project_name, ""), max_chars=4000)
    if fail_ctx.strip():
        return (
            "Failing test information for this bug (use as primary debugging context):\n"
            f"{fail_ctx}\n"
            "---- End of failing test info ----\n\n"
        )
    return (
        "Failing test information: (not provided for this bug)\n"
        "---- End of failing test info ----\n\n"
    )

def chat_generate(messages, max_new_tokens=256, temperature=0.6, top_p=0.9):
    input_ids = tokenizer_instruct.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model_instruct.device)

    terminators = [
        tokenizer_instruct.eos_token_id,
        tokenizer_instruct.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.no_grad():
        outputs = model_instruct.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer_instruct.decode(response, skip_special_tokens=True)

# -----------------------
# View-specific prompts (ALL use failing tests)
# -----------------------
VIEW_PROMPTS = {
    "type": (
        "You are given a code snippet where the target statement is marked by 'rank2fixstart' and 'rank2fixend'.\n"
        "Use the failing test information (assertion + stack trace) to infer the most likely fault type/category.\n"
        "1) Identify the tagged statement.\n"
        "2) Determine the fault type (e.g., incorrect condition, off-by-one, null handling, API misuse, data structure misuse, etc.).\n"
        "Return format:\n"
        "- Tagged statement: ...\n"
        "- Fault type: ...\n"
        "- Evidence: 2-4 bullets tied to failing test/stack trace\n"
    ),
    "cause": (
        "You are given a code snippet where the target statement is marked by 'rank2fixstart' and 'rank2fixend'.\n"
        "Use the failing test information (assertion + stack trace) to infer the root cause of the bug.\n"
        "1) Identify the tagged statement.\n"
        "2) Explain why the failing test fails (connect assertion/stack trace to program state / logic).\n"
        "Return format:\n"
        "- Tagged statement: ...\n"
        "- Root cause: ...\n"
        "- Evidence: 2-4 bullets tied to failing test/stack trace\n"
    ),
    "fix": (
        "You are given a code snippet where the target statement is marked by 'rank2fixstart' and 'rank2fixend'.\n"
        "Use the failing test information (assertion + stack trace) to propose the most plausible repair intent.\n"
        "1) Identify the tagged statement.\n"
        "2) Describe the likely fix action (e.g., adjust condition, handle null, correct boundary, update API usage).\n"
        "Return format:\n"
        "- Tagged statement: ...\n"
        "- Repair intent: ...\n"
        "- Evidence: 2-4 bullets tied to failing test/stack trace\n"
    ),
}


# -----------------------
# Run inference for each view and save separate pkl
# -----------------------
for view_name in ["type", "cause", "fix"]:
    print(f"\n[Run] View = {view_name}")
    d4j_results = {}

    for project in all_project_names[start_project:end_project]:
        fail_prefix = build_fail_prefix(project)
        project_results = []

        for t, code_snippet in enumerate(src_code[project]):
            messages = [
                {"role": "system", "content": "You are a Java debugging assistant. Be concise and evidence-based."},
                {"role": "user", "content": fail_prefix + "Code snippet:\n" + code_snippet + "\n\n" + VIEW_PROMPTS[view_name]},
            ]

            pred = chat_generate(messages, max_new_tokens=256, temperature=0.6, top_p=0.9)

            pack = (
                f"[Bug/Project] {project}\n"
                f"[View] {view_name}\n"
                + fail_prefix
                + "Code snippet:\n"
                + code_snippet
                + "\n****LLM output:****\n"
                + pred
            )
            project_results.append(pack)

        d4j_results[project] = project_results

    out_path = f"../d4j_{view_name.capitalize()}_Inference_Results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(d4j_results, f)

    print(f"[Saved] {out_path}")

print("\nAll views done.")  # code view need not to inference 



# step4. ============predict defect4j multi-view semantic FL values ==========
print("Begin Step4 (multi-view predict) ......!")

import sys
import pickle
import torch
import torch.nn.functional as F

from unsloth import FastLanguageModel

# -------------------------
# 1) Load meta info (optional, kept for compatibility)
# -------------------------
with open("./statements.pkl", "rb") as f:
    statements = pickle.load(f)
with open("./faulty_statement_index.pkl", "rb") as f:
    faulty_statement_number = pickle.load(f)

# -------------------------
# 2) Load Step3 inference results (multi-view inputs)
#    Each is a dict: {project_name: [text0, text1, ...]}
# -------------------------
with open("./d4j_Code_Inference_Results.pkl", "rb") as f:    
    d4j_code = pickle.load(f)

with open("./d4j_Type_Inference_Results.pkl", "rb") as f:
    d4j_type = pickle.load(f)

with open("./d4j_Cause_Inference_Results.pkl", "rb") as f:
    d4j_cause = pickle.load(f)

with open("./d4j_Fix_Inference_Results.pkl", "rb") as f:
    d4j_fix = pickle.load(f)


# -------------------------
# 3) Project range
# -------------------------
all_project_names = sorted(d4j_cause.keys())  
# start_project = int(sys.argv[1])
# end_project = int(sys.argv[2])
start_project = 0
end_project = len(all_project_names)
print("Project range:", start_project, end_project)

# -------------------------
# 4) Load 4 fine-tuned models (view-specific)
# -------------------------
max_seq_length = 2048
dtype = None
load_in_4bit = True

VIEW_MODEL_PATHS = {
    "code":  "./best_model_code_FL",
    "type":  "./best_model_type_FL",
    "cause": "./best_model_cause_FL",
    "fix":   "./best_model_fix_FL",
}

def load_view_model(path):
    m, tok = FastLanguageModel.from_pretrained(
        model_name=path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(m)
    m.eval()
    return m, tok

models = {}
tokenizers = {}
for v, p in VIEW_MODEL_PATHS.items():
    print(f"Loading finetuned model for view={v}: {p}")
    m, tok = load_view_model(p)
    models[v] = m
    tokenizers[v] = tok

alpaca_prompt = """### Instruction:{}, ### Input:{}, ### Response:{}"""
instruction = (
    "Based on the code snippet and its analysis below, determine if the line marked with "
    "'rank2fixstart' and 'rank2fixend' has defects. Output '1' for defects and '0' for no defects."
)

@torch.no_grad()
def predict_prob_1(view_name: str, text_input: str) -> float:
    model = models[view_name]
    tok = tokenizers[view_name]

    # token ids for '1' and '0'
    id_for_1 = tok.convert_tokens_to_ids("1")
    id_for_0 = tok.convert_tokens_to_ids("0")

    pad_token_id = tok.pad_token_id
    if pad_token_id is None:
        pad_token_id = tok.eos_token_id

    inputs = tok(
        [alpaca_prompt.format(instruction, text_input, "")],
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=3,
        output_scores=True,
        return_dict_in_generate=True,
        use_cache=True,
        pad_token_id=pad_token_id,
    )

    # first step scores: [1, vocab]
    scores0 = outputs.scores[0]
    probs0 = F.softmax(scores0, dim=-1)

    prob_for_1 = probs0[0, id_for_1].item() if id_for_1 is not None else 0.0
    prob_for_0 = probs0[0, id_for_0].item() if id_for_0 is not None else 0.0


    return float(prob_for_1)

# -------------------------
# 6) Multi-view prediction loop
# -------------------------
multi_view_scores = {}   # {project: {t: {"code":p,"type":p,"cause":p,"fix":p}}}

for project in all_project_names[start_project:end_project]:
    n = min(
        len(d4j_code.get(project, [])),
        len(d4j_type.get(project, [])),
        len(d4j_cause.get(project, [])),
        len(d4j_fix.get(project, [])),
    )
    if n == 0:
        print(f"[WARN] {project} has no snippets, skip.")
        continue

    every_project_result = {}        # {t: {"code":..., "type":..., "cause":..., "fix":...}}
    every_project_result_list = []   # list of dicts, optional

    for t in range(n):
        text_code  = d4j_code[project][t]
        text_type  = d4j_type[project][t]
        text_cause = d4j_cause[project][t]
        text_fix   = d4j_fix[project][t]

        p_code  = predict_prob_1("code",  text_code)
        p_type  = predict_prob_1("type",  text_type)
        p_cause = predict_prob_1("cause", text_cause)
        p_fix   = predict_prob_1("fix",   text_fix)

        every_project_result[t] = {
            "code":  p_code,
            "type":  p_type,
            "cause": p_cause,
            "fix":   p_fix,
        }
        every_project_result_list.append(every_project_result[t])

    multi_view_scores[project] = every_project_result

    print(f"{project}=={every_project_result}")

# -------------------------
# 7) Save
# -------------------------
out_path = "../d4j_MultiView_Semantic_Scores.pkl"
with open(out_path, "wb") as f:
    pickle.dump(multi_view_scores, f)
print(f"[Saved] multi-view scores -> {out_path}")

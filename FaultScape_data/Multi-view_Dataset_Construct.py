import pickle
import re
import random
import json
import os
import datetime
import time
import sys
import difflib

'''
This file is used to generate fine-tuning datasets for LLaMA-3-70B.

Input:
    dataset.pkl data + well-designed prompt
    (positive samples require prompt, negative samples do not)

Output:
    LLaMA-3-70B predictions for different data samples
'''

# Fault Type Descriptions

FAULT_TYPE_EXPLANATIONS = {

    "MoveStmt":
        "Move Statement: A statement is located in an incorrect position and should be moved.",

    "InsertNullPointerChecker":
        "Missing Null Check: A variable or expression is used without checking whether it is null.",

    "MutateLiteralExpr":
        "Incorrect Literal Expression: A constant value in the statement is incorrect.",

    "MutateOperators":
        "Incorrect Operator: An operator (e.g., relational, arithmetic, instanceof, or parentheses) is incorrectly used.",

    "MutateMethodInvExpr":
        "Incorrect Method Invocation: The method name or one of its arguments is incorrect.",

    "MutateVariable":
        "Incorrect Variable: The variable used in the statement is incorrect.",

    "MutateDataType":
        "Incorrect Data Type: The data type in a cast expression or variable declaration is incorrect.",

    "MutateConditionalExpr":
        "Incorrect Conditional Expression: A conditional expression should be added, removed, or modified.",

    "MutateReturnStmt":
        "Incorrect Return Expression: The expression in the return statement is incorrect.",

    "RemoveBuggyStmt":
        "Redundant Statement: The statement should be removed because it should not appear at this location.",

    "InsertMissedStmt":
        "Missing Statement: A required statement (e.g., method invocation, return statement, conditional statement, or try-catch block) is missing."
}


# Diff Extraction Utilities

def extract_changed_blocks_with_context(old_code, new_code, context=1):

    old_lines = old_code.splitlines()
    new_lines = new_code.splitlines()

    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)

    results = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():

        if tag == "equal":
            continue

        old_start = max(0, i1 - context)
        old_end = min(len(old_lines), i2 + context)

        new_start = max(0, j1 - context)
        new_end = min(len(new_lines), j2 + context)

        old_block = [
            line.rstrip()
            for line in old_lines[old_start:old_end]
            if line.strip()
        ]

        new_block = [
            line.rstrip()
            for line in new_lines[new_start:new_end]
            if line.strip()
        ]

        results.append({

            "tag": tag,

            "old_block": old_block,

            "new_block": new_block

        })

    return results


# Simplified diff format for LLM input

def simplified_diff_for_llm(old_code, new_code, context=1):

    blocks = extract_changed_blocks_with_context(
        old_code,
        new_code,
        context=context
    )

    if not blocks:
        return "No changes."

    formatted_blocks = []

    for idx, block in enumerate(blocks, 1):

        old_text = "\n".join(block["old_block"]).strip() or "None"

        new_text = "\n".join(block["new_block"]).strip() or "None"

        formatted_blocks.append(

            f"Changed block {idx} before fix:\n"
            f"{old_text}\n\n"
            f"Changed block {idx} after fix:\n"
            f"{new_text}"

        )

    return "\n\n".join(formatted_blocks)



# Read dataset.pkl

def read_pkl(filepath):

    with open(filepath, "rb") as f:
        return pickle.load(f)



def write_pkl(obj, filepath):

    with open(filepath, "wb") as f:
        pickle.dump(obj, f)



print("Begin......!")
print("### Reading dataset.pkl")

dataset_pkl_path = r'FaultScape_data\Multi-view_data\dataset.pkl'

dataset_pkl = read_pkl(dataset_pkl_path)

faultType = dataset_pkl.keys()

print(faultType)

fault_types = list(dataset_pkl.keys())



for fault_type in fault_types:

    print(

        f"[code view] {fault_type}: "
        f"positive={len(dataset_pkl[fault_type]['positive'])}, "
        f"negative={len(dataset_pkl[fault_type]['negative'])}, "
        f"patch={len(dataset_pkl[fault_type]['patch'])}"

    )



all_data_count = sum(

    len(dataset_pkl[fault_type]['positive']) +
    len(dataset_pkl[fault_type]['negative'])

    for fault_type in fault_types

)

print(f"Total data count: {all_data_count}")



print(dataset_pkl['MoveStmt'].keys())

print(

    "Example positive snippet:",
    dataset_pkl['MoveStmt']['positive'][1]
)

print(

    "Example negative snippet:",
    dataset_pkl['MoveStmt']['negative'][1]
)

print(

    "Example patch snippet:",
    dataset_pkl['MoveStmt']['patch'][1]
)



'''
# 2. Build training datasets

We generate 4 datasets:

code_fl_data.json
type_fl_data.json
fix_fl_data.json
cause_fl_data.json

Each dataset follows a structured JSON format.
'''



# Dataset 1: code_fl_data.json

code_fl_data = []

for type1 in faultType:

    positive_list = dataset_pkl[type1]['positive']
    negative_list = dataset_pkl[type1]['negative']

    max_len = max(len(positive_list), len(negative_list))

    for i in range(max_len):

        if i < len(positive_list):

            code_fl_data.append({

                "content":
                    "The following code snippet contains a marked statement:\n"
                    + positive_list[i],

                "label": "1"

            })


        if i < len(negative_list):

            code_fl_data.append({

                "content":
                    "The following code snippet contains a marked statement:\n"
                    + negative_list[i],

                "label": "0"

            })



print("code_fl_data example:", code_fl_data[0:10])



# Dataset 2: type_fl_data.json

type_fl_data = []

for type1 in faultType:

    positive_list = dataset_pkl[type1]['positive']
    negative_list = dataset_pkl[type1]['negative']

    max_len = max(len(positive_list), len(negative_list))

    fault_desc = FAULT_TYPE_EXPLANATIONS.get(type1, type1)


    for i in range(max_len):

        if i < len(positive_list):

            type_fl_data.append({

                "content":

                    "The following code snippet contains a marked statement:\n"

                    + positive_list[i]

                    + " ****The fault type of the statement marked in this code snippet is: "

                    + type1 + " - " + fault_desc + "****",

                "label": "1"

            })


        if i < len(negative_list):

            type_fl_data.append({

                "content":

                    "The following code snippet contains a marked statement:\n"

                    + negative_list[i]

                    + " ****The fault type of the statement marked in this code snippet is: None, because the marked statement is not faulty****",

                "label": "0"

            })



print("type_fl_data example:", type_fl_data[0:10])



# Dataset 3: fix_fl_data.json

fix_fl_data = []

for type1 in faultType:

    positive_list = dataset_pkl[type1]['positive']
    patch_list = dataset_pkl[type1]['patch']
    negative_list = dataset_pkl[type1]['negative']

    max_len = max(len(positive_list), len(negative_list))


    for i in range(max_len):

        if i < len(positive_list) and i < len(patch_list):

            diff_result = simplified_diff_for_llm(

                positive_list[i],
                patch_list[i]

            )

            fix_fl_data.append({

                "content":

                    "The following code snippet contains a marked statement:\n"

                    + positive_list[i]

                    + "\n\nThe fix for the marked statement in this code snippet is:\n"

                    + diff_result,

                "label": "1"

            })


        if i < len(negative_list):

            fix_fl_data.append({

                "content":

                    "The following code snippet contains a marked statement:\n"

                    + negative_list[i]

                    + "\n\nThe fix for the marked statement in this code snippet is: None, because the marked statement is not faulty.",

                "label": "0"

            })



print("fix_fl_data example:", fix_fl_data[0:10])



# Save first 3 datasets

with open(r'FaultScape_data\Multi-view_data\code_fl_data.json', 'w') as f:
    json.dump(code_fl_data, f, indent=4)

with open(r'FaultScape_data\Multi-view_data\type_fl_data.json', 'w') as f:
    json.dump(type_fl_data, f, indent=4)

with open(r'FaultScape_data\Multi-view_data\fix_fl_data.json', 'w') as f:
    json.dump(fix_fl_data, f, indent=4)


print("Saved code_fl_data.json, type_fl_data.json, fix_fl_data.json successfully!")



# Dataset 4: cause_fl_data.json
# you can use the LLaMA-3-70B api or your local LLaMA-3-70B

from openai import OpenAI

API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key
API_BASE = "YOUR_API_BASE_URL_HERE"  # Replace with your actual API base URL

MODEL_NAME = "meta-llama/llama-3.3-70b-instruct"

client = OpenAI(
    base_url=API_BASE,
    api_key=API_KEY,
)

def build_bug_cause_prompt(

    code_snippet,
    fault_type,
    fault_desc,
    diff_result

):

    return f"""
You are given a buggy Java code snippet.

A suspicious statement is marked using rank2fixstart and rank2fixend.

Code snippet:
{code_snippet}

Fault type:
{fault_type} - {fault_desc}

Fix information:
{diff_result}

Task:
Identify the underlying cause of the bug in the marked statement.

Rules:
- Explain why the marked statement is incorrect in its context
- Do NOT mention the fix
- Do NOT repeat the fault type name
- Output ONLY one or two sentences
- Start with exactly:

The bug cause of this statement is:

Output:
The bug cause of the marked statement in this code snippet is: <bug cause>
""".strip()



def infer_bug_cause(

    code_snippet,
    fault_type,
    fault_desc,
    diff_result,
    max_retry=3

):

    prompt = build_bug_cause_prompt(

        code_snippet,
        fault_type,
        fault_desc,
        diff_result

    )

    for attempt in range(max_retry):

        try:

            response = client.chat.completions.create(

                model=MODEL_NAME,

                messages=[

                    {
                        "role": "system",
                        "content":
                            "You are a software defect analysis assistant. "
                            "Return only ONE sentence describing the bug cause."
                    },

                    {
                        "role": "user",
                        "content": prompt
                    }

                ],

                temperature=0.0,
                max_tokens=200

            )

            text = response.choices[0].message.content.strip()

            return text


        except Exception as e:

            print(f"[retry {attempt+1}] {e}")

            time.sleep(1)


    return "The bug cause of this statement is: the statement is inconsistent with the surrounding program logic."



# Real-time streaming save

stream_path = r"FaultScape_data\Multi-view_data\cause_fl_data_stream.jsonl"

def save_realtime(item):

    with open(stream_path, "a", encoding="utf-8") as f:

        f.write(json.dumps(item, ensure_ascii=False) + "\n")

        f.flush()

# Generate cause_fl_data

cause_fl_data = []

for type1 in faultType:

    positive_list = dataset_pkl[type1]['positive']
    patch_list = dataset_pkl[type1]['patch']
    negative_list = dataset_pkl[type1]['negative']

    max_len = max(len(positive_list), len(negative_list))

    fault_desc = FAULT_TYPE_EXPLANATIONS.get(type1, type1)



    for i in range(max_len):

        # positive samples (LLM required)

        if i < len(positive_list) and i < len(patch_list):

            diff_result = simplified_diff_for_llm(

                positive_list[i],
                patch_list[i]

            )

            bug_cause = infer_bug_cause(

                code_snippet=positive_list[i],
                fault_type=type1,
                fault_desc=fault_desc,
                diff_result=diff_result

            )

            item = {

                "content":

                    "The following code snippet contains a marked statement:\n"

                    + positive_list[i]

                    + "\n\nThe fault type of the marked statement in this code snippet is: "

                    + type1 + " - " + fault_desc

                    + "\n\nBug Cause Analysis-- "

                    + bug_cause,

                "label": "1"

            }

            cause_fl_data.append(item)

            save_realtime(item)

            print(f"[saved] {type1} positive {i}")



        # negative samples (no LLM required)

        if i < len(negative_list):

            item = {

                "content":

                    "The following code snippet contains a marked statement:\n"

                    + negative_list[i]

                    + "\n\nThe fault type of the marked statement in this code snippet is: None, because the marked statement is not faulty."

                    + "\n\nThe bug cause of the marked statement in this code snippet is: None, because the marked statement is not faulty.",

                "label": "0"

            }

            cause_fl_data.append(item)

            save_realtime(item)

            print(f"[saved] {type1} negative {i}")


final_path = r"FaultScape_data\Multi-view_data\cause_fl_data.json"

with open(final_path, "w", encoding="utf-8") as f:

    json.dump(

        cause_fl_data,
        f,
        indent=4,
        ensure_ascii=False

    )

print("\nSaved final dataset to:", final_path)
print("Saved streaming dataset to:", stream_path)

print("All datasets generated successfully!")
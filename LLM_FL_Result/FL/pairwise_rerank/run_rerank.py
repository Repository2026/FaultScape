import os
import json
import pickle
from config import TOP_K
from pairwise_rerank import pairwise_ranking



def load_pickle(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)
    
def get_index_from_id(id_to_index, subdir, code_id):
    id_list = id_to_index.get(subdir, [])
    if code_id in id_list:
        return id_list.index(code_id)
    return None

def extract_top_code_ids_and_snippets(parent_dir, statements_pkl_path, src_code_pkl_path, top_k=10):
    result = {}

    # id -> index 
    id_to_index = load_pickle(statements_pkl_path)

    # index -> code snippet
    code_data = load_pickle(src_code_pkl_path)

    for subdir in sorted(os.listdir(parent_dir)):
        sub_path = os.path.join(parent_dir, subdir)
        ranking_file = os.path.join(sub_path, "ranking.txt")

        if os.path.isdir(sub_path) and os.path.isfile(ranking_file):
            code_snippets = []

            with open(ranking_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if not parts:
                        continue
                    code_id = parts[0]
                    # print("code_id is:",code_id)
                    
                    index = get_index_from_id(id_to_index, subdir, code_id)
                    # print("index is :",index)

                    if index is None:
                        snippet = "[Code Not Found: ID not in statements.pkl]"
                    elif not (0 <= index < len(code_data[subdir])):
                        snippet = "[Code Not Found: Index out of range]"
                    else:
                        snippet = code_data[subdir][index]
                    # print("snippet is:",snippet)

                    code_snippets.append((code_id, snippet))

                    if len(code_snippets) >= top_k:
                        break

            result[subdir] = code_snippets

    return result


def load_test_failure_info(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f) 




if __name__ == "__main__":

    ids_folder = r".\LLM_FL_Result\FL\ranking_task\run_model\sus_pos_rerank_new"
    statements_pkl_path = r".\LLM_FL_Result\FL\d4j_data\statements.pkl"
    src_code_pkl_path = r".\AA-FaultScape_data\src_code.pkl"
    test_failure_info_path = r".\LLM_FL_Result\FL\pairwise_rerank\test_failure_info.json"

    print(f"Extracting top {TOP_K} code snippets using IDs from: {ids_folder}")
    results = extract_top_code_ids_and_snippets(ids_folder, statements_pkl_path, src_code_pkl_path, TOP_K)

    # for proj, pairs in list(results.items())[:2]:
    #     print(f"\n== {proj} ==")
    #     for cid, code in pairs:
    #         print(f"ID: {cid}\nCode:\n{code}\n")


    print(f"Loading test failure info from: {test_failure_info_path}")
    test_failure_info = load_test_failure_info(test_failure_info_path)

    test_failure_info_sorted = sorted(test_failure_info, key=lambda x: x.get("filename", ""))

    # for i, item in enumerate(test_failure_info_sorted, 1):
    #     filename = item.get("filename", "")
    #     content = item.get("content", "")
    #     print(f"{i}. File: {filename}\nContent:\n{content}\n{'-'*60}\n")


    print(f"Loaded {len(results)} projects' code snippets and test failure info, start pairwise ranking...")

    # for project in sorted(results.keys()):
    for project in sorted(results.keys()):
        print(f"\n--- Ranking for project: {project} ---")
        
        snippets = results[project]  # [(code_id, snippet), ...]
        snippets_only = [snippet for _, snippet in snippets] 
        
        failure_info_for_project = [
            item for item in test_failure_info if item.get("filename") == project
        ]
        
        if not failure_info_for_project:
            print(f"Warning: No test failure info found for project {project}. Skipping ranking.")
            continue

        # print("the snippet is:",snippets_only)
        # print("the failure info is:",failure_info_for_project[0]['content'])


        prior_probs =   [1.0 / len(snippets_only)] * len(snippets_only)   
        ranking = pairwise_ranking(failure_info_for_project[0]['content'], snippets_only, prior_probs, alpha=0.5)

        print(f"Final ranking for {project}:")
        # for rank_idx, (idx, score) in enumerate(ranking, 1):
        #     code_id = snippets[idx][0]
        #     snippet_preview = snippets[idx][1].replace("\n", " ")[:80]
        #     print(f"Rank {rank_idx}: Snippet #{idx} (ID: {code_id}) - Score: {score:.4f}")
        #     # print(f"    Preview: {snippet_preview}")

        # === save ranking file ===
        output_dir = os.path.join(
            "F:/001_LLM_for_FL/AA-FaultScape/FaultScape/LLM_FL_Result/FL/pairwise_rerank/sus_pos_rerank_new_PCR",
            project
        )
        os.makedirs(output_dir, exist_ok=True)  
        output_file = os.path.join(output_dir, "ranking.txt")

        with open(output_file, "w", encoding="utf-8") as f:
            for rank_idx, (idx, score) in enumerate(ranking, 1):
                code_id = snippets[idx][0]
                snippet_preview = snippets[idx][1].replace("\n", " ")[:80]
                print(f"Rank {rank_idx}: Snippet #{idx} (ID: {code_id}) - Score: {score:.4f}")
                f.write(f"{code_id} {score}\n")


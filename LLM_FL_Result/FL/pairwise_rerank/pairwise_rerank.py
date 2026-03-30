# pairwise_rerank.py

import openai
import itertools
import json
from together import Together
from config import PROMPT_TEMPLATE_PATH


togather_API_key = "" # your api
client = Together(api_key=togather_API_key)


from config import API_KEY, API_BASE, MODEL_NAME, TOP_K

openai.api_key = API_KEY
openai.api_base = API_BASE

def build_prompt(test_failure_context, code_snippet_1, code_snippet_2, template_path=PROMPT_TEMPLATE_PATH):

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    return template.format(
        test_failure_context=test_failure_context,
        code_snippet_1=code_snippet_1,
        code_snippet_2=code_snippet_2
    ).strip()



def query_model(prompt):
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3-70B",
            messages=[
                {"role": "system", "content": "You are an expert in fault localization of Java code."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, 
            top_p=1.0,
            seed=42,
        )
        if response.choices:
            return response.choices[0].message.content.strip()
        else:
            # print("No valid response from Together API.")
            return None
    except Exception as e:
        # print(f"Together API error: {e}")
        return None

import re

def parse_response(response_text):
    
    compare_result = ""
    compare_confidence = 0.0

    try:
        response = json.loads(response_text)
        compare_result = response.get("Compare_Result", "").strip()
        compare_confidence = float(response.get("Compare_Confidence", 0.0))
    except Exception:
        result_match = re.search(r'"Compare_Result"\s*:\s*"([^"]+)"', response_text)
        compare_result = result_match.group(1).strip() if result_match else ""

        conf_match = re.search(r'"Compare_Confidence"\s*:\s*([0-9.]+)', response_text)
        compare_confidence = float(conf_match.group(1)) if conf_match else 0.0

    if compare_result not in ("First", "Second"):
        return "", 0.0

    return compare_result, compare_confidence

def extract_context(content, marker_start="rank2fixstart", marker_end="rank2fixend", context_size=1000):
    start_pos = content.find(marker_start)
    end_pos = content.find(marker_end)

    if start_pos == -1 or end_pos == -1:
        return content
    
    start_context = content[max(0, start_pos - context_size):start_pos]  
    end_context = content[end_pos + len(marker_end):end_pos + len(marker_end) + context_size] 

    if (end_pos + len(marker_end) + context_size) < len(content):
        content_with_context = start_context + marker_start + content[start_pos + len(marker_start):end_pos] + marker_end + end_context + "... \n " + content[-240:-1]
    else:
        content_with_context = start_context + marker_start + content[start_pos + len(marker_start):end_pos] + marker_end + end_context
    
    
    return content_with_context
    


def pairwise_ranking(test_failure_context, code_snippets, prior_probs, alpha=0.5):


    # print("====the test failure context is:",test_failure_context)
    # print("=====the code snippets is:",code_snippets)

    scores = [0.0] * len(code_snippets)
    counts = [0] * len(code_snippets)

    for i, j in itertools.combinations(range(len(code_snippets)), 2):

        code_snippets[i].replace("rank2fixstart","<//rank2fixstart//>").replace("rank2fixend","<//rank2fixend//>")
        code_snippets[j].replace("rank2fixstart","<//rank2fixstart//>").replace("rank2fixend","<//rank2fixend//>")
        code_snippets[i] = extract_context(code_snippets[i])
        code_snippets[j] = extract_context(code_snippets[j])


        prompt = build_prompt(test_failure_context, code_snippets[i], code_snippets[j])
        print("the code snippet 1 is:",code_snippets[i])
        print("the code snippet 2 is:",code_snippets[j])

        # print ("====the prompt is:",prompt)


        response = query_model(prompt)


        # print("====the response is:",response)
        # print("====the response is:", response.encode('utf-8', errors='ignore').decode('utf-8'))

        if response is None:
            print(" Warning: response is None.")
        else:
            # try:
            #     print("====the response is:", response.encode('gbk', errors='replace').decode('gbk'))
            # except Exception as e:
            #     print(f"[print error] {e}")

            try:
                print("====the response is:", response.encode('utf-8', errors='replace').decode('utf-8'))
            except Exception as e:
                print(f"[print error] {e}")



        '''
            ====the response is: ```json
            {
            "Two_codeStatement": {
                "First_code_snippet": "line 1: return \" title=\"\" + toolTipText + \"\" alt=\"\"\"; // This line does not properly escape the double quotes in the toolTipText",
                "Second_code_snippet": "line 1: super(); // This line is a constructor call and does not seem to be related to the generation of the tool tip fragment"
            },
            "Compare_bugCause": "The failing test `org.jfree.chart.imagemap.junit.StandardToolTipTagFragmentGeneratorTests.testGenerateURLFragment()` indicates that the generated tool tip fragment does not match the expected output. The expected output is `title=\"Series [&quot;A&quot;], 100.0\" alt=\"\"` but the actual output is `title=\"Series [\"A\"], 100.0\" alt=\"\"`. This suggests that the issue lies in the way the `toolTipText` is being processed. In the first code snippet, the line `return \" title=\"\" + toolTipText + \"\" alt=\"\"\";` does not properly escape the double quotes in the `toolTipText`. This would cause the generated HTML to be invalid, leading to the observed error. In contrast, the second code snippet is a constructor call and does not seem to be related to the generation of the tool tip fragment. The probability score difference is not provided, but based on the failing test and the code, the first snippet is more likely to contain the bug.",
            "Compare_Result": "First",
            "Compare_Confidence": 0.9
            }
        '''


        if response is None:
            continue


        result, confidence = parse_response(response)
        # print(result)     
        # print(confidence) 



        if result == "First":
            scores[i] += confidence
            counts[i] += 1
        elif result == "Second":
            scores[j] += confidence
            counts[j] += 1
        else:
            pass

    max_score = max(scores) if max(scores) > 0 else 1.0


    # print("Scores:", scores)
    # print("Counts:", counts)
    # print("Max score:", max_score)
    # print("Prior probs:", prior_probs)




    final_scores = []
    for i in range(len(scores)):
        s_norm = scores[i] / max_score
        if counts[i] == 0:
            final_score = prior_probs[i]
        else:
            final_score = alpha * prior_probs[i] + (1 - alpha) * s_norm
        final_scores.append(final_score)

    ranked = sorted(enumerate(final_scores), key=lambda x: x[1], reverse=True)
    return ranked
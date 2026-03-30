import os
import pickle

# data merge
src_root = r".\LLM_FL_Result\FL\pairwise_rerank\sus_pos_rerank_new_PCR"
dst_root = r".\LLM_FL_Result\FL\pairwise_rerank\sus_pos_rerank_new_all"

for subdir in os.listdir(src_root):
    src_txt_path = os.path.join(src_root, subdir, "ranking.txt")
    dst_txt_path = os.path.join(dst_root, subdir, "ranking.txt")

    if not os.path.isfile(src_txt_path) or not os.path.isfile(dst_txt_path):
        print("skip")
        continue  # skip non-existing file


    # read source file and add extra weight to Top-10 ( distinguish PCR algorithm reranking from original ranking)
    with open(src_txt_path, "r", encoding="utf-8") as f:
        src_lines = f.readlines()
    
    top_lines = []
    for line in src_lines[:10]:
        parts = line.strip().rsplit(" ", 1)
        if len(parts) == 2:
            try:
                new_score = float(parts[1]) + 1.0
                top_lines.append(f"{parts[0]} {new_score:.10f}\n")  
            except ValueError:
                top_lines.append(line)    
        else:
            top_lines.append(line) 

    with open(dst_txt_path, "r", encoding="utf-8") as f:
        dst_lines = f.readlines()

    n = len(top_lines)
    new_lines = top_lines + dst_lines[n:]

    with open(dst_txt_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

print("data merge done!")



def load_from_file(file_path):  
	with open(file_path, "rb") as file:
		return pickle.load(file)


# calculate all result
for task_id in range(1, 11):

	root = r".\LLM_FL_Result\FL\ranking_task\run_model\data/{}/".format(task_id)
	test_x_data = load_from_file(root + 'test/x.pkl')
	test_y_data = load_from_file(root + 'test/y.pkl')

	with open(r".\LLM_FL_Result\FL\d4j_data\statements.pkl", "rb") as file:
		statements = pickle.load(file)

	with open(r".\LLM_FL_Result\FL\ranking_task\faulty_statement_set.pkl", "rb") as file:
		faulty_statements = pickle.load(file)
		# print("faulty_statements_set is:",(faulty_statements) )

	with open(r".\LLM_FL_Result\FL\pairwise_rerank\result/{}.txt".format(task_id), "w") as result_file:

		for version in test_x_data:
			# print("the version is:", version)
			result_file.write("==={}\n".format(version))

			file_path = r".\LLM_FL_Result\FL\pairwise_rerank\sus_pos_rerank_new_all/{}/ranking.txt".format(version)
			
			# init sorted list
			sorted_sus_list = []

			with open(file_path, 'r', encoding='utf-8') as file:
				for line in file:
					line = line.strip()
			
					*line_part, score = line.split()
					line_text = " ".join(line_part)  
					score = float(score) 
					sorted_sus_list.append((line_text, score))

			rerank_sus_lines = [line for (line, score) in sorted_sus_list]  # line
			rerank_sus_scores = [float(score) for (line, score) in sorted_sus_list] # score

			current_faulty_statements = faulty_statements[version]

			for one_position_set in current_faulty_statements: 
				current_min_index = 1e8
				for buggy_line in one_position_set: 
					if buggy_line not in rerank_sus_lines: 
						continue
					
					buggy_index = len(rerank_sus_scores) - rerank_sus_scores[::-1].index(rerank_sus_scores[rerank_sus_lines.index(buggy_line)])
					if buggy_index < current_min_index:
						current_min_index = buggy_index
				if current_min_index == 1e8:
					continue

				result_file.write(str(current_min_index) + "\n")




def parse_model_output_file():    
	total_result = []
	version_list = []
	for group_index in range(1, 11):
		with open(r".\LLM_FL_Result\FL\pairwise_rerank\result/{}.txt".format(group_index), "r") as file:
			content = file.readlines()
		current_result = []
		first_flag = True
		for line in content:
			line = line.strip()
			if line.startswith("==="):
				if len(version_list) > 0:
					if len(current_result) == 0:
						if not first_flag:
							version_list.pop()
					else:
						total_result.append(current_result[:])
						current_result = []
				version_list.append(line.replace("===", ""))
			else:
				current_result.append(float(line))
			first_flag = False
		if len(current_result) > 0:
			total_result.append(current_result[:])
		else:
			version_list.pop()
	return total_result, version_list


for k in range(1):
    print ("no. {} group".format(k+1))
    result_dict = {}
    result_dict['top1_total_list'] = []
    result_dict['top3_total_list'] = []
    result_dict['top5_total_list'] = []
    result_dict['MFR_list'] = []
    result_dict['MAR_list'] = []

    for i in range(1):
        print("no {}group, no.{}".format(k+1,i+1) )

        # # data pre-processing for cross-validation setup 
        # status = os.system("python3 pipeline_group.py >/dev/null 2>&1")
        # assert status == 0

        # # 10-fold cross-validation training and testing
        # for group_index in range(1,11):  
        #     status = os.system("python3 train.py {} >/dev/null 2>&1".format(group_index))
        #     assert status == 0

        total_result, version_list = parse_model_output_file()

        top1_total = 0
        top3_total = 0
        top5_total = 0
        all_position_total = []
        first_position_total = []

        # print("\nStatistics for each project.")
        for project in ["Chart", "Closure", "Math", "Mockito", "Lang", "Time"]:

            top1 = 0
            top3 = 0
            top5 = 0
            all_position = []
            first_position = []

            for i, version in enumerate(version_list):
                if not version.startswith(project):
                    continue
                bugs = total_result[i]
                rank = []
                for bug in bugs:
                    rank.append(bug)
                min_rank = min(rank)
                avg_rank = sum(rank) / (len(rank))

                if min_rank <= 1:
                    top1 += 1
                if min_rank <= 3:
                    top3 += 1
                if min_rank <= 5:
                    top5 += 1
                first_position.append(min_rank)
                all_position.append(avg_rank)

            top1_total += top1
            top3_total += top3
            top5_total += top5
            all_position_total += all_position
            first_position_total += first_position

        print("Statistics for all projects.")
        print("=" * 20)
        print("Top1\t{}".format(top1_total))
        print("Top3\t{}".format(top3_total))
        print("Top5\t{}".format(top5_total))
        print("MFR\t{}".format(round(sum(first_position_total) / len(first_position_total), 2)))
        print("MAR\t{}".format(round(sum(all_position_total) / len(all_position_total), 2)))


        
import os
import pickle
import numpy as np
import re

data_path = "./FaultScape/LLM_FL_Result/FL/d4j_data"

spectrum_path = os.path.join(data_path, "spectrum.pkl")
with open(spectrum_path, "rb") as file:
	spectrum = pickle.load(file)

mutation_path = os.path.join(data_path, "mutation.pkl")
with open(mutation_path, "rb") as file:
	mutation = pickle.load(file)

LLM_result_path = "./FaultScape/LLM_FL_Result/FL/ranking_task/LLM_semantic.pkl"
with open(LLM_result_path, "rb") as file:
	LLM_result_pkl = pickle.load(file)

with open("./FaultScape/LLM_FL_Result/FL/ranking_task/faulty_statement_index.pkl", "rb") as file:   
	faulty_stmts = pickle.load(file)

print("faulty_stmts: ",faulty_stmts)




# add spectrum, mutation features 
for version_00 in LLM_result_pkl.keys():
	i = 0
	for sp, mu, llm in zip(  spectrum[version_00], mutation[version_00], LLM_result_pkl[version_00]      ):
		
		spectrum_sum = sum(sp)
		mutation_sum = sum(mu)
		llm_sum = sum(llm)

		LLM_result_pkl[version_00][i][0] = ( llm[0] )
		LLM_result_pkl[version_00][i][1] = ( llm[1] )
		LLM_result_pkl[version_00][i][2] = ( llm[2] )
		LLM_result_pkl[version_00][i][3] = ( llm[3] )


		LLM_result_pkl[version_00][i].append (llm[0])
		LLM_result_pkl[version_00][i].append (llm[1])
		LLM_result_pkl[version_00][i].append (llm[2])
		LLM_result_pkl[version_00][i].append (llm[3])


		LLM_result_pkl[version_00][i].append (spectrum_sum + llm_sum)
		LLM_result_pkl[version_00][i].append (mutation_sum + llm_sum)
		i = i+1



with open("./FaultScape/LLM_FL_Result/FL/d4j_data/src_code.pkl", "rb") as file1:   
        src_code = pickle.load(file1)


# parse labels (0 for faulty and 1 for non-faulty)   
label = {}
for version in LLM_result_pkl:
	current_label = [1] * len(spectrum[version]) 
	for faulty_index in faulty_stmts[version]:
			current_label[faulty_index - 1] = 0  
	label[version] = current_label    

# parse data
data = {}
for version in LLM_result_pkl:
	data[version] = []
	# init
	for i in range(len(spectrum[version])):  
		data[version].append([])


	# spectrum features (3-dim: ochiai, dstar, tarantula) 
	current_spectrum = np.array(spectrum[version])
	# print(current_spectrum)   
	for i in range(3):
		scores = current_spectrum[:, i].tolist() 
		sorted_scores = sorted(scores, reverse=True)    
		ranking_scores = [sorted_scores[::-1].index(score) / len(scores) for score in scores]   

		for i, score in enumerate(ranking_scores):
			data[version][i].append(score)
		
	current_mutation = np.array(mutation[version])
	for i in range(4):
		scores = current_mutation[:, i].tolist()
		sorted_scores = sorted(scores, reverse=True)
		ranking_scores = [sorted_scores[::-1].index(score) / len(scores) for score in scores]
		for i, score in enumerate(ranking_scores):
			data[version][i].append(score)
		
	current_semantic = np.array(LLM_result_pkl[version])
	for i in range(10):
		scores = current_semantic[:, i].tolist()
		sorted_scores = sorted(scores, reverse=True)
		ranking_scores = [sorted_scores[::-1].index(score) / len(scores) for score in scores]
		for i, score in enumerate(ranking_scores):
			data[version][i].append(score)
		

	assert (len(data[version]) == len(spectrum[version]) )


# save data and label
with open("./FaultScape/LLM_FL_Result/FL/ranking_task/run_model/x.pkl", "wb") as file:
	pickle.dump(data, file)
with open("./FaultScape/LLM_FL_Result/FL/ranking_task/run_model/y.pkl", "wb") as file:
	pickle.dump(label, file)

print("completes the combination of the dynamic features and the four llms features, 3+4+(llm1,llm2,llm3,llm4).")


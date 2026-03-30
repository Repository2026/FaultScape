#!/bin/sh

cd   ./FaultScape/LLM_FL_Result/FL/ranking_task/
python ./FaultScape/LLM_FL_Result/FL/ranking_task/gen_data_for_MLP.py

cd ./FaultScape/LLM_FL_Result/FL/ranking_task/run_model
python ./FaultScape/LLM_FL_Result/FL/ranking_task/run_model/run_group.py
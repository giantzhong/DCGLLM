# 1. python data_preprocess/prepare_mimiciii_data.py --target CKD
python data_preprocess/prepare_mimiciv_data.py --target CKD 

# 2. python data_preprocess/utils.py --dataset MIMIC3 --target CKD
python data_preprocess/utils.py --dataset MIMIC4 --target CKD

# 3. python simple_build_graph/build_graph_score.py --dataset MIMIC4 --target CKD
# python simple_build_graph/build_graph_score.py --dataset MIMIC3 --target CKD
# python simple_build_graph/build_graph_score.py --dataset HuaDong --target dm
python simple_build_graph/build_graph.py 

# 4. python data_preprocess/data_postprocess.py --dataset MIMIC3
# python data_preprocess/data_postprocess.py --dataset MIMIC3 --target CKD
# python data_preprocess/data_postprocess.py --dataset HuaDong --target dm
python data_preprocess/data_postprocess.py

# 5. python data_preprocess/dataset_merge.py --dataset MIMIC3 --target CKD
# python data_preprocess/dataset_merge.py --dataset HuaDong --target dm
python data_preprocess/dataset_merge.py

# 6.
python main_dcgllm.py

# PPL+Biomistral7B
# graph_raw_BioMistral-7B.pkl
# graph_raw_norm_BioMistral-7B.pkl

# PPL+HuatuoGPT_II
# graph_HuatuoGPT_II.pkl
# graph_norm_HuatuoGPT_II.pkl

# Score+Biomistral7B
# graph_BioMistral-7B.pkl
# graph_norm_BioMistral-7B.pkl
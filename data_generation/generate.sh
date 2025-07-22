python create_queries.py --dataset CODEX --index_only
python create_queries.py --dataset CODEX --gen_train --save_name
python create_fol.py --dataset CODEX --gen_train
python create_nl.py --dataset CODEX --gen_train
python create_training_data.py --dataset CODEX --gen_train

python create_queries.py --dataset UMLS --index_only
python create_queries.py --dataset UMLS --gen_train --save_name
python create_fol.py --dataset UMLS --gen_train
python create_nl.py --dataset UMLS --gen_train
python create_training_data.py --dataset UMLS --gen_train

python create_queries.py --dataset FB15k-237 --gen_train --save_name
python create_fol.py --dataset FB15k-237 --gen_train
python create_nl.py --dataset FB15k-237 --gen_train
python create_training_data.py --dataset FB15k-237 --gen_train
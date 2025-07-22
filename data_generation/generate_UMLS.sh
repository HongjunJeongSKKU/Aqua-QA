python create_queries.py --dataset UMLS --index_only
python create_queries.py --dataset UMLS --gen_train --save_name
python create_fol.py --dataset UMLS --gen_train
python create_nl.py --dataset UMLS --gen_train
python create_training_data.py --dataset UMLS --gen_train
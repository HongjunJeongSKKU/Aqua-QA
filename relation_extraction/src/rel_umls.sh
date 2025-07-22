python run_query_extract_w_neg.py --data_path UMLS --train_data_name synthetic_drop_dup --exp_name UMLS_neg_8e-4_xl_1e-3 --model_name T5-xl --constrain_token --num_epochs 3 --lr 8e-4 --l2 1e-3 --train_batch_size 32
python generate_kg_info_ori_q_neg.py --data_path UMLS --exp_name UMLS_neg_8e-4_xl_1e-3 --checkpoint 2256 --data_type train
python generate_kg_info_ori_q_neg.py --data_path UMLS --exp_name UMLS_neg_8e-4_xl_1e-3 --checkpoint 2256 --data_type valid
python generate_kg_info_ori_q_neg.py --data_path UMLS --exp_name UMLS_neg_8e-4_xl_1e-3 --checkpoint 2256 --data_type test

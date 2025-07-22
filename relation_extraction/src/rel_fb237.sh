python run_query_extract_w_neg.py --data_path FB15k-237 --train_data_name synthetic_drop_dup --exp_name fb237_neg_1e-4_xl_1e-3 --model_name T5-xl --constrain_token --num_epochs 3 --lr 1e-4 --l2 1e-3
python generate_kg_info_ori_q_neg.py --data_path FB15k-237 --exp_name fb237_neg_1e-4_xl_1e-3 --checkpoint 36567 --data_type train
python generate_kg_info_ori_q_neg.py --data_path FB15k-237 --exp_name fb237_neg_1e-4_xl_1e-3 --checkpoint 36567 --data_type valid
python generate_kg_info_ori_q_neg.py --data_path FB15k-237 --exp_name fb237_neg_1e-4_xl_1e-3 --checkpoint 36567 --data_type test
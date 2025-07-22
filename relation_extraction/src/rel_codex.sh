python run_query_extract_w_neg.py --data_path CODEX --train_data_name synthetic_drop_dup --exp_name codex_neg_4e-4_xl_1e-4 --model_name T5-xl --constrain_token --num_epochs 5 --lr 4e-4 --l2 1e-4
python generate_kg_info_ori_q_neg.py --data_path CODEX --exp_name codex_neg_4e-4_xl_1e-4 --checkpoint 3365 --data_type train
python generate_kg_info_ori_q_neg.py --data_path CODEX --exp_name codex_neg_4e-4_xl_1e-4 --checkpoint 3365 --data_type valid
python generate_kg_info_ori_q_neg.py --data_path CODEX --exp_name codex_neg_4e-4_xl_1e-4 --checkpoint 3365 --data_type test
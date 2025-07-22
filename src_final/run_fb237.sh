python train.py --exp_name FB15k-237 --use_scheduler cosine --train_batch_size 1024 --lr 4e-4 --do_train --aligner_path FB237_ex --do_test_new --num_epochs 100 --smoothing 0.9 --neg_layer_learnable --data_path FB15k-237 --path_exp_name fb237_neg_1e-4_xl_1e-3 --path_checkpoint 36567 --kge_hidden_size 2000 --kge_path FB237-complex


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import numpy as np
import json
import random
import torch
import numpy as np

import argparse 
from utils import get_llm_model, dict_to_namespace, get_nl_dataset_all
from dataset import TrainDataset, TestDataset
from align_kg_w_lm_new_neg.src.KGE.model_src import Complex
from models.reasoner import MainReasoner
from managers import Trainer
import logging
import time
from align_kg_w_lm_new_neg.src.models.models import Ailgner_self_mask_modality

import pickle


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(params):
    if params.do_train: 

        llm_model, tokenizer, llm_config = get_llm_model(params.llm_model_name)
        params.llm_hidden_size = llm_config.hidden_size

        if params.data_path.split("/")[-1] in ['UMLS', 'CODEX', 'FB15k-237']:
            with open(os.path.join(params.data_path,"ent2id.pkl"), "rb") as f:
                ent2id = pickle.load(f)
            with open(os.path.join(params.data_path,"rel2id.pkl"), "rb") as f:
                rel2id = pickle.load(f)                
            logging.info(f"Train data preprocessing start")
            b_start_time = time.time()
            train_data = get_nl_dataset_all(data_path = params.data_path, exp_name = params.path_exp_name, checkpoint = params.path_checkpoint, mode='train_sampled' if params.do_sampled else 'train', params = params, rel2id = rel2id, ent2id = ent2id, tokenizer = tokenizer)
            train_dataset = TrainDataset(train_data, tokenizer = tokenizer, ent2id = ent2id, rel2id = rel2id, negative_sample_size = params.negative_sample_size, max_tokens = params.max_tokens, params = params)
            end_time = time.time() - b_start_time
            logging.info(f"Train data constructing done, time: {end_time}")
            
            logging.info(f"Valid data preprocessing start")
            b_start_time = time.time()
            valid_data = get_nl_dataset_all(data_path = params.data_path, exp_name = params.path_exp_name, checkpoint = params.path_checkpoint, mode='valid', params = params, rel2id = rel2id, ent2id = ent2id, tokenizer = tokenizer)
            valid_dataset = TestDataset(valid_data, tokenizer = tokenizer, ent2id = ent2id, rel2id = rel2id, max_tokens = params.max_tokens, params = params, mode='valid')
            end_time = time.time() - b_start_time
            logging.info(f"Valid data constructing done, time: {end_time}")
        else:
            raise NotImplementedError

        test_dataset = None
        params.num_ents = len(ent2id)
        params.num_rels = len(rel2id)

        kge_model = Complex(params = params)
        kge_model.load_from_ckpt_path(params.kge_path)
        with open(os.path.join(params.aligner_path, "params.json"), "r") as f:
            align_params = dict_to_namespace(json.load(f))

        aligner = Ailgner_self_mask_modality(params = align_params, llm_model = llm_model, kge_model = kge_model)


        aligner.load_state_dict(torch.load(os.path.join(params.aligner_path, "best_model/best_model.pth" if not params.aligner_epoch else f"model_epoch_{params.aligner_epoch}.pth")), strict=False)
        total_params = sum(p.numel() for p in aligner.parameters())
        learnable_params = sum(p.numel() for p in aligner.parameters() if p.requires_grad)
        #print(f'aligner Total parameters: {total_params}')
        print(f'aligner Learnable parameters: {learnable_params}')

        with open(os.path.join(params.exp_dir, "params.json"), 'w') as f:
            json.dump(dict(vars(params)), f, indent=4)

        if torch.cuda.is_available():
            params.device = torch.device('cuda')
        model = MainReasoner(params=params, aligner = aligner, tokenizer = tokenizer)
        total_params = sum(p.numel() for p in model.parameters())
        learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f'Total parameters: {total_params}')
        logging.info(f'Learnable parameters: {learnable_params}')

        model = model.to(params.device)
        trainer = Trainer(model = model, params = params, train = train_dataset, valid = valid_dataset, test = test_dataset)
        trainer.train()

    if params.do_test_new:
        if params.do_train:
            model.to('cpu')
            torch.cuda.empty_cache()
        with open(os.path.join(params.exp_dir, "params.json"), 'r') as f:
            saved_params = dict_to_namespace(json.load(f))
        if torch.cuda.is_available():
            saved_params.device = torch.device('cuda')
        llm_model, tokenizer, llm_config = get_llm_model(saved_params.llm_model_name)

        kge_model = Complex(params = saved_params)   
        kge_model.load_from_ckpt_path(saved_params.kge_path)
        with open(os.path.join(saved_params.aligner_path, "params.json"), "r") as f:
            align_params = dict_to_namespace(json.load(f))

        aligner = Ailgner_self_mask_modality(params = align_params, llm_model = llm_model, kge_model = kge_model)
        aligner.load_state_dict(torch.load(os.path.join(saved_params.aligner_path, 'best_model/best_model.pth' if not saved_params.aligner_epoch else f"model_epoch_{params.aligner_epoch}.pth")), strict=False)

        best_model = MainReasoner(params=saved_params, aligner=aligner, tokenizer = tokenizer)

        model_state_dict = torch.load(os.path.join(saved_params.exp_dir, 'best_model/best_model.pth'))

        best_model.load_state_dict(model_state_dict, strict=False)
        best_model = best_model.to(saved_params.device)

        if saved_params.data_path.split("/")[-1] in ['UMLS', 'CODEX', 'FB15k-237']:
            with open(os.path.join(saved_params.data_path,"ent2id.pkl"), "rb") as f:
                ent2id = pickle.load(f)
            with open(os.path.join(saved_params.data_path,"rel2id.pkl"), "rb") as f:
                rel2id = pickle.load(f)                

            
            logging.info(f"Test data preprocessing start")
            b_start_time = time.time()
            test_data = get_nl_dataset_all(data_path = saved_params.data_path, exp_name = saved_params.path_exp_name, checkpoint = saved_params.path_checkpoint, mode='test', params = saved_params, rel2id = rel2id, ent2id = ent2id, tokenizer = tokenizer)
            test_dataset = TestDataset(test_data, tokenizer = tokenizer, ent2id = ent2id, rel2id = rel2id, max_tokens = saved_params.max_tokens, params = saved_params, mode='test')
            end_time = time.time() - b_start_time
            logging.info(f"Test data constructing done, time: {end_time}")
        else:
            raise NotImplementedError

        trainer = Trainer(model = best_model, params = saved_params, train = None, valid = None, test = test_dataset)
        trainer.test_new()

        if saved_params.data_path.split("/")[-1] in ['CODEX', 'FB15k-237'] and os.path.exists(os.path.join(saved_params.data_path, f'path_from_{saved_params.path_exp_name}_{saved_params.path_checkpoint}',f"{saved_params.data_path.split('/')[-1]}-test_hard-subq_sampled_max{saved_params.num_max_sq}_w_path.parquet")):
            
            test_data = get_nl_dataset_all(data_path = saved_params.data_path, exp_name = saved_params.path_exp_name, checkpoint = saved_params.path_checkpoint, mode='test_hard', params = saved_params, rel2id = rel2id, ent2id = ent2id, tokenizer = tokenizer)
            test_dataset = TestDataset(test_data, tokenizer = tokenizer, ent2id = ent2id, rel2id = rel2id, max_tokens = saved_params.max_tokens, params = saved_params, mode='test_hard')

            trainer = Trainer(model = best_model, params = saved_params, train = None, valid = None, test = test_dataset)
            trainer.test_new()


        




if __name__ == '__main__':



    # Training regime params
    parser = argparse.ArgumentParser(description='TransE model')

    parser.add_argument("--lr", type=float, default=4e-4,
                        help="Learning rate of the optimizer")
    parser.add_argument("--l2", type=float, default=1e-2,
                        help="Regularization constant for GNN weights")
    parser.add_argument('--seed', default=0,
                        type=int, help='Seed for randomization')
    parser.add_argument('--max_length', default=128,
                        type=int, help='max_length')
    parser.add_argument('--train_batch_size', default=1024,
                        type=int, help='train_batch_size')
    parser.add_argument('--eval_batch_size',default=256,
                        type=int, help='eval_batch_size')
    parser.add_argument('--data_path', default="NL-FB15k-237_prompt_2",
                        type=str, help='data_path')
    parser.add_argument('--path_exp_name', default="t5_xl_query_fin_neg_1e-4",
                        type=str, help='data_path')
    parser.add_argument('--path_checkpoint', default="36681",
                        type=int, help='data_path')
    parser.add_argument('--llm_model_name', default="st",
                        type=str, help='model_name')
    parser.add_argument('--exp_name', default="ex",
                        type=str, help='model_name')
    parser.add_argument('--kge_path', default='only_rel', #"ComplEx-2024.07.20-02_59_46", #Complex_nl_init_norm_False",
                        type=str, help='model_name')
    parser.add_argument('--aligner_path', default='aligner_2',
                        type=str, help='model_name')
    parser.add_argument('--aligner_epoch', default=0,
                        type=int, help='model_name')
    parser.add_argument('--negative_sample_size', default=512,
                        type=int, help='negative_sample_size')
    parser.add_argument('--max_tokens', default=80,
                        type=int, help='max_tokens')
    parser.add_argument('--valid_per_epoch', default=5,
                        type=int, help='valid_per_epoch')
    parser.add_argument('--save_per_epoch', default=50,
                        type=int, help='valid_per_epoch')
    parser.add_argument('--num_heads', default=16,
                        type=int, help='valid_per_epoch')
    parser.add_argument('--reasoner_hidden_size', default=1024,
                        type=int, help='valid_per_epoch')
    parser.add_argument('--kge_hidden_size', default=2000,
                        type=int, help='valid_per_epoch')
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Learning rate of the optimizer")
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1,
                        help="Learning rate of the optimizer")
    parser.add_argument('--num_epochs', default=15,
                        type=int, help='valid_per_epoch')
    parser.add_argument('--num_hidden_layers', default=8,
                        type=int, help='valid_per_epoch')
    parser.add_argument('--num_max_sq', default=5,
                        type=int, help='valid_per_epoch')
    parser.add_argument('--num_max_path', default=3,
                        type=int, help='valid_per_epoch')
    parser.add_argument('--num_max_path_len', default=4,
                        type=int, help='valid_per_epoch')
    parser.add_argument('--num_info_rel', default=3,
                        type=int, help='valid_per_epoch')
    parser.add_argument('--train_log_step', default=100,
                        type=int, help='valid_per_epoch')
    parser.add_argument("--smoothing", type=float, default=0.9,
                        help="Learning rate of the optimizer")

    parser.add_argument('--do_test_new', action='store_true', help="do test")
    parser.add_argument('--do_train', action='store_true', help="do test")
    parser.add_argument('--do_sampled', action='store_true', help="do test")
    parser.add_argument('--use_adapter', action='store_true', help="do test")
    parser.add_argument('--re_adj', action='store_true', help="do test")
    parser.add_argument('--non_valid', action='store_true', help="do test")
    parser.add_argument("--topk", type=int, default=3,
                        help="Learning rate of the optimizer")
    parser.add_argument('--layer_norm_eps', default=1e-12, type=float)
    parser.add_argument('--num_proc', default=16,
                        type=int, help='valid_per_epoch')


    parser.add_argument('--drop_dup', type=str2bool, default = True)
    parser.add_argument('--neg_layer_learnable', action='store_true')
    parser.add_argument('--use_scheduler', choices=['cosine', 'not_use'], default = 'cosine')
    parser.add_argument('--warm_up_ratio', default=0.01, type=float)



    params = parser.parse_args()
    params.llm_model_name = params.llm_model_name.lower()

    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)

    params.llm_model_name = params.llm_model_name.lower()
    params.aligner_path = os.path.join(f"align_kg_w_lm_new_neg/save_path/{params.data_path}", params.aligner_path)

    params.data_path = os.path.join("../data",params.data_path)
    params.kge_path = os.path.join("align_kg_w_lm_new_neg/src/KGE/models", params.kge_path, "best_valid.model")



    logging.basicConfig(level=logging.INFO)
    exp_dir = f"../experiments/{params.exp_name}"
    params.exp_dir = exp_dir

    if not os.path.exists("../experiments"):
        os.makedirs("../experiments")
    if not os.path.exists(params.exp_dir):
        os.makedirs(params.exp_dir)
    if not os.path.exists(os.path.join(params.exp_dir, 'logs')):
            os.makedirs(os.path.join(params.exp_dir, 'logs'))
    if params.do_test_new and params.do_train:
        file_handler = logging.FileHandler(os.path.join(params.exp_dir, f"logs/logs.txt"), mode='a')
        
    else:
        file_handler = logging.FileHandler(os.path.join(params.exp_dir, f"logs/test_logs.txt"), mode='a')

    logging.root.addHandler(file_handler)
    logging.info('============ Initialized logger ============')
    logging.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items())))




    main(params)

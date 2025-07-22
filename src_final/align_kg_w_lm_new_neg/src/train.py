import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['WANDB_DISABLED'] = 'true'
import logging
import time
import json
import torch
import random
import numpy as np
import argparse
from utils import * 
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from dataset import TrainDataset, TestDataset
from managers import Trainer
from KGE.model_src import Complex
from models.models import Ailgner_self_mask_modality
from datasets import Dataset as HFDataset

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

    print("data_load")
    start = time.time()
    if params.do_train:
        original_df = pd.read_parquet(os.path.join(f"../data/{params.data_path}",f"{params.data_name}.parquet"))
        original_df = original_df[['query_type', 'query', 'q_entity', 'positive_path', 'negative_path']]
        original_df = original_df.copy()

        train_df, test_df = train_test_split(original_df, test_size=params.valid_ratio, random_state=params.seed)
        
        print(f"data_load finished {time.time() - start}")
        with open(f"../data/{params.data_path}/rel2id.pkl", 'rb') as f:
            rel2id = pickle.load(f)
        with open(f"../data/{params.data_path}/ent2id.pkl", 'rb') as f:
            ent2id = pickle.load(f)
        params.num_ents = len(ent2id)
        params.num_rels = len(rel2id)
        rel_set = set()

        for key in rel2id.keys():
            rel_set.add(f"{key}")

        llm_model, tokenizer, llm_config = get_llm_model(params.llm_model_name)

        params.llm_hidden_size = llm_config.dim
        kge_model = Complex(params = params)
        kge_model.load_from_ckpt_path(params.kge_path)
        print(sum(p.numel() for p in kge_model.parameters()))
        print(sum(p.numel() for p in llm_model.parameters()))
        start = time.time()
        params.max_length_input = 70
        logging.info(f"max_length_input : {params.max_length_input}")
        if torch.cuda.is_available():
            params.device = torch.device('cuda')
        start_time = time.time()
        train_query = HFDataset.from_dict({'nl_query': train_df['query'].tolist()})
        train_query_tokens = train_query.map(lambda examples: tokenizer(examples['nl_query'], max_length=params.max_length_input, truncation=True, padding='max_length'), batched=True, num_proc = 16)
        
        train_query_tokens = {'input_ids': train_query_tokens['input_ids'], 'attention_mask': train_query_tokens['attention_mask']}

        valid_query = HFDataset.from_dict({'nl_query': test_df['query'].tolist()})
        valid_query_tokens = valid_query.map(lambda examples: tokenizer(examples['nl_query'], max_length=params.max_length_input, truncation=True, padding='max_length'), batched=True, num_proc = 16)
        
        valid_query_tokens = {'input_ids': valid_query_tokens['input_ids'], 'attention_mask': valid_query_tokens['attention_mask']}



        train_dataset = TrainDataset(data = train_df, tokenizer = tokenizer, ent2id = ent2id, rel2id = rel2id, params = params, query = train_query_tokens)

        start_time = time.time()
        valid_dataset = TestDataset(data = test_df, tokenizer = tokenizer, ent2id = ent2id, rel2id = rel2id, params = params, query = valid_query_tokens)


        model = Ailgner_self_mask_modality(params = params, llm_model = llm_model, kge_model = kge_model)


        total_params = sum(p.numel() for p in model.parameters())
        learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logging.info(f'Total parameters: {total_params}')
        logging.info(f'Learnable parameters: {learnable_params}')
        model = model.to(params.device)
        params_dict = vars(params)
        for key, value in params_dict.items():
            if isinstance(value, torch.device):
                params_dict[key] = str(value)
        with open(os.path.join(params.output_directory, "params.json"), 'w') as f:
            json.dump(dict(vars(params)), f, indent=4)
        trainer = Trainer(model = model, params = params, train = train_dataset, valid = valid_dataset)
        trainer.train()
        logging.info("Finished")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TransE model')
    parser.add_argument("--num_epochs", "-ne", type=int, default=100,
                        help="Learning rate of the optimizer")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate of the optimizer")
    parser.add_argument("--l2", type=float, default=1e-3,
                        help="Regularization constant for GNN weights")
    parser.add_argument('--seed', default=0,
                        type=int, help='Seed for randomization')

    parser.add_argument('--train_batch_size', default=1024,
                        type=int, help='train_batch_size')
    parser.add_argument('--eval_batch_size',default=256,
                        type=int, help='eval_batch_size')
    parser.add_argument('--data_path', default="UMLS",
                        type=str, help='data_path')

    parser.add_argument('--exp_name', default="ex",
                        type=str, help='model_name')

    parser.add_argument('--kge_hidden_size', default=2000,
                        type=int, help='train_batch_size')
    parser.add_argument('--negative_sample_size', default=512,
                        type=int, help='train_batch_size')
    parser.add_argument('--ent_negative_sample_size', default=256,
                        type=int, help='train_batch_size')
    parser.add_argument('--rel_negative_sample_size', default=64,
                        type=int, help='train_batch_size')
    parser.add_argument('--data_name', default="query_extraction_data_all_fin_drop_dup",
                        type=str, help='data_path')
    parser.add_argument('--kge_path', default='UMLS-complex', 
                        type=str, help='model_name')

    parser.add_argument('--valid_ratio', default=0.05,
                        type=float, help='train_batch_size')

    parser.add_argument('--smoothing', default=0.1,
                        type=float, help='train_batch_size')
    parser.add_argument('--llm_model_name', default='st', 
                        type=str, help='model_name')
    parser.add_argument('--do_train', action='store_true', help="do test")
    parser.add_argument('--do_test', action='store_true', help="do test")
    parser.add_argument('--num_layers',default=2,
                        type=int, help='eval_batch_size')
    parser.add_argument('--train_log_step',default=100,
                        type=int, help='eval_batch_size')      
    parser.add_argument('--valid_per_epoch',default=10,
                        type=int, help='eval_batch_size')  
    parser.add_argument('--save_per_epoch',default=100,
                        type=int, help='eval_batch_size')                      
    parser.add_argument('--num_heads',default=16,
                        type=int, help='eval_batch_size')
    parser.add_argument('--dropout',default=0.1,
                        type=float, help='eval_batch_size')
    parser.add_argument('--eps', default=1e-12, type=float)


    parser.add_argument('--aligner_epoch',default=0,
                        type=int, help='eval_batch_size')
    parser.add_argument('--use_type_neg',type=str2bool, default=True,
                   help='constrain token')

    parser.add_argument('--probs', default=0.5, type=float)


    parser.add_argument('--use_scheduler', choices=['cosine', 'not_use'], default = 'cosine')
    parser.add_argument('--warm_up_ratio', default=0.01, type=float)

    parser.add_argument('--use_q_visible', action='store_true', default = True)
    parser.add_argument('--use_all_visible', action='store_true')

    params = parser.parse_args()
    params.kge_path = os.path.join("KGE/models", params.kge_path, "best_valid.model")

    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)

    output_directory = os.path.join(f"../save_path/{params.data_path}", params.exp_name)
    params.output_directory = output_directory
    if not os.path.exists("../save_path"):
        os.mkdir("../save_path")
    if not os.path.exists(f"../save_path/{params.data_path}"):
        os.mkdir(f"../save_path/{params.data_path}")        
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    



    logging.basicConfig(level=logging.INFO)
    if not os.path.exists(os.path.join(output_directory, 'logs')):
            os.makedirs(os.path.join(output_directory, 'logs'))
    if not params.do_train and params.do_test:
        file_handler = logging.FileHandler(os.path.join(output_directory, f"logs/origin_test_logs.txt"), mode='a')
    else:
        file_handler = logging.FileHandler(os.path.join(output_directory, f"logs/logs.txt"), mode='w')

    logging.root.addHandler(file_handler)
    logging.info('============ Initialized logger ============')
    logging.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items())))


    main(params)

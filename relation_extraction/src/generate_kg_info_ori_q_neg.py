import os

from transformers import T5ForConditionalGeneration, T5Tokenizer

from datasets import Dataset
from peft import PeftModel, PeftConfig


import logging
import json
import torch
import random
import numpy as np
import argparse
from functools import partial
from tqdm import tqdm
import re
from concurrent.futures import ProcessPoolExecutor
from utils import RestrictVocabLogitsProcessor, batchify
import json
import pickle
from functools import partial
import pandas as pd
import copy
import dill



def main(params):

    
    with open(os.path.join(params.exp_dir, 'params.json'), 'r') as f:
        train_params = json.load(f)
    device = torch.device('cuda')

    finetuned_model = T5ForConditionalGeneration.from_pretrained(params.last_checkpoint).to(device)
    tokenizer = T5Tokenizer.from_pretrained(params.last_checkpoint)


    if train_params['constrain_token']:
        with open(os.path.join(params.exp_dir,'allowed_token.pkl'), 'rb') as pickle_file:
        
            allowed_token = pickle.load(pickle_file)

        allowed_token_ids = tokenizer.convert_tokens_to_ids(allowed_token)
        restrict_vocab_processor = RestrictVocabLogitsProcessor(allowed_token_ids)


    if params.data_path in ['FB15k-237', 'UMLS', 'CODEX']:
        law_data_path = os.path.join(f"../data/{params.data_path}", f"{params.data_type}-subq_drop_dup.parquet")

    else:
        raise NotImplementedError
    law_data = pd.read_parquet(law_data_path)

    ori_q = law_data['query'].tolist()
    ori_q_dict = {'query': ori_q}
    query_dataset = Dataset.from_dict(ori_q_dict)

    def preprocess_function_ex(examples):
        input_text = [f"Please extract the relations that exist in the knowledge graph from the following question: {query}" for query in examples['query']]
        model_inputs = tokenizer(input_text, padding='longest')
        return model_inputs

    ex_dataset = copy.deepcopy(query_dataset)

    serialized_preprocess_function_ex = dill.dumps(preprocess_function_ex)
    ex_tokens = ex_dataset.map(lambda x: dill.loads(serialized_preprocess_function_ex)(x), batched=True, num_proc=16)

    params.max_length_input = 0
    for val in ex_tokens['input_ids']:
        if params.max_length_input < len(val):
            params.max_length_input = len(val)
    params.max_length_input = min(params.max_length_input, 100)
    logging.info(f"Max length input: {params.max_length_input}")
    
    def preprocess_function(examples):
        input_text = [f"Please extract the relations that exist in the knowledge graph from the following question: {query}" for query in examples['query']]
        model_inputs = tokenizer(input_text, truncation = True, return_tensors='pt', padding='max_length', max_length=params.max_length_input , pad_to_max_length = True)
        return model_inputs

    serialized_preprocess_function = dill.dumps(preprocess_function)
    tokenized_dataset = query_dataset.map(lambda x: dill.loads(serialized_preprocess_function)(x), batched=True, num_proc=16)
    logging.info("tokenizing finished")

    num_samples = len(tokenized_dataset['input_ids'])
    logging.info(f"Num samples: {num_samples}")
    total_batches = num_samples // params.batch_size + (1 if num_samples % params.batch_size != 0 else 0)
    output_queries = []

    logging.info("generating started")
    tokenized_dataset_selected = {'input_ids': torch.tensor(tokenized_dataset['input_ids'], dtype=torch.long), 'attention_mask': torch.tensor(tokenized_dataset['attention_mask'], dtype=torch.long)}
    for batch in tqdm(batchify(tokenized_dataset_selected, params.batch_size), total=total_batches, desc="Generating outputs"):
        with torch.no_grad():
            outputs = finetuned_model.generate(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                num_beams=params.num_beams,
                num_return_sequences=params.num_sequences,
                logits_processor=[restrict_vocab_processor] if train_params['constrain_token'] else None, 
                max_new_tokens = train_params['max_length_label']
            )
            output_queries.extend(outputs.cpu())

    logging.info("Generating finished")
    batches = [output_queries[i:i + params.batch_size] for i in range(0, len(output_queries), params.batch_size)]
    decode_batch = partial(tokenizer.batch_decode, skip_special_tokens=False)
    logging.info("decoding start")
    with ProcessPoolExecutor() as executor:

        results = list(tqdm(executor.map(decode_batch, batches), total=len(batches)))

    
    decoded_output = [item for sublist in results for item in sublist]
    

    tokens_to_remove = ['<pad>', '<PATH>', '<SEP>', '</PATH>', '</s>', '<unk>']

    
    pattern = re.compile(r'\s*(' + '|'.join(re.escape(token) for token in tokens_to_remove) + r')\s*')

    logging.info("remove spetial token start")
    decoded_output = [pattern.sub(' ', sentence).strip() for sentence in decoded_output]
    decoded_output = [sentence.split(' ') for sentence in decoded_output]
    pos_rel_lst = []
    neg_rel_lst = []
    stop_indicator = False
    for val in decoded_output:
        tmp_pos = []
        tmp_neg = []
        for idx, v in enumerate(val):
            if stop_indicator:
                stop_indicator = False
                continue
            if "<NEG>" in v:
                if idx + 1 == len(val):
                    continue
                else:
                    if "<NEG>" in val[idx+1]:
                        continue
                    else:
                        tmp_neg.append(val[idx + 1][1:-1])
                stop_indicator = True
            else:
                tmp_pos.append(v[1:-1])
        pos_rel_lst.append(tmp_pos)
        neg_rel_lst.append(tmp_neg)
    
    final_pd = law_data.copy()
    final_pd['ori_rel_pos'] = pos_rel_lst
    final_pd['ori_rel_neg'] = neg_rel_lst
    logging.info("generated data save")
    if not os.path.exists(os.path.join(f"../data/{params.data_path}", f'path_from_{params.exp_name}_{params.checkpoint}')):
            os.makedirs(os.path.join(f"../data/{params.data_path}", f'path_from_{params.exp_name}_{params.checkpoint}'))
    if not params.use_query_extraction:
        final_pd.to_parquet(os.path.join(f"../data/{params.data_path}", 
                                         f'path_from_{params.exp_name}_{params.checkpoint}', 
                                         f"{params.data_path}-{params.data_type}-subq_sampled_max5_w_path.parquet"), index=False)
    else:
        final_pd.to_parquet(os.path.join(f"../data/{params.data_path}", f'path_from_{params.exp_name}_{params.checkpoint}', f"{params.data_name}_w_path.parquet"), index=False)

    logging.info("finished")


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='TransE model')
    parser.add_argument("--gpu", type=int, default=0,
                        help="Learning rate of the optimizer")
    parser.add_argument('--seed', dest='seed', default=0,
                        type=int, help='Seed for randomization')
    parser.add_argument('--checkpoint', default="36681",
                        type=int, help='checkpoint')
    parser.add_argument('--exp_name', default="t5_xl_query_fin_neg_1e-4",
                        type=str, help='model_name')
    parser.add_argument('--data_type', default="train",
                        type=str, help='model_name')
    parser.add_argument('--batch_size',  default=1024,
                        type=int, help='batch_size')
    parser.add_argument('--num_beams',  default=1,
                        type=int, help='batch_size')
    parser.add_argument('--num_sequences',  default=1,
                        type=int, help='batch_size')
    parser.add_argument('--data_path', default="FB15k-237",
                        type=str, help='model_name')
    parser.add_argument('--data_name', default="query_extraction_data_all_fin",
                        type=str, help='model_name')
    parser.add_argument('--use_query_extraction', action='store_true')

    params = parser.parse_args()

    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)

    logging.basicConfig(level=logging.INFO)
    exp_dir = f"../save_path/{params.data_path}/{params.exp_name}"
    params.exp_dir = exp_dir
    last_checkpoint = os.path.join(exp_dir, f'checkpoint-{params.checkpoint}')
    params.last_checkpoint = last_checkpoint
    if not os.path.exists(os.path.join(last_checkpoint, 'generate_logs')):
            os.makedirs(os.path.join(last_checkpoint, 'generate_logs'))

    file_handler = logging.FileHandler(os.path.join(last_checkpoint, f"generate_logs/log_test_seq_num_{params.num_sequences}.txt"), mode='w')

    logging.root.addHandler(file_handler)
    logging.info('============ Initialized logger ============')
    logging.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items())))
    
    main(params)
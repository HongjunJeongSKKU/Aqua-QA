import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig
import argparse
from datasets import Dataset
import json
import numpy as np
import random
import logging
import torch
import pandas as pd
import ast
from torch.optim.lr_scheduler import _LRScheduler
import math


TYPE_TO_IDX = {'1p': 0,
    '2p': 1,
    '3p': 2,
    '2i': 3,
    '3i': 4,
    'pi': 5,
    'ip': 6,
    '2u': 7,
    'up': 8,
    '2in': 9,
    '3in': 10,
    'inp': 11,
    'pin': 12,
    'pni': 13
}

def restore_nested_list(input_ids_list, attention_mask_list, nested_list):
    idx = 0
    restored_list = []
    for sublist in nested_list:
        restored_sublist = {
            'input_ids': [],
            'attention_mask': []
        }
        for _ in sublist:
            restored_sublist['input_ids'].append(input_ids_list[idx])
            restored_sublist['attention_mask'].append(attention_mask_list[idx])
            idx += 1
        restored_sublist['input_ids'] = torch.stack(restored_sublist['input_ids'])
        restored_sublist['attention_mask'] = torch.stack(restored_sublist['attention_mask'])
        restored_list.append(restored_sublist)
    return restored_list

def is_all_empty(nested_list):
    if not nested_list:
        return True
    return all(isinstance(sublist, list) and is_all_empty(sublist) for sublist in nested_list)

def flatten_nested_list(nested_list):
    result = []
    for element in nested_list:
        if isinstance(element, list):
            result.extend(flatten_nested_list(element))
        else:
            result.append(element)
    return result
def linearize_each_element(nested_list):
    linearized_list = []
    for element in nested_list:
        flattened_element = flatten_nested_list(element)
        linearized_list.append(flattened_element)
    return linearized_list

def get_llm_model(llm_name):
    if llm_name == "llama3-8b":
        model_name = "meta-llama/Meta-Llama-3-8B"
        ModelClass =AutoModelForCausalLM
        TokenizerClass = AutoTokenizer
    elif llm_name == "llama3-8b_inst":
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        ModelClass = AutoModelForCausalLM
        TokenizerClass = AutoTokenizer
    elif llm_name == "llama2_7b":
        model_name = "meta-llama/Llama-2-7b-hf"
        ModelClass = AutoModelForCausalLM
        TokenizerClass = AutoTokenizer
    elif llm_name == "llama2_13b":
        model_name = "meta-llama/Llama-2-13b-hf"
        ModelClass = AutoModelForCausalLM
        TokenizerClass = AutoTokenizer

    elif llm_name == "e5":
        model_name = "intfloat/e5-large-v2"
        ModelClass = AutoModel
        TokenizerClass = AutoTokenizer

    elif llm_name.lower() == "bert":
        model_name = "bert-base-uncased"
        ModelClass = AutoModel
        TokenizerClass = AutoTokenizer

    elif llm_name.lower() == "robert":
        model_name = "roberta-base"
        ModelClass = AutoModel
        TokenizerClass = AutoTokenizer
    elif llm_name.lower() == 'st':
        model_name = "sentence-transformers/multi-qa-distilbert-cos-v1"
        ModelClass = AutoModel
        TokenizerClass = AutoTokenizer
    elif llm_name.lower() == 'l12':
        model_name = "sentence-transformers/all-MiniLM-L12-v2"
        ModelClass = AutoModel
        TokenizerClass = AutoTokenizer
    else:
        raise ValueError(f"Unknown language model: {llm_name}.")

    if llm_name[:5] == "llama":
        model = ModelClass.from_pretrained(model_name, token = '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')
        tokenizer = TokenizerClass.from_pretrained(model_name, token = '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')
        tokenizer.pad_token = tokenizer.eos_token
        config = AutoConfig.from_pretrained(model_name, token = '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')
    else:
        model = ModelClass.from_pretrained(model_name)
        tokenizer = TokenizerClass.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)

    return model, tokenizer, config

def batch_to_gpu(batch, device):

    all_query, all_query_padding_mask, paths, paths_padding_mask, neg_indicator, pos, negs = batch
    all_query = {k: v.to(device) for k, v in all_query.items()}
    all_query_padding_mask = all_query_padding_mask.to(device)
    paths = paths.to(device)
    paths_padding_mask = paths_padding_mask.to(device)
    pos = pos.to(device)
    negs = negs.to(device)
    neg_indicator = neg_indicator.to(device)
    return all_query, all_query_padding_mask, paths, paths_padding_mask, neg_indicator, pos, negs

def batch_to_gpu_test(batch, device):
    #all_query, all_query_padding_mask, paths, paths_padding_mask, easy, hard, query_type, type_emb, pos_emb, adj = batch
    all_query, all_query_padding_mask, paths, paths_padding_mask, neg_indicator, easy, hard, query_type = batch
    all_query = {k: v.to(device) for k, v in all_query.items()}
    all_query_padding_mask = all_query_padding_mask.to(device)
    paths = paths.to(device)
    paths_padding_mask =  paths_padding_mask.to(device)
    neg_indicator = neg_indicator.to(device)
    return all_query, all_query_padding_mask, paths, paths_padding_mask, neg_indicator, easy, hard, query_type







def dict_to_namespace(d):
    namespace = argparse.Namespace()
    for key, value in d.items():
        setattr(namespace, key, value)
    return namespace

def get_preprocessed_dataset(f_name, clip_path = False, max_hop = 4, all_rel = None, all_ent = None):
    query_path_set = []
    with open(f_name, "r") as f:
        for line in f:
            # 각 라인을 JSON 객체로 파싱하여 리스트에 추가
            tmp = json.loads(line.strip())
            query_path_set.append(tmp)

    query_path_set_processed = {'nl_query': [], 'nl_paths': [], 'nl_sub_query': [], 'labels': []}

    for idx, val in enumerate(query_path_set):
        if idx == 100:
            break
        q = val[0]['question']
        randn = random.randint(1, 4)
        sub_q = [q for _ in range(0,randn) ]
        if not sub_q:
          raise Exception("sub_q error")
        paths = []
        for i in range(0, randn):
          sq_path = []
          for j in range(0, 3):
            sq_path.append(random.sample(all_rel, random.randint(0, max_hop)))
          paths.append(sq_path)

        query_path_set_processed['nl_query'].append(q)
        query_path_set_processed['nl_sub_query'].append(sub_q)
        query_path_set_processed['nl_paths'].append(paths)

        ##여기 지워야됨
        max_samples = 20
        num_samples = np.random.geometric(p=0.3) # Geometric distribution as an example
        num_samples = max(1, min(num_samples, max_samples))
        samples = np.random.choice(range(len(all_ent)), size=num_samples, replace=False)

        query_path_set_processed['labels'].append(list(np.array(all_ent)[samples]))
        ####
    return Dataset.from_dict(query_path_set_processed)



def log_metrics(mode, step = None, metrics = None):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        if not step is None:
            logging.info('%s %s at epoch %d: %f' % (mode, metric, step, metrics[metric]))
        else:
            logging.info('%s %s: %f' % (mode, metric, metrics[metric]))

def save_model(model, params, epoch, type = None):
    state_dict = model.state_dict()

    for key in list(state_dict.keys()):  
        if key.startswith('llm_model') or key.startswith('kge_model') or key.startswith('aligner'):
            del state_dict[key]
    if type == 'best':
        if not os.path.exists(os.path.join(params.exp_dir, 'best_model')):
            os.makedirs(os.path.join(params.exp_dir, 'best_model'))
        torch.save(state_dict, os.path.join(params.exp_dir, 'best_model', f'best_model.pth'))
    else:
        torch.save(state_dict, os.path.join(params.exp_dir, f'model_epoch_{epoch}.pth'))


def get_nl_dataset_labels(data_path, exp_name, checkpoint, mode, prob = 0.5, ent2id = None):

    if mode == 'train_sampled' :
        target_df = pd.read_parquet(os.path.join(data_path, f"path_from_{exp_name}_{checkpoint}", 
        f'FB15k-237-{mode}_{prob}-subq_sampled_max5_w_path_drop_dup.parquet'))
    else:
        target_df = pd.read_parquet(os.path.join(data_path, f"path_from_{exp_name}_{checkpoint}", 
        f'FB15k-237-{mode}-subq_sampled_max5_w_path_drop_dup.parquet'))
        
    if mode[:5] == 'train':
        target_df['ans'] = [[ent2id[v] for v in val] for val in target_df['ans'].tolist()]

        query_path_set_processed = {'labels_id': target_df['ans'].tolist()}
    elif mode in ['valid', 'test']:
        target_df['easy_ans'] = [[ent2id[v] for v in val] for val in target_df['easy_ans'].tolist()]
        target_df['hard_ans'] = [[ent2id[v] for v in val] for val in target_df['hard_ans'].tolist()]

        query_path_set_processed = {'easy_answer_id': target_df['easy_ans'].tolist(),
                                    'hard_answer_id': target_df['hard_ans'].tolist()}
    return query_path_set_processed





def get_nl_dataset(data_path, exp_name, checkpoint, mode, params = None, prob = 0.25, rel2id = None, ent2id = None, tokenizer = None):

    if mode == 'train_sampled' :
        target_df = pd.read_parquet(os.path.join(data_path, f"path_from_{params.path_exp_name}_{params.path_checkpoint}", 
        f'FB15k-237-{mode}_{prob}-subq_sampled_max5_w_path.parquet' if not params.drop_dup else  f'FB15k-237-{mode}_{prob}-subq_sampled_max5_w_path_drop_dup.parquet'))
    else:
        target_df = pd.read_parquet(os.path.join(data_path, f"path_from_{params.path_exp_name}_{params.path_checkpoint}", 
        f'FB15k-237-{mode}-subq_sampled_max5_w_path.parquet' if not params.drop_dup else f'FB15k-237-{mode}-subq_sampled_max5_w_path_drop_dup.parquet'))
    #target_df = target_df.sample(frac=0.01)
    target_df['ori_rel_pos'] = [[rel2id[v] for v in val] for val in target_df['ori_rel_pos'].tolist()]
    target_df['ori_rel_neg'] = [[rel2id[v] for v in val] for val in target_df['ori_rel_neg'].tolist()]

    sub_query_padding_mask = []
    sub_q_lst = target_df['sampled_sub'].tolist()
    for val in sub_q_lst:
        tmp = [1 for _ in range(0, len(val))]
        if len(tmp) < params.num_max_sq:
            tmp.extend([0 for _ in range(0, params.num_max_sq - len(val))])
            sub_query_padding_mask.append(tmp)
        elif len(tmp) == params.num_max_sq:
            sub_query_padding_mask.append(tmp)
        else:
            raise Exception("sub query num error")

    sub_q_lst_w_padding = []
    for sub_q in sub_q_lst:
        sub_q = list(sub_q)

        while len(sub_q) < params.num_max_sq:
            sub_q.append("")
        sub_q_lst_w_padding.append(sub_q)
    sub_q_lst_w_padding = np.array(sub_q_lst_w_padding)
    paths_padding_mask = []

    ori_rel_list_w_padding_all = []
    paths_padding_mask_all = []

    num_max_rel_all = {'pos': max([len(ori_rel_pos) for ori_rel_pos in target_df['ori_rel_pos'].tolist()]),
                       'neg': max([len(ori_rel_neg) for ori_rel_neg in target_df['ori_rel_neg'].tolist()])}
    num_max_rel = max(num_max_rel_all['pos'], num_max_rel_all['neg'])

    for ori_rel_pos, ori_rel_neg in zip(target_df['ori_rel_pos'].tolist(), target_df['ori_rel_neg'].tolist()):

        tmp_padding = {'pos': [], 'neg': []}
        tmp_ori_rel = {'pos': [], 'neg': []}
        ori_rel_all = {'pos': ori_rel_pos, 'neg': ori_rel_neg}

        for rel_type, target_ori_rel in ori_rel_all.items():
            tmp = [1 for _ in range(0, len(target_ori_rel))]

            if len(tmp) < num_max_rel_all[rel_type]:
                tmp.extend([0 for _ in range(0, num_max_rel_all[rel_type] - len(tmp))])
                target_ori_rel.extend([-1 for _ in range(0, num_max_rel_all[rel_type] - len(target_ori_rel))])

            elif len(tmp) == num_max_rel_all[rel_type]:
                pass
                
            else:
                print(target_ori_rel)
                raise Exception("path num error")
            tmp_padding[rel_type] = tmp
            tmp_ori_rel[rel_type] = target_ori_rel
        
        paths_padding_mask_all.append(tmp_padding['pos'] + tmp_padding['neg'])
        ori_rel_list_w_padding_all.append(tmp_ori_rel['pos'] + tmp_ori_rel['neg']) 











    nl_query = Dataset.from_dict({"nl_query": target_df['origin'].tolist()})
    nl_query_tokens = nl_query.map(lambda examples: tokenizer(examples['nl_query'], max_length=params.max_tokens, truncation=True, padding='max_length'), batched=True, num_proc = params.num_proc)
    nl_sub_query = Dataset.from_dict({"sub_query": [v for val in sub_q_lst_w_padding for v in val]})
    nl_sub_query_tokens = nl_sub_query.map(lambda examples: tokenizer(examples['sub_query'], max_length=params.max_tokens, truncation=True, padding='max_length'), batched=True, num_proc = params.num_proc)

    query_input_ids = torch.tensor(np.array(nl_query_tokens['input_ids']), dtype= torch.long).view(len(target_df), params.max_tokens)
    query_attention_mask = torch.tensor(np.array(nl_query_tokens['attention_mask']), dtype =torch.long).view(len(target_df), params.max_tokens)
    sub_query_input_ids = torch.tensor(np.array(nl_sub_query_tokens['input_ids']), dtype =torch.long).view(len(target_df), params.num_max_sq, params.max_tokens)
    sub_query_attention_mask = torch.tensor(np.array(nl_sub_query_tokens['attention_mask']), dtype = torch.long).view(len(target_df), params.num_max_sq, params.max_tokens)
    sub_query_padding_mask = torch.tensor(sub_query_padding_mask, dtype=torch.bool).view(len(target_df), params.num_max_sq)
    nl_paths = torch.tensor(ori_rel_list_w_padding_all, dtype=torch.long)
    paths_padding_mask = torch.tensor(paths_padding_mask_all, dtype = torch.bool)
    neg_indicator = torch.cat([torch.zeros(nl_paths.size(0), num_max_rel_all['pos']), torch.ones(nl_paths.size(0), num_max_rel_all['neg'])], dim = -1).bool()

    if mode[:5] == 'train':
        target_df['ans'] = [[ent2id[v] for v in val] for val in target_df['ans'].tolist()]

        query_path_set_processed = {'query': {'input_ids': query_input_ids, 'attention_mask': query_attention_mask},
                                    'sub_query': {'input_ids': sub_query_input_ids, 'attention_mask': sub_query_attention_mask},
                                    'paths': nl_paths, 'path_padding_mask': paths_padding_mask,  'neg_indicator': neg_indicator,
                                    'sub_query_padding_mask': sub_query_padding_mask, 'labels_id': target_df['ans'].tolist()}
    elif mode in ['valid', 'test', 'valid_hard', 'test_hard']:
        target_df['easy_ans'] = [[ent2id[v] for v in val] for val in target_df['easy_ans'].tolist()]
        target_df['hard_ans'] = [[ent2id[v] for v in val] for val in target_df['hard_ans'].tolist()]

        query_path_set_processed = {'query': {'input_ids': query_input_ids, 'attention_mask': query_attention_mask},
                                    'sub_query': {'input_ids': sub_query_input_ids, 'attention_mask': sub_query_attention_mask},
                                    'paths': nl_paths, 'path_padding_mask': paths_padding_mask, 'neg_indicator': neg_indicator,
                                    'sub_query_padding_mask': sub_query_padding_mask, 'easy_answer_id': target_df['easy_ans'].tolist(),
                                    'hard_answer_id': target_df['hard_ans'].tolist(), 'query_type': target_df['query_type'].tolist()}
    else:
        raise Exception("data error")
    return query_path_set_processed #Dataset.from_dict(query_path_set_processed)

def truncate_list(lst, threshold):
    if len(lst) > threshold:
        return list(np.random.choice(lst, threshold, replace=False))
    return lst


def get_nl_dataset_umls_codex(data_path, exp_name, checkpoint, mode, params = None, prob = 0.25, rel2id = None, ent2id = None, tokenizer = None):
    #if mode == "train":
    target_df = pd.read_parquet(os.path.join(data_path, f'path_from_{exp_name}_{checkpoint}',f"{data_path.split('/')[-1]}-{mode}-subq_sampled_max{params.num_max_sq}_w_path.parquet"))

    target_df['ori_rel_pos'] = [[rel2id[v] for v in val if v != ''] for val in target_df['ori_rel_pos'].tolist()]
    target_df['ori_rel_neg'] = [[rel2id[v] for v in val if v != ''] for val in target_df['ori_rel_neg'].tolist()]
    target_df['sampled_sub'] = target_df['sub'].apply(lambda x: truncate_list(x, params.num_max_sq))
    sub_query_padding_mask = []
    sub_q_lst = target_df['sampled_sub'].tolist()
    for val in sub_q_lst:
        tmp = [1 for _ in range(0, len(val))]
        if len(tmp) < params.num_max_sq:
            tmp.extend([0 for _ in range(0, params.num_max_sq - len(val))])
            sub_query_padding_mask.append(tmp)
        elif len(tmp) == params.num_max_sq:
            sub_query_padding_mask.append(tmp)
        else:
            raise Exception("sub query num error")

    sub_q_lst_w_padding = []
    for sub_q in sub_q_lst:
        sub_q = list(sub_q)

        while len(sub_q) < params.num_max_sq:
            sub_q.append("")
        sub_q_lst_w_padding.append(sub_q)
    sub_q_lst_w_padding = np.array(sub_q_lst_w_padding)
    paths_padding_mask = []

    ori_rel_list_w_padding_all = []
    paths_padding_mask_all = []

    num_max_rel_all = {'pos': max([len(ori_rel_pos) for ori_rel_pos in target_df['ori_rel_pos'].tolist()]),
                       'neg': max([len(ori_rel_neg) for ori_rel_neg in target_df['ori_rel_neg'].tolist()])}
    num_max_rel = max(num_max_rel_all['pos'], num_max_rel_all['neg'])

    for ori_rel_pos, ori_rel_neg in zip(target_df['ori_rel_pos'].tolist(), target_df['ori_rel_neg'].tolist()):

        tmp_padding = {'pos': [], 'neg': []}
        tmp_ori_rel = {'pos': [], 'neg': []}
        ori_rel_all = {'pos': ori_rel_pos, 'neg': ori_rel_neg}

        for rel_type, target_ori_rel in ori_rel_all.items():
            tmp = [1 for _ in range(0, len(target_ori_rel))]

            if len(tmp) < num_max_rel_all[rel_type]:
                tmp.extend([0 for _ in range(0, num_max_rel_all[rel_type] - len(tmp))])
                target_ori_rel.extend([-1 for _ in range(0, num_max_rel_all[rel_type] - len(target_ori_rel))])

            elif len(tmp) == num_max_rel_all[rel_type]:
                pass
                
            else:
                print(target_ori_rel)
                raise Exception("path num error")
            tmp_padding[rel_type] = tmp
            tmp_ori_rel[rel_type] = target_ori_rel
        
        paths_padding_mask_all.append(tmp_padding['pos'] + tmp_padding['neg'])
        ori_rel_list_w_padding_all.append(tmp_ori_rel['pos'] + tmp_ori_rel['neg']) 











    nl_query = Dataset.from_dict({"nl_query": target_df['query'].tolist()})
    nl_query_tokens = nl_query.map(lambda examples: tokenizer(examples['nl_query'], max_length=params.max_tokens, truncation=True, padding='max_length'), batched=True, num_proc = params.num_proc)
    nl_sub_query = Dataset.from_dict({"sub_query": [v for val in sub_q_lst_w_padding for v in val]})
    nl_sub_query_tokens = nl_sub_query.map(lambda examples: tokenizer(examples['sub_query'], max_length=params.max_tokens, truncation=True, padding='max_length'), batched=True, num_proc = params.num_proc)

    query_input_ids = torch.tensor(np.array(nl_query_tokens['input_ids']), dtype= torch.long).view(len(target_df), params.max_tokens)
    query_attention_mask = torch.tensor(np.array(nl_query_tokens['attention_mask']), dtype =torch.long).view(len(target_df), params.max_tokens)
    sub_query_input_ids = torch.tensor(np.array(nl_sub_query_tokens['input_ids']), dtype =torch.long).view(len(target_df), params.num_max_sq, params.max_tokens)
    sub_query_attention_mask = torch.tensor(np.array(nl_sub_query_tokens['attention_mask']), dtype = torch.long).view(len(target_df), params.num_max_sq, params.max_tokens)
    sub_query_padding_mask = torch.tensor(sub_query_padding_mask, dtype=torch.bool).view(len(target_df), params.num_max_sq)
    nl_paths = torch.tensor(ori_rel_list_w_padding_all, dtype=torch.long)#.view(len(target_df), params.num_info_rel)#, params.max_tokens)
    paths_padding_mask = torch.tensor(paths_padding_mask_all, dtype = torch.bool)#.view(len(target_df), params.num_info_rel)
    neg_indicator = torch.cat([torch.zeros(nl_paths.size(0), num_max_rel_all['pos']), torch.ones(nl_paths.size(0), num_max_rel_all['neg'])], dim = -1).bool()

    if mode == 'train':
        target_df['ans'] = [[ent2id[v] for v in val] for val in target_df['answer'].tolist()]

        query_path_set_processed = {'query': {'input_ids': query_input_ids, 'attention_mask': query_attention_mask},
                                    'sub_query': {'input_ids': sub_query_input_ids, 'attention_mask': sub_query_attention_mask},
                                    'paths': nl_paths, 'path_padding_mask': paths_padding_mask,  'neg_indicator': neg_indicator,
                                    'sub_query_padding_mask': sub_query_padding_mask, 'labels_id': target_df['ans'].tolist()}
    elif mode in ['valid', 'test', 'valid_hard', 'test_hard']:
        target_df['easy_ans'] = [[ent2id[v] for v in val] for val in target_df['easy_ans'].tolist()]
        target_df['hard_ans'] = [[ent2id[v] for v in val] for val in target_df['hard_ans'].tolist()]

        query_path_set_processed = {'query': {'input_ids': query_input_ids, 'attention_mask': query_attention_mask},
                                    'sub_query': {'input_ids': sub_query_input_ids, 'attention_mask': sub_query_attention_mask},
                                    'paths': nl_paths, 'path_padding_mask': paths_padding_mask, 'neg_indicator': neg_indicator,
                                    'sub_query_padding_mask': sub_query_padding_mask, 'easy_answer_id': target_df['easy_ans'].tolist(),
                                    'hard_answer_id': target_df['hard_ans'].tolist(), 'query_type': target_df['query_type'].tolist()}
    else:
        raise Exception("data error")
    return query_path_set_processed 






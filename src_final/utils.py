import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig
import argparse
from datasets import Dataset
import numpy as np
import logging
import torch
import pandas as pd



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


def log_metrics(mode, step = None, metrics = None):
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

def truncate_list(lst, threshold):
    if len(lst) > threshold:
        return list(np.random.choice(lst, threshold, replace=False))
    return lst


def get_nl_dataset_all(data_path, exp_name, checkpoint, mode, params = None, rel2id = None, ent2id = None, tokenizer = None):
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
    nl_paths = torch.tensor(ori_rel_list_w_padding_all, dtype=torch.long)
    paths_padding_mask = torch.tensor(paths_padding_mask_all, dtype = torch.bool)
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






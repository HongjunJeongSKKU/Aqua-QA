import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig
from datasets import Dataset, DatasetDict
import bitsandbytes as bnb

import json
import torch
import pandas as pd
import pickle
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import logging
import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
import argparse


TYPE_TO_IDX_TEST = {'1p': 0,
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


TYPE_TO_IDX_TRAIN = {'1p': 0,
    '2p': 1,
    '3p': 2,
    '2i': 3,
    '3i': 4,
    '2in': 5,
    '3in': 6,
    'inp': 7,
    'pin': 8,
    'pni': 9,
    '2u': 10
}

def dict_to_namespace(d):
    namespace = argparse.Namespace()
    for key, value in d.items():
        setattr(namespace, key, value)
    return namespace


def get_preprocessed_query_dataset_all_in_one(query_path_set, use_ent = False):

 
    query_path_set_processed = {'query': query_path_set['query'].tolist()}

    return Dataset.from_dict(query_path_set_processed)







def calculate_max_length(dataset, tokenizer):
    texts = [f"{example}" for example in dataset['query'].tolist()]
    encodings = tokenizer(texts, add_special_tokens=True, padding = 'longest')
    max_length = len(encodings['input_ids'][0])

    return max_length









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

    all_query, pos_ent, pos_rel, neg_ent, neg_rel, q_type, and_label, or_label, not_label, rel_neg_indicator = batch
    all_query['input_ids'] = all_query['input_ids'].to(device)
    all_query['attention_mask'] = all_query['attention_mask'].to(device)
    pos_ent = pos_ent.to(device)
    pos_rel = pos_rel.to(device)
    neg_ent = neg_ent.to(device)
    neg_rel = neg_rel.to(device)
    q_type = q_type.to(device)
    and_label = and_label.to(device).float()
    or_label = or_label.to(device).float()
    not_label = not_label.to(device).float()
    rel_neg_indicator = rel_neg_indicator.to(device)#.float()

    return all_query, pos_ent, pos_rel, neg_ent, neg_rel, q_type, and_label, or_label, not_label, rel_neg_indicator

    

def batch_to_gpu_test(batch, device):
    all_query, ents_answer, rels_answer, query_type, and_label, or_label, not_label, rel_neg_indicator = batch
    all_query['input_ids'] = all_query['input_ids'].to(device)
    all_query['attention_mask'] = all_query['attention_mask'].to(device)




    return all_query, ents_answer, rels_answer, query_type, and_label, or_label, not_label, rel_neg_indicator

def collate_fn(batch):

    all_query, pos_ent, pos_rel, neg_ent, neg_rel, q_type, and_label, or_label, not_label, rel_neg_indicator = map(list, zip(*batch)) 

    pos_ent = torch.stack(pos_ent)
    pos_rel = torch.stack(pos_rel)
    
    neg_ent = torch.stack(neg_ent)
    neg_rel = torch.stack(neg_rel)
    q_type = torch.stack(q_type)
    and_label = torch.stack(and_label)
    or_label = torch.stack(or_label)
    not_label = torch.stack(not_label)
    rel_neg_indicator = torch.stack(rel_neg_indicator)





    # 하나의 dict로 합치기
    all_query = {
        'input_ids': torch.stack([item['input_ids'] for item in all_query]),
        'attention_mask': torch.stack([item['attention_mask'] for item in all_query])
    }



    return all_query, pos_ent, pos_rel, neg_ent, neg_rel, q_type, and_label, or_label, not_label, rel_neg_indicator



def collate_fn_test(batch):

    all_query, ents_answer, rels_answer, query_type, and_label, or_label, not_label = map(list, zip(*batch)) 



    and_label = torch.stack(and_label)
    or_label = torch.stack(or_label)
    not_label = torch.stack(not_label)
    multi_label = torch.stack(multi_label)

    ents_answer = torch.stack(ents_answer)
    rels_answer = {
        'pos_rel': torch.stack([item['pos_rel'] for item in rels_answer]),
        'neg_rel': torch.stack([item['neg_rel'] for item in rels_answer])
    }


    # 하나의 dict로 합치기
    all_query = {
        'input_ids': torch.stack([item['input_ids'] for item in all_query]),
        'attention_mask': torch.stack([item['attention_mask'] for item in all_query])
    }



    return all_query, ents_answer, rels_answer, query_type, and_label, or_label, not_label, multi_label

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
        if key.startswith('llm_model') or key.startswith('kge_model'):
            del state_dict[key]
    if type == 'best':
        if not os.path.exists(os.path.join(params.output_directory, 'best_model')):
            os.makedirs(os.path.join(params.output_directory, 'best_model'))
        torch.save(state_dict, os.path.join(params.output_directory, 'best_model', f'best_model.pth'))
    else:
        torch.save(state_dict, os.path.join(params.output_directory, f'model_epoch_{epoch}.pth'))






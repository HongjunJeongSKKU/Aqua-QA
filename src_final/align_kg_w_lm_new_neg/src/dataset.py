import os
from torch.utils.data import Dataset
import torch
import numpy as np
from utils import TYPE_TO_IDX_TRAIN, TYPE_TO_IDX_TEST
from datasets import Dataset as HFDataset
import random
from tqdm import tqdm

class TrainDataset(Dataset):
    def __init__(self, data, tokenizer, ent2id, rel2id, params= None, query = None, num_proc = 16):
        self.params = params
        self.rel2id = rel2id
        self.ent2id = ent2id
        self.type_to_id = TYPE_TO_IDX_TRAIN
        self.id_to_type = {v:k for k, v in self.type_to_id.items()}
        self.nl_query = data['query'].tolist()
        self.q_entity = self.convert_ent_to_id(data['q_entity'].tolist())
        self.pos_rel_lst = self.convert_rel_to_id(data['positive_path'].tolist())
        self.neg_rel_lst = self.convert_rel_to_id(data['negative_path'].tolist())
        self.all_query_type = self.convert_type_to_id(data['query_type'].tolist())
        self.and_label = [1 if 'i' in val else 0 for val in data['query_type'].tolist()]
        self.or_label = [1 if 'u' in val else 0 for val in data['query_type'].tolist()]
        self.not_label = [1 if 'n' in val else 0 for val in data['query_type'].tolist()]
        self.multi_label = [1 if val in ['2p', '3p'] else 0 for val in data['query_type'].tolist()]

        self.num_ents = params.num_ents
        self.num_rels = params.num_rels
        self.ent_negative_sample_size = self.params.ent_negative_sample_size
        self.rel_negative_sample_size = self.params.rel_negative_sample_size
        self.tokenizer = tokenizer
        self.max_tokens = self.params.max_length_input
        self.num_proc = num_proc
        self.dtype = torch.long
        self.queries = {'input_ids': torch.tensor(query['input_ids'], dtype=self.dtype), 'attention_mask': torch.tensor(query['attention_mask'], dtype=torch.bool)} if query is not None else self.tokenize_nl_query(self.nl_query)  

    def __len__(self):
        return len(self.nl_query)

    def __getitem__(self, idx):
        all_query_dict = {'input_ids': self.queries['input_ids'][idx],
                          'attention_mask': self.queries['attention_mask'][idx]}
        pos_ent = torch.tensor(np.random.choice(self.q_entity[idx]), dtype = self.dtype)

        if 'n' in self.id_to_type[self.all_query_type[idx]]:
            if random.random() > 0.5:
                pos_rel = self.pos_rel_lst[idx]
                #pos_rel = torch.tensor(np.random.choice(self.pos_rel_lst[idx]), dtype = self.dtype)
                rel_neg_indicator = 0
            else:
                pos_rel = self.neg_rel_lst[idx]
                #pos_rel = torch.tensor(np.random.choice(self.neg_rel_lst[idx]), dtype = self.dtype)
                rel_neg_indicator = 1
        else:
            pos_rel = self.pos_rel_lst[idx]
            rel_neg_indicator = 0
        neg_ent = list(set(range(self.num_ents)) - set(self.q_entity[idx]))
        neg_rel = list(set(range(self.num_rels)) - set(pos_rel))
        pos_rel = torch.tensor(np.random.choice(pos_rel), dtype=self.dtype)
        if self.ent_negative_sample_size is not None and self.rel_negative_sample_size is not None:
            neg_ent = torch.tensor(random.sample(neg_ent, min(self.ent_negative_sample_size, len(neg_ent))), dtype = self.dtype)
            neg_rel = torch.tensor(random.sample(neg_rel, min(self.rel_negative_sample_size, len(neg_rel))), dtype = self.dtype)
        else:
            neg_ent = torch.tensor(neg_ent, dtype=self.dtype)
            neg_rel = torch.tensor(neg_rel, dtype=self.dtype)
        return all_query_dict, pos_ent, pos_rel, neg_ent, neg_rel, torch.tensor(self.all_query_type[idx], dtype=self.dtype), torch.tensor(self.and_label[idx], dtype = self.dtype), torch.tensor(self.or_label[idx], dtype = self.dtype), torch.tensor(self.not_label[idx], dtype = self.dtype), torch.tensor(rel_neg_indicator, dtype = torch.bool)
    
    def convert_type_to_id(self, q_type_lst):
        type_id = []
        for val in q_type_lst:
            type_id.append(self.type_to_id[val])
        return type_id
   
    def convert_ent_to_id(self, ent_lst):
        ents_id = []
        for val in ent_lst:
            ents_id.append([self.ent2id[ent] for ent in val])
          
        return ents_id

    def convert_rel_to_id(self, rel_lst):
        rels_id = []
        for val in rel_lst:
            rels_id.append([self.rel2id[rel] for rel in val])
        return rels_id

    def tokenize_nl_query(self, nl_origin):
        nl_query = HFDataset.from_dict({'nl_query': nl_origin})
        nl_query_tokens = nl_query.map(lambda examples: self.tokenizer(examples['nl_query'], max_length=self.max_tokens, truncation=True, padding='max_length', return_tensors='pt'), batched=True, num_proc = self.num_proc)
        
        return {'input_ids': torch.tensor(nl_query_tokens['input_ids'], dtype = self.dtype), 'attention_mask': torch.tensor(nl_query_tokens['attention_mask'], dtype=torch.bool)}



class TestDataset(Dataset):
    def __init__(self, data, tokenizer, ent2id, rel2id, params= None, query=None, num_proc = 16):
        self.params = params
        self.rel2id = rel2id
        self.ent2id = ent2id
        self.type_to_id = TYPE_TO_IDX_TRAIN
        self.id_to_type = {v: k for k, v in self.type_to_id.items()}
        self.nl_query = data['query'].tolist()
        self.q_entity = self.convert_ent_to_id(data['q_entity'].tolist())

        self.pos_rel_lst = self.convert_rel_to_id(data['positive_path'].tolist())
        self.neg_rel_lst = self.convert_rel_to_id(data['negative_path'].tolist())

        self.all_query_type = self.convert_type_to_id(data['query_type'].tolist())
        self.and_label = [1 if 'i' in val else 0 for val in data['query_type'].tolist()]
        self.or_label = [1 if 'u' in val else 0 for val in data['query_type'].tolist()]
        self.not_label = [1 if 'n' in val else 0 for val in data['query_type'].tolist()]
        
        self.num_ents = params.num_ents
        self.num_rels = params.num_rels
        self.ent_negative_sample_size = self.params.ent_negative_sample_size
        self.rel_negative_sample_size = self.params.rel_negative_sample_size
        self.tokenizer = tokenizer
        self.max_tokens = self.params.max_length_input
        self.num_proc = num_proc
        self.dtype = torch.long
        self.queries = {'input_ids': torch.tensor(query['input_ids'], dtype=self.dtype), 'attention_mask': torch.tensor(query['attention_mask'], dtype=torch.bool)} if query is not None else self.tokenize_nl_query(self.nl_query)  

    def __len__(self):
        return len(self.nl_query)

    def __getitem__(self, idx):
        all_query_dict = {'input_ids': self.queries['input_ids'][idx],
                          'attention_mask': self.queries['attention_mask'][idx]}
        all_ents_ids = np.arange(self.num_ents)
        ents_answer = torch.from_numpy(np.isin(all_ents_ids, self.q_entity[idx]))
        all_rels_ids = np.arange(self.num_rels)
        rels_answer = {'pos_rel': torch.from_numpy(np.isin(all_rels_ids, self.pos_rel_lst[idx])),
                       'neg_rel': torch.from_numpy(np.isin(all_rels_ids, self.neg_rel_lst[idx]))}
        if 'n' in self.id_to_type[self.all_query_type[idx]]:
            rel_neg_indicator = 1
        else:
            rel_neg_indicator = 0










        return all_query_dict, ents_answer, rels_answer, self.all_query_type[idx], self.and_label[idx], self.or_label[idx], self.not_label[idx], torch.tensor(rel_neg_indicator, dtype = torch.bool)
    
    def convert_type_to_id(self, q_type_lst):
        type_id = []
        for val in q_type_lst:
            type_id.append(self.type_to_id[val])
        return type_id
   
    def convert_ent_to_id(self, ent_lst):
        ents_id = []
        for val in ent_lst:
            ents_id.append([self.ent2id[ent] for ent in val])
          
        return ents_id

    def convert_rel_to_id(self, rel_lst):
        rels_id = []
        for val in rel_lst:
            rels_id.append([self.rel2id[rel] for rel in val])
        return rels_id

    def tokenize_nl_query(self, nl_origin):
        nl_query = HFDataset.from_dict({'nl_query': nl_origin})
        nl_query_tokens = nl_query.map(lambda examples: self.tokenizer(examples['nl_query'], max_length=self.max_tokens, truncation=True, padding='max_length', return_tensors='pt'), batched=True, num_proc = self.num_proc)
        
        return {'input_ids': torch.tensor(nl_query_tokens['input_ids'], dtype = self.dtype), 'attention_mask': torch.tensor(nl_query_tokens['attention_mask'], dtype=torch.bool)}






class TestDataset_original(Dataset):
    def __init__(self, data, tokenizer, ent2id, rel2id, params= None, num_proc = 16):
        self.params = params
        self.rel2id = rel2id
        self.ent2id = ent2id
        self.type_to_id = TYPE_TO_IDX_TEST
        self.nl_query = data['query'].tolist()
        self.q_entity = self.convert_ent_to_id(data['q_entity'].tolist())
        self.paths = self.convert_rel_to_id(data['path_all'].tolist())
        self.all_query_type = self.convert_type_to_id(data['query_type'].tolist())
        self.and_label = [1 if 'i' in val else 0 for val in data['query_type'].tolist()]
        self.or_label = [1 if 'u' in val else 0 for val in data['query_type'].tolist()]
        self.not_label = [1 if 'n' in val else 0 for val in data['query_type'].tolist()]
        self.multi_label = [1 if val in ['2p', '3p']  else 0 for val in data['query_type'].tolist()]
        
        self.num_ents = params.num_ents
        self.num_rels = params.num_rels
        self.ent_negative_sample_size = self.params.ent_negative_sample_size
        self.rel_negative_sample_size = self.params.rel_negative_sample_size
        self.tokenizer = tokenizer
        self.max_tokens = self.params.max_length_input
        self.num_proc = num_proc
        self.dtype = torch.long
        self.queries = self.tokenize_nl_query(self.nl_query)  

    def __len__(self):
        return len(self.nl_query)

    def __getitem__(self, idx):
        all_query_dict = {'input_ids': self.queries['input_ids'][idx],
                          'attention_mask': self.queries['attention_mask'][idx]}
        all_ents_ids = np.arange(self.num_ents)
        ents_answer = torch.from_numpy(np.isin(all_ents_ids, self.q_entity[idx]))
        all_rels_ids = np.arange(self.num_rels)
        rels_answer = torch.from_numpy(np.isin(all_rels_ids, self.paths[idx]))

        return all_query_dict, ents_answer, rels_answer, self.all_query_type[idx], self.and_label[idx], self.or_label[idx], self.not_label[idx], self.multi_label[idx]#torch.tensor(self.all_query_type[idx], dtype=self.dtype), torch.tensor(self.and_label[idx], dtype = self.dtype), torch.tensor(self.or_label[idx], dtype = self.dtype), torch.tensor(self.not_label[idx], dtype = self.dtype)
    
    def convert_type_to_id(self, q_type_lst):
        type_id = []
        for val in q_type_lst:
            type_id.append(self.type_to_id[val])
        return type_id
   
    def convert_ent_to_id(self, ent_lst):
        ents_id = []
        for val in ent_lst:
            ents_id.append([self.ent2id[ent] for ent in val])
          
        return ents_id

    def convert_rel_to_id(self, rel_lst):
        rels_id = []
        for val in rel_lst:
            rels_id.append([self.rel2id[rel] for rel in val])
        return rels_id

    def tokenize_nl_query(self, nl_origin):
        nl_query = HFDataset.from_dict({'nl_query': nl_origin})
        nl_query_tokens = nl_query.map(lambda examples: self.tokenizer(examples['nl_query'], max_length=self.max_tokens, truncation=True, padding='max_length', return_tensors='pt'), batched=True, num_proc = self.num_proc)
        
        return {'input_ids': torch.tensor(nl_query_tokens['input_ids'], dtype = self.dtype), 'attention_mask': torch.tensor(nl_query_tokens['attention_mask'], dtype=torch.bool)}


import os
from torch.utils.data import Dataset
import torch
import numpy as np
from utils import TYPE_TO_IDX

class TrainDataset(Dataset):
    def __init__(self, data, tokenizer, ent2id, rel2id, negative_sample_size = 500, max_tokens = 80, num_proc = 16, params= None):
        self.data = data
        self.labels_id = data['labels_id']
        self.rel2id = rel2id
        self.ent2id = ent2id
        self.num_ents = len(self.ent2id)
        self.num_rels = len(self.rel2id)
        self.negative_sample_size = negative_sample_size
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.dtype = torch.long
        self.num_proc = num_proc
        self.params = params
        self.num_max_sq = self.params.num_max_sq
        self.num_max_path = self.params.num_max_path
        self.num_max_path_len = self.params.num_max_path_len
        self.saved_path_name = os.path.join(self.params.data_path, self.params.llm_model_name, self.params.path_exp_name, str(self.params.path_checkpoint))
        file_name = f'train_sampled_{self.params.prob}'if self.params.do_sampled else 'train' 
        self.saved_file_name = os.path.join(self.params.data_path, self.params.llm_model_name, self.params.path_exp_name, str(self.params.path_checkpoint), f'{file_name}.pth' if not params.drop_dup else f'{file_name}_drop_dup.pth')
        if not os.path.exists(self.saved_path_name):
            os.makedirs(self.saved_path_name)
        else:
            pass

        self.data['all_query'] = {'input_ids': torch.cat([self.data['query']['input_ids'].unsqueeze(1), self.data['sub_query']['input_ids']], dim = 1), 
                                      'attention_mask': torch.cat([self.data['query']['attention_mask'].unsqueeze(1), self.data['sub_query']['attention_mask']], dim = 1)}
        self.data['all_query_padding_mask'] = torch.cat([torch.ones((self.data['sub_query_padding_mask'].size(0),2), dtype=torch.bool),
                                                            torch.ones((self.data['sub_query_padding_mask'].size(0),1), dtype=torch.bool),
                                                            self.data['sub_query_padding_mask']], dim=-1)



    def __len__(self):
        return len(self.data['all_query']['input_ids'])

    def __getitem__(self, idx):
        all_query_dict = {'input_ids': self.data['all_query']['input_ids'][idx],
                          'attention_mask': self.data['all_query']['attention_mask'][idx]}
        positive_sample = np.random.choice(self.data['labels_id'][idx], 1)
        positive_sample = torch.tensor(positive_sample, dtype=torch.long)

        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.num_ents, size=self.negative_sample_size * 2)
            mask = np.in1d(
                negative_sample,
                self.data['labels_id'][idx],
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        negative_samples = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_samples = torch.tensor(negative_samples, dtype=torch.long)

        return all_query_dict, self.data['all_query_padding_mask'][idx], self.data['paths'][idx], self.data['path_padding_mask'][idx], self.data['neg_indicator'][idx], positive_sample, negative_samples

    def save_data(self, processed_data, save_path):
        saved_data = {}


        for k, v in processed_data.items():
            if k in ['all_query', 'all_query_padding_mask', 'paths', 'path_padding_mask', 'neg_indicator', 'labels_id']:
                saved_data[k] = v
            else:
                pass

        torch.save(saved_data, save_path)
    
    def load_data(self, load_path):
        return torch.load(load_path)





class TestDataset(Dataset):
    def __init__(self, data, tokenizer, ent2id, rel2id, max_tokens = 80, num_proc = 16, params = None, mode=None):
        self.data = data
        #self.hard_answer_id = data['hard_answer_id']
        #self.easy_answer_id = data['easy_answer_id']
        self.rel2id = rel2id
        self.ent2id = ent2id
        self.num_ents = len(self.ent2id)
        self.num_rels = len(self.rel2id)
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.dtype = torch.long
        self.num_proc = num_proc
        self.mode = mode
        self.params = params
        self.num_max_sq = self.params.num_max_sq
        self.num_max_path = self.params.num_max_path
        self.num_max_path_len = self.params.num_max_path_len
        self.saved_path_name = os.path.join(self.params.data_path, self.params.llm_model_name, self.params.path_exp_name, str(self.params.path_checkpoint))
        self.saved_file_name = os.path.join(self.params.data_path, self.params.llm_model_name, self.params.path_exp_name, str(self.params.path_checkpoint), f'{mode}.pth' if not params.drop_dup else f'{mode}_drop_dup.pth')
        if not os.path.exists(self.saved_path_name):
            os.makedirs(self.saved_path_name)
        else:
            pass

        self.type_to_id = TYPE_TO_IDX
        self.data['query_type_id'] = self.convert_type_to_id()

        self.data['all_query'] = {'input_ids': torch.cat([self.data['query']['input_ids'].unsqueeze(1), self.data['sub_query']['input_ids']], dim = 1), 
                                'attention_mask': torch.cat([self.data['query']['attention_mask'].unsqueeze(1), self.data['sub_query']['attention_mask']], dim = 1)}
        self.data['all_query_padding_mask'] = torch.cat([torch.ones((self.data['sub_query_padding_mask'].size(0),2), dtype=torch.bool),
                                                        torch.ones((self.data['sub_query_padding_mask'].size(0),1), dtype=torch.bool),
                                                        self.data['sub_query_padding_mask']], dim=1)
            



    def __len__(self):
        return len(self.data['all_query']['input_ids'])

    def __getitem__(self, idx):
        all_query_dict = {'input_ids': self.data['all_query']['input_ids'][idx],
                          'attention_mask': self.data['all_query']['attention_mask'][idx]}
        all_ents_ids = np.arange(self.num_ents)
        easy_answer = torch.from_numpy(np.isin(all_ents_ids, self.data['easy_answer_id'][idx]))
        hard_answer = torch.from_numpy(np.isin(all_ents_ids, self.data['hard_answer_id'][idx]))
        return all_query_dict, self.data['all_query_padding_mask'][idx], self.data['paths'][idx], self.data['path_padding_mask'][idx], self.data['neg_indicator'][idx], easy_answer, hard_answer, self.data['query_type_id'][idx]

    def save_data(self, processed_data, save_path):
        saved_data = {}
        
        for k, v in processed_data.items():
            if k in ['all_query', 'all_query_padding_mask', 'paths', 'path_padding_mask', 'neg_indicator', 'query_type_id', 'easy_answer_id', 'hard_answer_id']:
                saved_data[k] = v
            else:
                pass
        torch.save(saved_data, save_path)
    
    def load_data(self, load_path):
        return torch.load(load_path)


    def convert_type_to_id(self):
        type_id = []
        for val in self.data['query_type']:
            type_id.append(self.type_to_id[val])
        return type_id



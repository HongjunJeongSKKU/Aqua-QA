from transformers import BertConfig
import torch
from torch import nn
from .modeling_bert import BertModel


class BertEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.llm_hidden_size = params.llm_hidden_size
        self.dim_ent_embedding = params.ent_dim
        self.dim_rel_embedding = params.rel_dim
        self.hidden_size = params.reasoner_hidden_size
        self.num_heads = params.num_heads
        self.head_dim = self.hidden_size // params.num_heads
        self.num_hidden_layers = params.num_hidden_layers
        self.dropout = params.dropout
        self.attention_probs_dropout_prob = params.attention_probs_dropout_prob
    
        config = BertConfig(
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
            intermediate_size=self.hidden_size,
            hidden_dropout_prob=self.dropout,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            enc_dist=None,
            fp16 = False,
        )
        self.bert = BertModel(config)

        self.rev_proj1 = nn.Linear(self.hidden_size, self.dim_ent_embedding)
        self.rev_proj2 = nn.Linear(self.hidden_size, self.dim_rel_embedding)


    def forward(self, initial_node_embeddings, pad_ids):

        hidden_states = self.bert(
            inputs_embeds=initial_node_embeddings,
            attention_mask=pad_ids,

        ).last_hidden_state

        # batch, hd
        cls1 = hidden_states[:, 0]
        cls2 = hidden_states[:, 1]

        h = self.rev_proj1(cls1)
        r = self.rev_proj2(cls2)

        return h, r



class MainReasoner(nn.Module):
    def __init__(self, params, aligner, tokenizer):
        super().__init__()
        self.params = params
        self.device = params.device

        self.aligner = self.freeze_parameters(aligner)
        self.tokenizer = tokenizer
        self.llm_hidden_size = self.params.llm_hidden_size
        self.llm_model_name = self.params.llm_model_name
        self.hidden_size = self.params.reasoner_hidden_size
        self.dim_rel_embedding = self.aligner.kge_model.dim_rel_embedding#self.params.rel_dim
        self.dim_ent_embedding = self.aligner.kge_model.dim_ent_embedding#self.params.ent_dim
        self.params.rel_dim = self.dim_rel_embedding
        self.params.ent_dim = self.dim_ent_embedding
        self.encoder = BertEncoder(params)
        self.dropout = nn.Dropout(self.params.dropout)


        self.initial_proj = nn.Linear(self.dim_rel_embedding, self.hidden_size)
        self.initial_norm = nn.LayerNorm(self.hidden_size, eps=self.params.layer_norm_eps)


        self.llm_proj = nn.Linear(self.llm_hidden_size, self.dim_rel_embedding)
        self.llm_norm = nn.LayerNorm(self.dim_rel_embedding, eps=self.params.layer_norm_eps)
        self.abstract_nodes = nn.Embedding(2, self.dim_rel_embedding)

        if self.params.neg_layer_learnable:
            self.neg_rel_layer = nn.Linear(self.dim_rel_embedding, self.dim_rel_embedding)

        self.abs_idx = torch.tensor([0, 1], dtype=torch.long)

    def to(self, device):
        super().to(device)
        self.abs_idx = self.abs_idx.to(device)
        return self
    def operator_token_init(self):

        raise NotImplementedError
        

    def forward(self, inputs):
        all_query, all_query_padding_mask, path_ids, paths_padding_mask, neg_indicator, tail = inputs

            
        with torch.no_grad():

            origin_size = all_query['input_ids'].size() # batch_size * (1 + num_sub_query) * max_tokens
            model_input = {'input_ids': torch.cat([all_query['input_ids'][:,0], all_query['input_ids'][:,1:,:][all_query_padding_mask[:,3:]].clone().view(-1, origin_size[2])], dim = 0),
                            'attention_mask': torch.cat([all_query['attention_mask'][:,0], all_query['attention_mask'][:,1:,:][all_query_padding_mask[:,3:]].clone().view(-1, origin_size[2])], dim = 0)}
            if self.params.llm_model_name[:5] == 'llama':
                raise NotImplementedError

            else:
                all_nl_emb = self.aligner.llm_model(**model_input, return_dict=True).last_hidden_state
                ori_q = all_nl_emb[:len(all_query['input_ids'][:,0])]#.last_hidden_state
                sub_q = all_nl_emb[len(all_query['input_ids'][:,0]):]#.last_hidden_state
            
            aligner_emb = self.aligner.kg_extract(all_nl_emb[:len(all_query['input_ids'][:,0])], all_query['attention_mask'][:,0].bool())
            sub_q = self.extract_feature_except_pad(sub_q, model_input['attention_mask'][len(all_query['input_ids'][:,0]):])


            sub_q_emb = torch.zeros((origin_size[0], origin_size[1] - 1, sub_q.size(-1)), device = sub_q.device)
            sub_q_emb[all_query_padding_mask[:,3:]] = sub_q
        
        ori_q_emb = self.llm_proj(ori_q)
        sub_q_emb = self.llm_proj(sub_q_emb)
        path_embs = self.get_kg_embs(path_ids, type = 'rel') 
        if self.params.neg_layer_learnable:
            path_embs[neg_indicator] = self.neg_rel_layer(path_embs[neg_indicator])
        else:
            path_embs[neg_indicator] = self.aligner.neg_transform(path_embs[neg_indicator])


        batch_size, num_path, hidden_dim = path_embs.size()


        
        abs_nodes = self.abstract_nodes(self.abs_idx).unsqueeze(0).repeat(batch_size, 1, 1)
        initial_node_embeddings = self.dropout(self.initial_norm(self.initial_proj(torch.cat([abs_nodes, aligner_emb, ori_q_emb, sub_q_emb, path_embs.view(batch_size,-1,hidden_dim)], dim =1))))
        pad_ids = torch.cat([all_query_padding_mask[:,:2],  
                                torch.ones(batch_size, aligner_emb.size(1), dtype = paths_padding_mask.dtype).to(paths_padding_mask.device),
                                all_query['attention_mask'][:,0], 
                                all_query_padding_mask[:,3:], 
                                paths_padding_mask.view(batch_size, -1)], dim = -1)


            

        abs_head, abs_rel = self.encoder(initial_node_embeddings, pad_ids)



        t = self.aligner.kge_model(abs_head, abs_rel)
        logits = self.aligner.kge_model.get_preds(t, tail)

        return logits
    
    def get_kg_embs(self, path_ids, type = None):


        if type == 'ent':
            embs = self.aligner.kge_model.embeddings[0][path_ids]
            embs[path_ids == -1] = 0

        elif type == 'rel':
            embs = self.aligner.kge_model.embeddings[1][path_ids]
            embs[path_ids == -1] = 0
        else:
            raise Exception("KG type error")
        return embs



    def freeze_parameters(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model

    def extract_feature_except_pad(self,embeddings, batch):
        if self.llm_model_name[:5] == 'llama':
            hidden_states = embeddings.hidden_states[-1]
            attention_mask = batch.unsqueeze(-1).expand(hidden_states.size())

            masked_hidden_states = hidden_states * attention_mask
            sum_hidden_states = masked_hidden_states.sum(dim=1)
            count_non_pad_tokens = attention_mask.sum(dim=1)
            sentence_emb = sum_hidden_states / count_non_pad_tokens
        elif self.llm_model_name in ['st', 'l12']:
            if self.llm_model_name == 'st':
                token_embeddings = embeddings#.last_hidden_state #First element of model_output contains all token embeddings
            elif self.llm_model_name == 'l12':
                token_embeddings = embeddings[0]
            else:
                raise NotImplementedError
            input_mask_expanded = batch.unsqueeze(-1).expand(token_embeddings.size()).float()
            sentence_emb = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        else:
            raise NotImplementedError
        return sentence_emb






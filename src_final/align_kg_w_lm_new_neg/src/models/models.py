import torch
import torch.nn as nn


TOKEN_TO_IDX = {
    'TYPE': 0,
    'ENT': 1,
    'REL': 2,
    'AND': 3,
    'OR': 4,
    'NOT': 5,
}
TOKEN_TO_IDX_KG = {
    'ENT': 0,
    'REL': 1,
    'TYPE': 2,
    'AND': 3,
    'OR': 4,
    'NOT': 5
}

TOKEN_TO_IDX_NL_CL = {
    'ENT': 0,
    'REL': 1,
    'TYPE': 2,
    'AND': 3,
    'OR': 4,
    'NOT': 5,
}








TOKEN_TO_IDX_W_M = {
    'TYPE': 0,
    'ENT': 1,
    'REL': 2,
    'AND': 3,
    'OR': 4,
    'NOT': 5,
    'MULTI': 6
}





TOKEN_TO_IDX_TOTAL_OP= {
    'TYPE': 0,
    'ENT': 1,
    'REL': 2,
    'OP': 3
}


TOKNE_TO_IDX_NOT_TYPE = {
    'ENT': 0,
    'REL': 1,
    'AND': 2,
    'OR': 3,
    'NOT': 4,
}
TOKNE_TO_IDX_CL = {
    'ENT': 0,
    'REL': 1,
    'AND': 2,
    'OR': 3,
    'NOT': 4,
}


class Ailgner_self_mask_modality(nn.Module):
    def __init__(self, params, llm_model, kge_model):
        super().__init__()
        self.params = params
        self.kge_hidden_size = params.kge_hidden_size
        self.llm_hidden_size = params.llm_hidden_size
        self.token_to_idx = TOKEN_TO_IDX
        self.query_token = nn.Embedding(len(self.token_to_idx), self.kge_hidden_size)
        self.type_id = self.token_to_idx['TYPE']
        self.output_type_layer = nn.Linear(self.llm_hidden_size, self.kge_hidden_size)


        self.num_heads = self.params.num_heads
        encoder_layer = nn.TransformerEncoderLayer(d_model = self.llm_hidden_size, nhead = self.num_heads, dim_feedforward=self.llm_hidden_size, 
                                                        dropout=self.params.dropout, activation = 'gelu', layer_norm_eps = self.params.eps, batch_first = True)
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers = self.params.num_layers)
        self.input_layer = nn.Linear(self.kge_hidden_size, self.llm_hidden_size)
        self.ent_layer = nn.Linear(self.llm_hidden_size, self.kge_hidden_size)
        self.rel_layer = nn.Linear(self.llm_hidden_size, self.kge_hidden_size)
        self.op_layer = nn.Linear(self.llm_hidden_size, self.kge_hidden_size)


        self.and_head = nn.Linear(self.kge_hidden_size, 1)
        self.or_head = nn.Linear(self.kge_hidden_size, 1)
        self.not_head = nn.Linear(self.kge_hidden_size, 1)
        self.ent_id = self.token_to_idx['ENT']
        self.rel_id = self.token_to_idx['REL']
        self.and_id = self.token_to_idx['AND']
        self.or_id = self.token_to_idx['OR']
        self.not_id = self.token_to_idx['NOT']
        self.kge_model = self.freeze_parameters(kge_model)
        self.llm_model = self.freeze_parameters(llm_model)
        self.ent_emb = self.kge_model.embeddings[0]
        self.rel_emb = self.kge_model.embeddings[1]
        self.num_token_type = len(self.token_to_idx)
        self.neg_transform = nn.Linear(self.kge_hidden_size, self.kge_hidden_size)


    def forward(self, nl_query):
        with torch.no_grad():
            nl_emb = self.llm_model(**nl_query).last_hidden_state
        x = self.input_layer(self.query_token.weight.unsqueeze(0).repeat(nl_emb.size(0), 1, 1))
        inputs_embeds = torch.cat([x, nl_emb], dim = 1)

        if self.params.use_q_visible:
            mask = torch.zeros(nl_query['attention_mask'].size(0), self.query_token.weight.size(0) + nl_query['attention_mask'].size(1), self.query_token.weight.size(0),dtype=torch.bool).to(inputs_embeds.device)

            mask[:, self.query_token.weight.size(0):, :] = True
        elif self.params.use_all_visible:
            mask = torch.ones(nl_query['attention_mask'].size(0), self.query_token.weight.size(0) + nl_query['attention_mask'].size(1), self.query_token.weight.size(0),dtype=torch.bool).to(inputs_embeds.device)
        else:
            mask = torch.zeros(nl_query['attention_mask'].size(0), self.query_token.weight.size(0) + nl_query['attention_mask'].size(1), self.query_token.weight.size(0),dtype=torch.bool).to(inputs_embeds.device)
        nl_mask = nl_query['attention_mask'].unsqueeze(1).repeat(1, self.query_token.weight.size(0) + nl_query['attention_mask'].size(1),1)
        
        expanded_mask = torch.cat([mask, nl_mask], dim = -1)
        expanded_mask = expanded_mask.unsqueeze(1).repeat(1, self.num_heads,1,1).view(-1, self.query_token.weight.size(0) + nl_query['attention_mask'].size(1), self.query_token.weight.size(0) + nl_query['attention_mask'].size(1))

        x = self.layers(src = inputs_embeds, mask = ~expanded_mask)
        
        type_token = self.output_type_layer(x[:, self.type_id, :]).unsqueeze(1)
        ent_token = self.ent_layer(x[:, self.ent_id,:]).unsqueeze(1)
        rel_token = self.rel_layer(x[:, self.rel_id,:]).unsqueeze(1)
        op_token = self.op_layer(x[:, [self.and_id, self.or_id, self.not_id]])
        
        
        return torch.cat([type_token, ent_token, rel_token, op_token], dim = 1)

    def kg_extract(self, nl_emb, attn_mask):

        x = self.input_layer(self.query_token.weight.unsqueeze(0).repeat(nl_emb.size(0), 1, 1))
        inputs_embeds = torch.cat([x, nl_emb], dim = 1)

        if self.params.use_q_visible:
            mask = torch.zeros(attn_mask.size(0), self.query_token.weight.size(0) + attn_mask.size(1), self.query_token.weight.size(0),dtype=torch.bool).to(inputs_embeds.device)

            mask[:, self.query_token.weight.size(0):, :] = True
        elif self.params.use_all_visible:
            mask = torch.ones(attn_mask.size(0), self.query_token.weight.size(0) + attn_mask.size(1), self.query_token.weight.size(0),dtype=torch.bool).to(inputs_embeds.device)

        else:
            mask = torch.zeros(attn_mask.size(0), self.query_token.weight.size(0) + attn_mask.size(1), self.query_token.weight.size(0),dtype=torch.bool).to(inputs_embeds.device)

        nl_mask = attn_mask.unsqueeze(1).repeat(1, self.query_token.weight.size(0) + attn_mask.size(1),1)
        
        expanded_mask = torch.cat([mask, nl_mask], dim = -1)
        expanded_mask = expanded_mask.unsqueeze(1).repeat(1, self.num_heads,1,1).view(-1, self.query_token.weight.size(0) + attn_mask.size(1), self.query_token.weight.size(0) + attn_mask.size(1))

        x = self.layers(src = inputs_embeds, mask = ~expanded_mask)

        type_token = self.output_type_layer(x[:, self.type_id, :]).unsqueeze(1)
        ent_token = self.ent_layer(x[:, self.ent_id,:]).unsqueeze(1)
        rel_token = self.rel_layer(x[:, self.rel_id,:]).unsqueeze(1)
        op_token = self.op_layer(x[:, [self.and_id, self.or_id, self.not_id]])

        return torch.cat([ent_token, rel_token, type_token, op_token], dim = 1)

    def freeze_parameters(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model

    def train_step(self, nl_query, pos_ent, pos_rel, neg_ent, neg_rel, q_type_neg = None, q_type = None, type_to_id = None, rel_neg_indicator = None):
        x = self(nl_query)


        ent_token = x[:, self.ent_id,:]
        rel_token = x[:,self.rel_id,:]
        and_token = self.and_head(x[:, self.and_id, :])
        or_token = self.or_head(x[:, self.or_id, :])
        not_token = self.not_head(x[:, self.not_id, :])
        pos_ent = self.get_kg_embs(pos_ent, type = 'ent')

        neg_ent = self.get_kg_embs(neg_ent, type = 'ent')

        target_ent_emb = torch.cat([pos_ent.unsqueeze(1), neg_ent], dim = 1)

        ent_token_preds = torch.bmm(ent_token.unsqueeze(1), target_ent_emb.transpose(1,2)).squeeze(1)
        
        pos_rel = self.get_kg_embs(pos_rel, type = 'rel')
        neg_rel = self.get_kg_embs(neg_rel, type = 'rel')

        target_rel_emb = torch.cat([pos_rel.unsqueeze(1), neg_rel], dim = 1)

        target_rel_emb[rel_neg_indicator] = self.neg_transform(target_rel_emb[rel_neg_indicator])
        rel_token_preds = torch.bmm(rel_token.unsqueeze(1), target_rel_emb.transpose(1,2)).squeeze(1)

        type_token = x[:,self.type_id,:]

        q_type_in_batch_pos = []

        for i in range(len(type_to_id)):


            q_type_in_batch_pos.append(torch.mean(type_token[torch.nonzero(q_type == i).squeeze(1)], dim = 0))
        q_type_in_batch_pos = torch.stack(q_type_in_batch_pos)

        target_type_emb = torch.cat([type_token.unsqueeze(1), type_token[q_type_neg]], dim = 1)
        type_token_preds = torch.bmm(q_type_in_batch_pos[q_type].unsqueeze(1), target_type_emb.transpose(1,2)).squeeze(1)

        return ent_token_preds, rel_token_preds, type_token_preds, and_token, or_token, not_token


    def test_step(self, nl_query):
        x = self(nl_query)


        ent_token = x[:, self.ent_id,:]
        rel_token = x[:,self.rel_id,:]
        
        and_token = torch.sigmoid(self.and_head(x[:, self.and_id, :]))
        or_token = torch.sigmoid(self.or_head(x[:, self.or_id, :]))
        not_token = torch.sigmoid(self.not_head(x[:, self.not_id, :]))
        target_rel_emb = self.rel_emb.unsqueeze(0).repeat(ent_token.size(0), 1, 1)
        target_rel_emb_neg = self.neg_transform(target_rel_emb.clone().detach())
        ent_token = torch.bmm(ent_token.unsqueeze(1), self.ent_emb.unsqueeze(0).repeat(ent_token.size(0), 1, 1).transpose(1,2)).squeeze(1)
        rel_token_pos = torch.bmm(rel_token.unsqueeze(1), target_rel_emb.transpose(1,2)).squeeze(1)

        rel_token_neg = torch.bmm(rel_token.unsqueeze(1), target_rel_emb_neg.transpose(1,2)).squeeze(1)

        type_token = x[:,self.type_id,:]

        return ent_token, {'pos_rel': rel_token_pos, 'neg_rel': rel_token_neg}, type_token, and_token, or_token, not_token


    def get_kg_embs(self, path_ids, type = None):

        if type == 'ent':
            embs = self.ent_emb[path_ids]
            embs[path_ids == -1] = 0

        elif type == 'rel':
            embs = self.rel_emb[path_ids]
            embs[path_ids == -1] = 0

        else:
            raise Exception
        return embs

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output#.last_hidden_state #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


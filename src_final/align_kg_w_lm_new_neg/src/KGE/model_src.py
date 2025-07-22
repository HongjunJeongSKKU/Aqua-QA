import torch.nn as nn
import torch

class KGE(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.ent_embedding = None
        self.rel_embedding = None
        self.ent_embedding = None
        self.rel_embedding = None
        self.params = params
        self.dim_rel_embedding = self.params.kge_hidden_size#.dim_rel_embedding
        self.dim_ent_embedding = self.params.kge_hidden_size#.dim_ent_embedding
        self.num_ents = self.params.num_ents
        self.num_rels = self.params.num_rels
        self.init_size = 1e-3

    def forward(self, lhs, rel):
        raise NotImplemented

    def load_from_ckpt_path(self, ckpt_path):
        raise NotImplemented

    def get_preds(self, pred_embedding, tgt_ent_idx=None):
        raise NotImplemented

    @staticmethod
    def calc_preds(pred_embedding, ent_embedding, tgt_ent_idx=None):
        if tgt_ent_idx is None:
            tgt_ent_embedding = ent_embedding.weight.transpose(0, 1)

            scores = pred_embedding @ tgt_ent_embedding

        else:
            tgt_ent_embedding = ent_embedding[tgt_ent_idx]

            pred_embedding = pred_embedding.unsqueeze(-1)

            scores = torch.bmm(tgt_ent_embedding, pred_embedding)
            scores = scores.squeeze(-1)

        return scores

class Complex(KGE):
    def __init__(self, params):
        super().__init__(params)



        self.params = params
        self.num_ents = self.params.num_ents
        self.num_rels = self.params.num_rels
        self.dim_ent_embedding = self.params.kge_hidden_size#.dim_ent_embedding
        self.dim_rel_embedding = self.params.kge_hidden_size#.dim_rel_embedding
        self.rank = self.dim_ent_embedding // 2

        self.embeddings = [self.ent_embedding, self.rel_embedding]


        self.ent_embedding = nn.Parameter(torch.zeros(self.num_ents, 2 * self.rank))#nn.Embedding(num_ents, 2 * self.rank)
        self.rel_embedding = nn.Parameter(torch.zeros(self.num_rels, 2 * self.rank))#nn.Embedding(num_rels, 2 * self.rank)



        self.embeddings = [self.ent_embedding, self.rel_embedding]


    def forward_emb(self, lhs, rel, to_score_idx=None):
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        if not to_score_idx:
            to_score = self.embeddings[0].weight
        else:
            to_score = self.embeddings[0](to_score_idx)

        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        return ((lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
                (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1))

    def forward(self, lhs, rel):
        lhs = torch.chunk(lhs, 2, -1)
        rel = torch.chunk(rel, 2, -1)

        output = ([lhs[0] * rel[0] - lhs[1] * rel[1], lhs[0] * rel[1] + lhs[1] * rel[0]])
        output = torch.cat(output, dim=-1)

        return output

    def get_preds(self, pred_embedding, tgt_ent_idx=None):
        return KGE.calc_preds(pred_embedding, self.ent_embedding, tgt_ent_idx)

    def get_factor(self, x):
        lhs = self.ent_embedding(x[0])
        rel = self.rel_embedding(x[1])
        rhs = self.ent_embedding(x[2])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        return (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))

    def load_from_ckpt_path(self, ckpt_path):
        params = torch.load(ckpt_path)

        try:

            self.embeddings[0].data = params['embeddings.0.weight']
            self.embeddings[1].data = params['embeddings.1.weight']
        except:
            self.embeddings[0].data = params['embeddings.0']
            self.embeddings[1].data = params['embeddings.1']

        self.ent_embedding_norm_mean = self.embeddings[0].data.norm(p=2, dim=1).mean().item()
        self.rel_embedding_norm_mean = self.embeddings[1].data.norm(p=2, dim=1).mean().item()

        self.embeddings[0].data /= self.ent_embedding_norm_mean
        self.embeddings[1].data /= self.rel_embedding_norm_mean

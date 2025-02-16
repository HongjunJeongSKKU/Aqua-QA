from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from tqdm import tqdm
from utils import batch_to_gpu, batch_to_gpu_test, save_model, collate_fn, TYPE_TO_IDX_TRAIN, log_metrics, TYPE_TO_IDX_TEST
import time
import logging
import collections
from collections import defaultdict
import os
import gc
#from sklearn.metrics import accuracy_score
from sklearn import metrics
from transformers import get_cosine_schedule_with_warmup
import math

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1,
                 reduction="mean"):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
            if self.reduction == 'sum' else loss

    def linear_combination(self, x, y, smoothing=None):
        if smoothing is None:
            smoothing = self.smoothing

        return smoothing * x + (1 - smoothing) * y

    def forward(self, preds, target):

        assert 0 <= self.smoothing < 1
        smoothing = self.smoothing

        n = preds.size(-1)


        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1)) / n
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction
        )
        return self.linear_combination(loss, nll, smoothing), log_preds




class Trainer():
    def __init__(self, model, params, train = None, valid = None, test = None, original = False):
        self.model = model
        self.params = params
        self.train_data = train
        self.valid_data = valid
        self.test_data = test
        self.type_to_id = TYPE_TO_IDX_TRAIN if not original else TYPE_TO_IDX_TEST
        self.id_to_type = {v:k for k, v in self.type_to_id.items()}
        param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.params.l2},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=params.lr,

        )

        
        self.loss_fn = LabelSmoothingLoss(smoothing = self.params.smoothing, reduction='mean')
        self.loss_bce = nn.BCEWithLogitsLoss(reduction='mean')
        if self.train_data is not None:
            if self.params.use_scheduler == 'cosine':
                total_steps = math.ceil(len(self.train_data) / self.params.train_batch_size) * self.params.num_epochs
                warmup_steps = math.ceil(total_steps * self.params.warm_up_ratio)
                self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps, num_cycles = 0.5)
            else:
                pass
    def train_epoch(self):
        self.model.train()
        train_dataloader = DataLoader(self.train_data, batch_size = self.params.train_batch_size, shuffle=True, num_workers=16, collate_fn = collate_fn)
        total_loss = 0
        total_batches = 0 

        total_loss_dict = {'type': 0, 'ent': 0, 'rel': 0, 'and': 0, 'or': 0, 'not': 0}
        b_start_time = time.time()
        logging.info("Train start")
        
        for b_idx, batch in enumerate(tqdm(train_dataloader)):
            self.optimizer.zero_grad()

            query, pos_ent, pos_rel, neg_ent, neg_rel, q_type, and_label, or_label, not_label, rel_neg_indicator = batch_to_gpu(batch, self.params.device)
            q_type_in_batch_neg = []

            for i in range(len(self.type_to_id)):
                q_type_in_batch_neg.append(torch.nonzero(~(q_type == i)).squeeze(1))


            min_length = min([t.size(0) for t in q_type_in_batch_neg])
            min_length = min(min_length, self.params.negative_sample_size)

            q_type_in_batch_neg = [t[torch.randperm(t.size(0))][:min_length] for t in  q_type_in_batch_neg]
            q_type_in_batch_neg = torch.stack(q_type_in_batch_neg)
            q_type_in_batch_neg = q_type_in_batch_neg[q_type]

            ent_preds, rel_preds, type_preds, and_preds, or_preds, not_preds = self.model.train_step(query, pos_ent, pos_rel, neg_ent, neg_rel, q_type_in_batch_neg, q_type, self.type_to_id, rel_neg_indicator)

            q_type_loss, _ = self.loss_fn(type_preds, torch.zeros(type_preds.size(0), dtype=torch.long, device= type_preds.device))
            
            ent_loss, _ = self.loss_fn(ent_preds, torch.zeros(ent_preds.size(0), dtype=torch.long, device= ent_preds.device))
            rel_loss, _ = self.loss_fn(rel_preds, torch.zeros(rel_preds.size(0), dtype=torch.long, device= rel_preds.device))

            and_loss = self.loss_bce(and_preds.squeeze(1), and_label)
            or_loss = self.loss_bce(or_preds.squeeze(1), or_label)
            not_loss = self.loss_bce(not_preds.squeeze(1), not_label)


            loss_dict = {'type': q_type_loss, 'ent': ent_loss, 'rel': rel_loss, 'and': and_loss, 'or': or_loss, 'not': not_loss}

            loss = 0.0
            loss += sum(loss_dict.values())
            loss.backward()
            self.optimizer.step()
            if hasattr(self, 'lr_scheduler'):
                self.lr_scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
            with torch.no_grad():
                total_loss += loss.item()
                for k,v in loss_dict.items():
                    total_loss_dict[k] += v.item()

            total_batches += 1
            if b_idx % self.params.train_log_step == 0:
                b_end_time = time.time() - b_start_time
                logging.info(f'Average total Loss to {b_idx}th batch,: {total_loss / total_batches}, Time: {b_end_time}' if self.params.use_scheduler == 'not_use' else f'Average total Loss to {b_idx}th batch,: {total_loss / total_batches}, Current lr: {current_lr}, Time: {b_end_time}')
                for k,v in total_loss_dict.items():
                    logging.info(f'Average {k} Loss to {b_idx}th batch : {v / total_batches}' )
                b_start_time = time.time()
        for k, v in total_loss_dict.items():
            total_loss_dict[k] = v / total_batches
        return total_loss / total_batches, total_loss_dict 

    def train(self):
        best_mrr = 0.
        for epoch in range(1, self.params.num_epochs + 1):

            time_start = time.time()
            loss, total_loss_dict = self.train_epoch()
            time_elapsed = time.time() - time_start
            for k,v in total_loss_dict.items():
                logging.info(f'Epoch {epoch}  {k} Loss to : {v}' )
            logging.info(f'Epoch {epoch} with loss: {loss} in {time_elapsed}')

            if epoch % self.params.valid_per_epoch == 0:
                torch.cuda.empty_cache()
                gc.collect()
                logging.info("valid start")
                metrics_ent, metrics_rel = self.validation(self.valid_data)
                #valid_all_metrics, valid_avg_list = self.evaluate(metrics, 'valid', epoch)
                _, ent_avg = self.evaluate(metrics_ent, 'Entity')
                _, rel_avg = self.evaluate(metrics_rel, 'Relation')
                if ent_avg['Ap']['MRR'] > best_mrr:
                    save_model(self.model, self.params, epoch, 'best')
                    logging.info(f'Best model saved in epoch {epoch}')
                    best_mrr = ent_avg['Ap']['MRR']

                # 수정된 state_dict 저장

                logging.info("valid finished")
                torch.cuda.empty_cache()
                gc.collect()
            if epoch % self.params.save_per_epoch == 0:
                save_model(self.model, self.params, epoch)
                logging.info(f'Model saved in epoch {epoch}')
        logging.info("Train finished")



        logging.info("training finished")
    def test(self):
        logging.info("Test start")
        metrics_ent, metrics_rel = self.validation(self.test_data)
        _, _ = self.evaluate(metrics_ent, 'Entity')
        _, _ = self.evaluate(metrics_rel, 'Relation')
        logging.info("Test finished")

    def validation(self, input_data):
        valid_dataloader = DataLoader(input_data, batch_size = self.params.eval_batch_size, shuffle=False, num_workers = 16)#, collate_fn = collate_fn_test)
        results_ent = {v: [] for v in self.type_to_id.values()} 
        results_rel = {v: [] for v in self.type_to_id.values()} 

        self.model.eval()

        all_and = {k : {'preds': [], 'labels': []} for k in self.type_to_id.keys()} 
        all_or = {k : {'preds': [], 'labels': []} for k in self.type_to_id.keys()} 
        all_not = {k : {'preds': [], 'labels': []} for k in self.type_to_id.keys()} 

        for batch in tqdm(valid_dataloader):
            all_query, ents_answer, rels_answer, query_type, and_label, or_label, not_label, rel_neg_indicator  = batch_to_gpu_test(batch, self.params.device)
            query_type = query_type.tolist()
            all_ents = torch.arange(input_data.num_ents, dtype = torch.long).expand(ents_answer.size(0), -1).to(self.params.device)
            all_rels = torch.arange(input_data.num_rels, dtype = torch.long).expand(rels_answer['pos_rel'].size(0), -1).to(self.params.device)
            with torch.no_grad():
                ent_token, rel_token, _, and_token, or_token, not_token = self.model.test_step(all_query)


            ent_token = ent_token.cpu()
            rel_token = {k: val.cpu() for k, val in rel_token.items()}
            and_label = and_label.tolist()
            or_label = or_label.tolist()
            not_label = not_label.tolist()


            and_token = and_token.cpu().squeeze(1).tolist()
            or_token = or_token.cpu().squeeze(1).tolist()
            not_token = not_token.cpu().squeeze(1).tolist()


            for idx in range(0, len(and_token)):

                q_idx = self.id_to_type[query_type[idx]]
                all_and[q_idx]['preds'].append(and_token[idx])
                all_and[q_idx]['labels'].append(and_label[idx])

                all_or[q_idx]['preds'].append(or_token[idx])
                all_or[q_idx]['labels'].append(or_label[idx])

                all_not[q_idx]['preds'].append(not_token[idx])
                all_not[q_idx]['labels'].append(not_label[idx])          




            argsort = torch.argsort(ent_token, dim=1, descending=True)
            ranking = argsort.clone()
            ranking = ranking.scatter_(1, argsort, all_ents.cpu())
            for idx in range(0, len(ent_token)):
                all_ids = torch.where(ents_answer[idx])[0]
                num_hard = len(all_ids)
                cur_ranking = ranking[idx, all_ids]
                cur_ranking, indices = torch.sort(cur_ranking)
                answer_list = torch.arange(num_hard)
                cur_ranking = cur_ranking - answer_list + 1 
                mrr = torch.mean(1. / cur_ranking).item()
                h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()
                results_ent[query_type[idx]].append({
                    'MRR': mrr,
                    'HITS1': h1,
                    'HITS3': h3,
                    'HITS10': h10,
                })

            ranking_dict = {}
            for key, val in rel_token.items():
                argsort = torch.argsort(val, dim=1, descending=True)
                ranking = argsort.clone()
                ranking = ranking.scatter_(1, argsort, all_rels.cpu())
                ranking_dict[key] = ranking

            for idx in range(0, len(rel_token['pos_rel'])):
                
                if not rel_neg_indicator[idx]:

                    all_ids = torch.where(rels_answer['pos_rel'][idx])[0]
                    num_hard = len(all_ids)
                    cur_ranking = ranking_dict['pos_rel'][idx, all_ids]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    answer_list = torch.arange(num_hard)
                    cur_ranking = cur_ranking - answer_list + 1 
                    mrr = torch.mean(1. / cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()
                    results_rel[query_type[idx]].append({
                        'MRR': mrr,
                        'HITS1': h1,
                        'HITS3': h3,
                        'HITS10': h10,
                    })
                else:
                    all_ids = torch.where(rels_answer['pos_rel'][idx])[0]
                    num_hard = len(all_ids)
                    cur_ranking = ranking_dict['pos_rel'][idx, all_ids]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    answer_list = torch.arange(num_hard)
                    cur_ranking = cur_ranking - answer_list + 1 

                    all_ids_neg = torch.where(rels_answer['neg_rel'][idx])[0]
                    num_hard_neg = len(all_ids_neg)
                    cur_ranking_neg = ranking_dict['neg_rel'][idx, all_ids_neg]
                    cur_ranking_neg, indices_neg = torch.sort(cur_ranking_neg)
                    answer_list_neg = torch.arange(num_hard_neg)
                    cur_ranking_neg = cur_ranking_neg - answer_list_neg + 1 
                    cur_ranking = torch.cat([cur_ranking, cur_ranking_neg])




                    mrr = torch.mean(1. / cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

                    results_rel[query_type[idx]].append({
                        'MRR': mrr,
                        'HITS1': h1,
                        'HITS3': h3,
                        'HITS10': h10,
                    })



        ############
        del all_query#, all_query_padding_mask, paths, paths_padding_mask, easy, hard, query_type, type_emb#, pos_emb, adj
        #del logits, argsort, ranking, all_ents
        torch.cuda.empty_cache()
        gc.collect()
        ###########

        for k, v in {'and': all_and, 'or': all_or, 'not': all_not}.items():
            all_average_preds = []
            all_average_labels = []
            for q_type, score_dict in v.items():

                logging.info(f"{k.upper()} {q_type} accuracy: {metrics.accuracy_score(score_dict['labels'], [1 if val > 0.5 else 0 for val in score_dict['preds']])}")
                all_average_labels.extend(score_dict['labels'])
                all_average_preds.extend(score_dict['preds'])

            logging.info(f"{k.upper()} all average auc: {metrics.roc_auc_score(all_average_labels, all_average_preds)}")
            logging.info(f"{k.upper()} all average auc_pr: {metrics.average_precision_score(all_average_labels, all_average_preds)}")




        metric_ent = self.get_ranking(results_ent)
        metric_rel = self.get_ranking(results_rel)
        return metric_ent, metric_rel#, all_and, all_or, all_not

    def get_ranking(self, results):

        logs = collections.defaultdict(list)
        for query_structure, res in results.items():
            query_structure = self.id_to_type[query_structure]
            logs[query_structure].extend(res)
        metrics = collections.defaultdict(lambda: collections.defaultdict(int))

        for idx, query_structure in enumerate(logs):
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]]) / len(
                    logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])
        return metrics 

    def evaluate(self, metrics, mode, epoch = None):
        all_metrics = defaultdict(float)

        p_query_types = [query_type for query_type in metrics.keys() if 'n' not in query_type]
        n_query_types = [query_type for query_type in metrics.keys() if 'n' in query_type]
        avg_list = {'Ap': None, 'An': None}
        for types, name in zip([p_query_types, n_query_types], ['Ap', 'An']):
            average_metrics = defaultdict(float)
            num_query_structures = 0
            num_queries = 0

            for query_type in types:
                # {'1p': {mrr:..., hit1:...,...}, '2p': {...},}
                log_metrics(mode + " " + query_type,epoch, metrics[query_type])
                for metric in metrics[query_type]:
                    #writer.add_scalar("_".join([mode, query_type, metric]),
                    #                metrics[query_type][metric], step)
                    all_metrics["_".join([query_type, metric])] = metrics[query_type][metric]
                    if metric != 'num_queries':
                        average_metrics[metric] += metrics[query_type][metric]
                num_queries += metrics[query_type]['num_queries']
                num_query_structures += 1

            for metric in average_metrics:
                if metric == 'num_queries':
                    pass
                else:
                    average_metrics[metric] /= num_query_structures
                all_metrics["_".join(["average", metric])] = average_metrics[metric]
            log_metrics(f'{mode} {name} average', epoch, average_metrics)
            avg_list[name] = average_metrics
        return all_metrics, avg_list






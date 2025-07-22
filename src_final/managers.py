from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

import torch.optim as optim
from tqdm import tqdm
from utils import batch_to_gpu, batch_to_gpu_test, log_metrics, save_model, TYPE_TO_IDX
import time
import logging
import collections
from collections import defaultdict
import os
import gc
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
    def __init__(self, model, params, train = None, valid = None, test = None):
        self.model = model
        self.params = params
        self.train_data = train
        self.valid_data = valid
        self.test_data = test
        self.type_to_id = TYPE_TO_IDX
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
        if self.train_data is not None:
            if self.params.use_scheduler == 'cosine':
                total_steps = math.ceil(len(self.train_data) / self.params.train_batch_size) * self.params.num_epochs
                warmup_steps = math.ceil(total_steps * self.params.warm_up_ratio)
                self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps, num_cycles = 0.5)
            else:
                pass

        
        self.loss_fn = LabelSmoothingLoss(smoothing = self.params.smoothing, reduction='none')
    
    def train_epoch(self):
        self.model.train()
        train_dataloader = DataLoader(self.train_data, batch_size = self.params.train_batch_size, shuffle=True, num_workers=16)
        total_loss = 0
        total_batches = 0 
        b_start_time = time.time()
        logging.info("Train start")
        
        for b_idx, batch in enumerate(tqdm(train_dataloader)):
            all_query, all_query_padding_mask, paths, paths_padding_mask, neg_indicator, pos, negs = batch_to_gpu(batch, self.params.device)
            self.optimizer.zero_grad()
            logits = self.model((all_query, all_query_padding_mask, paths, paths_padding_mask, neg_indicator, torch.cat([pos,negs], dim = -1)))
            loss, _ = self.loss_fn(logits, torch.zeros(logits.size(0), dtype=torch.long, device=logits.device))
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            if hasattr(self, 'lr_scheduler'):
                self.lr_scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
            with torch.no_grad():
                total_loss += loss.item()
            total_batches += 1
            if b_idx % self.params.train_log_step == 0:
                b_end_time = time.time() - b_start_time
                logging.info(f'Average Loss to {b_idx}th batch,: {total_loss / total_batches}, Time: {b_end_time}' if self.params.use_scheduler == 'not_use' else f'Average Loss to {b_idx}th batch,: {total_loss / total_batches}, current_lr: {current_lr}, Time: {b_end_time}') 
                b_start_time = time.time()
        return total_loss / total_batches 

    def train(self):
        best_mrr = 0.
        for epoch in range(1, self.params.num_epochs + 1):

            time_start = time.time()
            loss = self.train_epoch()
            time_elapsed = time.time() - time_start
            logging.info(f'Epoch {epoch} with loss: {loss} in {time_elapsed}')
            if epoch % self.params.valid_per_epoch == 0:
                torch.cuda.empty_cache()
                gc.collect()
                logging.info("valid start")
                metrics = self.validation(self.valid_data)
                _, valid_avg_list = self.evaluate(metrics, 'valid', epoch)

                if valid_avg_list['Ap']['MRR'] > best_mrr:
                    save_model(self.model, self.params, epoch, 'best')
                    logging.info(f'Best model saved in epoch {epoch}')
                    best_mrr = valid_avg_list['Ap']['MRR']

                if epoch % self.params.save_per_epoch == 0:
                    save_model(self.model, self.params, epoch)
                    logging.info(f'Model saved in epoch {epoch}')
                logging.info("valid finished")
                torch.cuda.empty_cache()
                gc.collect()
        logging.info("Train finished")



        logging.info("training finished")
    
    def test_new(self):
        logging.info("Test start")
        metrics = self.validation_new(self.test_data)
        _, _ = self.evaluate_new(metrics, 'test')
        logging.info("Test finished")


    def validation_new(self, input_data):

        valid_dataloader = DataLoader(input_data, batch_size = self.params.eval_batch_size, shuffle=False, num_workers = 16)
        if 'hard' in self.test_data.mode:
            hard_unique_query_type = set(self.test_data.data['query_type_id'])
            results = {}
            for v in self.type_to_id.values():
                if v in list(hard_unique_query_type):
                    results[v] = []
                else:
                    pass


        else:
            results = {v: [] for v in self.type_to_id.values()} 

        self.model.eval()

        for batch in tqdm(valid_dataloader):

            all_query, all_query_padding_mask, paths, paths_padding_mask, neg_indicator, easy, hard, query_type  = batch_to_gpu_test(batch, self.params.device)
            query_type = query_type.tolist()
            all_ents = torch.arange(input_data.num_ents, dtype = torch.long).expand(easy.size(0), -1).to(self.params.device)
            with torch.no_grad():
                logits = self.model((all_query, all_query_padding_mask, paths, paths_padding_mask, neg_indicator, all_ents))
            logits = logits.cpu()
            argsort = torch.argsort(logits, dim=1, descending=True)
            ranking = argsort.clone()
            ranking = ranking.scatter_(1, argsort, all_ents.cpu())
            for idx in range(0, len(easy)):
                easy_ids = torch.where(easy[idx])[0]
                hard_ids = torch.where(hard[idx])[0]
                num_easy = len(easy_ids)
                num_hard = len(hard_ids)
                all_ids = torch.cat([easy_ids, hard_ids])
                cur_ranking = ranking[idx, all_ids]
                cur_ranking, indices = torch.sort(cur_ranking)

                if torch.any(cur_ranking == 0):
                    new_hits_1 = 1
                else:
                    new_hits_1 = 0

                masks = indices >= num_easy
                answer_list = torch.arange(num_hard + num_easy)
                cur_ranking = cur_ranking - answer_list + 1 
                cur_ranking = cur_ranking[masks]

                mrr = torch.mean(1. / cur_ranking).item()
                h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()
                results[query_type[idx]].append({
                    'MRR': mrr,
                    'HITS1': h1,
                    'HITS3': h3,
                    'HITS10': h10,
                    'HITS1_NEW': new_hits_1,
                    'num_hard_answer': num_hard
                    #'num_samples': 1
                })
        ############
        del all_query, all_query_padding_mask, paths, paths_padding_mask, neg_indicator, easy, hard, query_type
        del logits, argsort, ranking, all_ents
        torch.cuda.empty_cache()
        gc.collect()
        ###########
        logs = collections.defaultdict(list)
        for query_structure, res in results.items():
            query_structure = self.id_to_type[query_structure]
            logs[query_structure].extend(res)
        metrics = collections.defaultdict(lambda: collections.defaultdict(int))

        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                if metric in ['num_samples']:
                    metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]])
                    continue
                metrics[query_structure][f"{metric}_sum"] = sum([log[metric] for log in logs[query_structure]])
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]]) / len(
                    logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        return metrics 
    def validation(self, input_data):
        valid_dataloader = DataLoader(input_data, batch_size = self.params.eval_batch_size, shuffle=False, num_workers = 16)
        results = {v: [] for v in self.type_to_id.values()} 

        self.model.eval()
        
        for batch in tqdm(valid_dataloader):
            all_query, all_query_padding_mask, paths, paths_padding_mask, neg_indicator, easy, hard, query_type  = batch_to_gpu_test(batch, self.params.device)
            query_type = query_type.tolist()
            all_ents = torch.arange(input_data.num_ents, dtype = torch.long).expand(easy.size(0), -1).to(self.params.device)
            with torch.no_grad():
                logits = self.model((all_query, all_query_padding_mask, paths, paths_padding_mask, neg_indicator, all_ents))
            logits = logits.cpu()
            argsort = torch.argsort(logits, dim=1, descending=True)
            ranking = argsort.clone()
            ranking = ranking.scatter_(1, argsort, all_ents.cpu())
            for idx in range(0, len(easy)):
                easy_ids = torch.where(easy[idx])[0]
                hard_ids = torch.where(hard[idx])[0]
                num_easy = len(easy_ids)
                num_hard = len(hard_ids)
                all_ids = torch.cat([easy_ids, hard_ids])
                cur_ranking = ranking[idx, all_ids]
                cur_ranking, indices = torch.sort(cur_ranking)
                masks = indices >= num_easy
                answer_list = torch.arange(num_hard + num_easy)
                cur_ranking = cur_ranking - answer_list + 1 
                cur_ranking = cur_ranking[masks]
                mrr = torch.mean(1. / cur_ranking).item()
                h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()
                results[query_type[idx]].append({
                    'MRR': mrr,
                    'HITS1': h1,
                    'HITS3': h3,
                    'HITS10': h10,
                    'num_hard_answer': num_hard
                })
        ############
        del all_query, all_query_padding_mask, paths, paths_padding_mask, neg_indicator, easy, hard, query_type
        del logits, argsort, ranking, all_ents
        torch.cuda.empty_cache()
        gc.collect()
        ###########
        logs = collections.defaultdict(list)
        for query_structure, res in results.items():
            query_structure = self.id_to_type[query_structure]
            logs[query_structure].extend(res)
        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
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
                log_metrics(mode + " " + query_type,epoch, metrics[query_type])
                for metric in metrics[query_type]:

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


    def evaluate_new(self, metrics, mode, epoch = None):
        all_metrics = defaultdict(float)

        p_query_types = [query_type for query_type in metrics.keys() if 'n' not in query_type]
        n_query_types = [query_type for query_type in metrics.keys() if 'n' in query_type]
        avg_list = {'Ap': None, 'An': None}
        for types, name in zip([p_query_types, n_query_types], ['Ap', 'An']):
            average_metrics = defaultdict(float)
            num_query_structures = 0
            num_queries = 0

            for query_type in types:
                log_metrics(mode + " " + query_type,epoch, metrics[query_type])
                for metric in metrics[query_type]:

                    all_metrics["_".join([query_type, metric])] = metrics[query_type][metric]
                    average_metrics[metric] += metrics[query_type][metric]
                    
                num_queries += metrics[query_type]['num_queries']
                num_query_structures += 1

            for metric in average_metrics:
                if metric == 'num_queries':
                    pass
                elif '_sum' in metric:
                    average_metrics[metric] /= average_metrics['num_queries']
                else:
                    average_metrics[metric] /= num_query_structures
                all_metrics["_".join(["average", metric])] = average_metrics[metric]
            log_metrics(f'{mode} {name} average', epoch, average_metrics)
            avg_list[name] = average_metrics
        return all_metrics, avg_list
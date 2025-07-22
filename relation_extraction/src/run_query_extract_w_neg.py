import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['WANDB_DISABLED'] = 'true'
from transformers import (T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, 
                          T5Tokenizer, DataCollatorForSeq2Seq, SchedulerType)

from datasets import DatasetDict


import logging
import json
import torch
import random
import numpy as np
import argparse
from utils import * 
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import dill


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(params):

    if params.data_path in ['UMLS', 'CODEX', 'FB15k-237']:
        original_df = pd.read_parquet(os.path.join(f"../data/{params.data_path}",f"{params.train_data_name}.parquet"))
        train_df, test_df = train_test_split(original_df, test_size=0.05, random_state=0)
        train_data = get_preprocessed_query_dataset_all_in_one_w_neg(train_df)
        test_data = get_preprocessed_query_dataset_all_in_one_w_neg(test_df)
        with open(os.path.join(f"../data/{params.data_path}", "rel2id.pkl"), 'rb') as f:
            rel2id = pickle.load(f)

        rel_set = set()

        for key in rel2id.keys():
            rel_set.add(f"<{key}>" if params.constrain_token else f"{key}")

        dataset = DatasetDict({
            'train': train_data,
            'test': test_data
        })
    else:
        raise Exception('data path error')





    if params.model_name[:2] == 'T5':
        tokenizer = T5Tokenizer.from_pretrained(f"google/flan-{params.model_name.lower()}")
        model = T5ForConditionalGeneration.from_pretrained(f"google/flan-{params.model_name.lower()}")

        if params.constrain_token:

            special_tokens = {'additional_special_tokens': ['<PATH>', '</PATH>', '<NEG>', '<SEP>'] + list(rel_set) }
            tokenizer.add_tokens(special_tokens['additional_special_tokens'])
            allowed_token = special_tokens['additional_special_tokens'] + ['</s>', '<pad>']

            allowed_token_ids = sorted(tokenizer.convert_tokens_to_ids(allowed_token))

            with open(os.path.join(output_directory, 'allowed_token.pkl'), 'wb') as pickle_file:
                pickle.dump(allowed_token, pickle_file)

        else:
            raise NotImplementedError

        model.resize_token_embeddings(len(tokenizer))
        if params.constrain_token:
            num_new_tokens = len(special_tokens['additional_special_tokens'])
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
        
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    else:
        raise NotImplementedError
    model.config.use_cache = False
        

    params.max_length_input = 100
    params.max_length_label = 2 + 2 * params.max_hop + 1 if params.constrain_token else calculate_max_length_query_extraction(dataset, tokenizer, mode='label')
    logging.info(f"input_max_length: {params.max_length_input}")
    logging.info(f"label_max_length: {params.max_length_label}")
    params_dict = vars(params)

    with open(os.path.join(output_directory, "params.json"), "w") as json_file:
        json.dump(params_dict, json_file, indent=4)

    def preprocess_function(examples):

        input_text = [f"Please extract the relations that exist in the knowledge graph from the following question: {query}" for query in examples['query']]
        formatted_path = []
        for pos, neg in zip(examples['pos_rel'], examples['neg_rel']):
            if len(neg) == 0:
                tmp = f"<PATH>{'<SEP>'.join(f'<{p}>' for p in pos)}</PATH>"
            else:
                tmp = f"<PATH>{'<SEP>'.join(f'<{p}>' for p in pos)}"
                tmp += f"<SEP><NEG>{'<SEP><NEG>'.join(f'<{p}>' for p in neg)}</PATH>"
            formatted_path.append(tmp)

        model_inputs = tokenizer(input_text, truncation=True, padding='max_length', max_length=params.max_length_input, pad_to_max_length = True)
        labels = tokenizer(formatted_path, truncation=True, padding='max_length', max_length=params.max_length_label, pad_to_max_length = True)

        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs


    if params.constrain_token:
        logging.info("constrain tokens")
        serialized_preprocess_function = dill.dumps(preprocess_function)

        tokenized_dataset = dataset.map(lambda x: dill.loads(serialized_preprocess_function)(x), batched=True, num_proc=16)
    else:
        raise NotImplementedError

    dataset.save_to_disk(os.path.join(output_directory, "tokenized_data"))

    if params.model_name[:2].lower() == 't5':
        TrainingArgsClass = Seq2SeqTrainingArguments
        TrainerClass = CustomSeq2SeqTrainer if params.constrain_token else Seq2SeqTrainer
    else:
        NotImplementedError
    
    if params.model_name[:2].lower() == 't5':
        if params.constrain_token:
            training_args = TrainingArgsClass(
                output_dir=output_directory,
                learning_rate=params.lr,
                weight_decay=params.l2,
                num_train_epochs=params.num_epochs,
                save_only_model=True,
                predict_with_generate=True,
                evaluation_strategy="epoch",
                logging_dir=f"{output_directory}/logs",  
                save_strategy="epoch",
                save_total_limit = 3,
                warmup_ratio = 0.01,
                lr_scheduler_type = SchedulerType.COSINE_WITH_RESTARTS,
                max_grad_norm = 1,
                per_device_train_batch_size = params.train_batch_size,
                per_device_eval_batch_size = params.eval_batch_size,
                logging_steps=10,
                push_to_hub=False)
            
            trainer = TrainerClass(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["test"],
                tokenizer=tokenizer,
                data_collator=data_collator,
                allowed_token_ids=allowed_token_ids,
                pad_token_id=tokenizer.pad_token_id
            )

        else:
            training_args = TrainingArgsClass(
                output_dir=output_directory,
                learning_rate=params.lr,
                weight_decay=params.l2,
                num_train_epochs=params.num_epochs,
                save_only_model=True,
                predict_with_generate=True,
                evaluation_strategy="epoch",
                logging_dir=f"{output_directory}/logs",  
                save_strategy="epoch",
                save_total_limit = 3,
                save_steps = 1,
                per_device_train_batch_size = params.train_batch_size,
                per_device_eval_batch_size = params.eval_batch_size,
                logging_steps=10,
                push_to_hub=False,
            )
            trainer = TrainerClass(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["test"],
                tokenizer=tokenizer,
                data_collator=data_collator
            )

    trainer.train()

    with open(os.path.join(output_directory, "logs/logs.txt"), 'a') as file:
        for val in trainer.state.log_history:
            log_entry = ' '.join('%s: %s' % (k, str(v)) for k, v in val.items())
            file.write(log_entry + "\n")
        file.write("finished\n")

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='TransE model')
    parser.add_argument("--num_epochs", "-ne", type=int, default=3,
                        help="Learning rate of the optimizer")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate of the optimizer")
    parser.add_argument("--l2", type=float, default=1e-3,
                        help="Regularization constant for GNN weights")
    parser.add_argument('--seed', default=0,
                        type=int, help='Seed for randomization')

    parser.add_argument('--train_batch_size', default=64,
                        type=int, help='train_batch_size')
    parser.add_argument('--eval_batch_size',default=64,
                        type=int, help='eval_batch_size')
    parser.add_argument('--data_path', default="FB15k-237",
                        type=str, help='data_path')

    parser.add_argument('--model_name', default="T5-large",
                        type=str, help='model_name')
    parser.add_argument('--exp_name', default="T5-base",
                        type=str, help='model_name')


    parser.add_argument('--constrain_token', action='store_true', default=True,
                        help='constrain token')


    parser.add_argument('--max_hop', default=3,
                        type=int, help='train_batch_size')
    parser.add_argument('--use_ent', action='store_true',
                    help='constrain token')

    parser.add_argument('--train_data_name', default="query_extraction_data_all_fin",
                        type=str, help='data_path')
    parser.add_argument('--test_data_name', default="query_extraction_data_all_fin",
                        type=str, help='data_path')

    params = parser.parse_args()
    if params.model_name[:2].lower()=='t5':
        params.max_length = 256
    else:
        raise NotImplementedError

    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)

    output_directory = os.path.join(f"../save_path/{params.data_path}", params.exp_name)

    if not os.path.exists("../save_path"):
        os.mkdir("../save_path")
    if not os.path.exists(f"../save_path/{params.data_path}"):
        os.mkdir(f"../save_path/{params.data_path}")
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    



    logging.basicConfig(level=logging.INFO)
    if not os.path.exists(os.path.join(output_directory, 'logs')):
            os.makedirs(os.path.join(output_directory, 'logs'))
    file_handler = logging.FileHandler(os.path.join(output_directory, f"logs/logs.txt"), mode='w')

    logging.root.addHandler(file_handler)
    logging.info('============ Initialized logger ============')
    logging.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items())))


    main(params)

from transformers import Seq2SeqTrainer, LogitsProcessor
from datasets import Dataset
import torch


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, allowed_token_ids=None, pad_token_id = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.allowed_token_ids = allowed_token_ids
        self.pad_token_id = pad_token_id

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if self.allowed_token_ids is not None:
            mask = torch.full(logits.shape, float('-inf')).to(logits.device)
            mask[..., self.allowed_token_ids] = 0
            
            logits = logits + mask


        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=self.pad_token_id)
        return (loss, outputs) if return_outputs else loss


class RestrictVocabLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids, scores):
        mask = torch.full(scores.shape, float('-inf')).to(scores.device)
        mask[..., self.allowed_token_ids] = 0
        scores += mask
        scores[..., self.allowed_token_ids]
        return scores
    
def batchify(inputs, batch_size):
    for i in range(0, len(inputs['input_ids']), batch_size):
        yield {key: val[i:i + batch_size] for key, val in inputs.items()}



def get_preprocessed_query_dataset_all_in_one_w_neg(query_path_set):


    query_path_set_processed = {'query': query_path_set['query'].tolist(), 'pos_rel': [[v for v in val] for val in query_path_set['positive_path'].tolist()], 'neg_rel': [[v for v in val] for val in query_path_set['negative_path'].tolist()]}

    return Dataset.from_dict(query_path_set_processed)





def calculate_max_length_query_extraction(dataset, tokenizer, mode, constrain_token = False):
    if mode not in ['input', 'label']:
        raise ValueError("Mode should be either 'input' or 'label'")

    max_length = 0

    if mode == 'input':
        texts = [f"Please extract the relations that exist in the knowledge graph from the following question: {example['query']}"
                 for key_name in dataset for example in dataset[key_name]]
        encodings = tokenizer(texts, add_special_tokens=True, padding = 'longest')
        max_length = len(encodings['input_ids'][0])

    elif mode == 'label':
        if constrain_token:
            texts = [f"<PATH>{'<SEP>'.join(f'<{p}>' for p in example['rel'])}</PATH>"
                    for key_name in dataset for example in dataset[key_name]]
        else:
            texts = [f"<PATH>{'<SEP>'.join(f'{p}' for p in example['rel'])}</PATH>"
                    for key_name in dataset for example in dataset[key_name]]

        encodings = tokenizer(texts, add_special_tokens=True, padding = 'longest')
        max_length = len(encodings['input_ids'][0])

    return max_length




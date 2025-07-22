# Synthetic Data Generation Process

## 1. Prepare KG data to /data
The list of KGs we use is following:

1. FB15k-237
2. UMLS
3. CoDEx-s

## 2. Query Generation
We use the same query generation process with the several CQA models (Query2Box, Query2Triple ...), and utilize the query generation code from [snap-stanford/KGReasoning](https://github.com/snap-stanford/KGReasoning)

For example, if we want to generate CoDEx datset.

1. First, create the indexed file.    
   We don't need to apply this process to FB15k-237 since it is already indexed.
```
python create_queries.py --dataset CODEX --index_only
```
1. Second, generate queries and answers.
```
python create_queries.py --dataset CODEX --gen_train --save_name
```

## 3. Convert triple query to FOL (First-Order Logic) form
```
python create_fol.py --dataset CODEX --gen_train
```

## 4. Convert FOL query to Natural Language Question
```
python create_nl.py --dataset CODEX --gen_train
```
## 5. Final Dataset
The final output data is located in data_generation/data/output/CODEX/synthetic_drop_dup.parquet
```
python create_training_data.py --dataset CODEX --gen_train
```

You can simply run generate.sh to process all datasets or select a specific dataset.
    
    ./generate.sh

    ./generate_FB15k-237.sh
    ./generate_CODEX.sh
    ./generate_UMLS
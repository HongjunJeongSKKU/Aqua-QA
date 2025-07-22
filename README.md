Aqua-QA
=============
This repository is the official implementation for the paper "Enhancing Complex Reasoning in Knowledge Graph Question Answering Through Query Graph Approximation".

This repository is based on [Q2T](https://github.com/YaooXu/Q2T) and [BetaE](https://github.com/snap-stanford/KGReasoning). 



## Synthetic Data Generation Process

<details>
    <summary>Check</summary>
    
    cd data_generation

   <h3>1. Prepare KG data to data</h3>
   
    The list of KGs we use is following:

    1. FB15k-237
    2. UMLS
    3. CoDEx-s

   <h3>2. Query Generation</h3>
-To generate additional synthetic data, please follow this process.-

We utilize the query generation code from <a href="https://github.com/snap-stanford/KGReasoning">snap-stanford/KGReasoning</a>

You can simply run generate.sh to process all datasets or select a specific dataset.

    ./generate.sh
    ./generate_FB15k-237.sh
    ./generate_CODEX.sh
    ./generate_UMLS

If you want to follow the generation process step by step, you can follow the steps below. The code is an example for the CoDEx dataset.

1. First, create the indexed file. We don't need to apply this process to FB15k-237 since it is already indexed.
    ```
    python create_queries.py --dataset CODEX --index_only
    ```
2. Second, generate queries and answers.
    ```
    python create_queries.py --dataset CODEX --gen_train --save_name
    ```

3. Convert triple query to FOL (First-Order Logic) form
    ```
    python create_fol.py --dataset CODEX --gen_train
    ```

4. Convert FOL query to Natural Language Question
    ```
    python create_nl.py --dataset CODEX --gen_train
    ```
5. Final Dataset
    The final output data is located in data_generation/data/output/CODEX/synthetic_drop_dup.parquet
    ```
    python create_training_data.py --dataset CODEX --gen_train
    ```
</details>

## Query decomposition module
<details>
    <summary>Check</summary>
This code will generate the sub questions from the original question.
The final output data is located in data/CODEX/train-subq_drop_dup.parquet
You can simply run generate.sh to process all datasets or select a specific dataset.

    cd query_decomposition
    ./generate.sh
    ./generate_FB15k-237.sh
    ./generate_CODEX.sh
    ./generate_UMLS

The generated sub-questions should be added to the row of the original question as a column named 'sub' in the form of a list.

When the data_type is one of train/valid/test, the data with sub-questions added as the sub column should be saved as data_type-subq_drop_dup.parquet.

For example, if sub-questions were generated for data/FB15k-237/train-drop_dup.parquet, the final data should be saved as data/FB15k-237/train-subq_drop_dup.parquet.


</details>

## Pretraining of KGE model
<details>
<summary>Check</summary>
Please refer <a href="https://github.com/YaooXu/Q2T/tree/master/ssl-relation-prediction">ssl-relation-prediction</a>.

The saved model should be located at src_final/align_kg_w_lm_new_neg/src/KGE/models.

</details>

## Pretraining of Relation extraction module
<details>
<summary>Check</summary>
Query decomposition should be performed before pretraining of Rleatoin extractoin module.

For pretraining of Relation extraction module, run the following commands:

    # The $data can be FB15k-237, CODEX or UMLS.
    cp data/$data/synthetic_drop_dup.parquet relation_extraction/data/$data
    cp data/$data/ent2id.pkl relation_extraction/data/$data
    cp data/$data/rel2id.pkl relation_extraction/data/$data
    cp data/$data/train-subq_drop_dup.relation_extraction/data/$data
    cp data/$data/valid-subq_drop_dup.relation_extraction/data/$data
    cp data/$data/test-subq_drop_dup.relation_extraction/data/$data
    
    cd relation_extraction/src

    # For FB15k-237 dataset
    ./rel_fb237.sh

    # For UMLS dataset
    ./rel_umls.sh

    # For CoDEx dataset
    ./rel_codex.sh

</details>

## Pretraining of Align module
<details>
<summary>Check</summary>

For pretraining of Align module, run the following commands:

    # The $data can be FB15k-237, CODEX or UMLS.
    cp data/$data/synthetic_drop_dup.parquet src_final/align_kg_w_lm_new_neg/data/$data
    cp data/$data/ent2id.pkl src_final/align_kg_w_lm_new_neg/data/$data
    cp data/$data/rel2id.pkl src_final/align_kg_w_lm_new_neg/data/$data

    cd src_final/align_kg_w_lm_new_neg/src
    
    # For FB15k-237 dataset
    ./align_fb237.sh
    
    #For UMLS dataset
    ./align_umls.sh

    #For CoDEx dataset
    ./align_codex.sh
</details>

## Training of Reasoner
<details>
<summary>Check</summary>

All of the above tasks must be completed in order to train Reasoner.

To prepare the data needed to train Reasoner, run the following command:

    # The $rel_added_data is path_from_$exp_name_$exp_checkpoint.
    cp -r relation_extraction/data/$data/$rel_added_data data/$data

Then, to reproduce the final results of CFKGQA for FB15k-237, CoDEx or UMLS, simply run the following commands:

    cd src_final
    #For FB15k-237 dataset
    ./run_fb237.sh

    #For UMLS dataset
    ./run_umls.sh

    #For CoDEx dataset
    ./run_codex.sh
</details>

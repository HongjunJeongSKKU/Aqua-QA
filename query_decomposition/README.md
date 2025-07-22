# Sub Question Generation
This code will generate the sub questions from the original question.
The final output data is located in data/CODEX/train-subq_drop_dup.parquet
## Generate Sub Question
```
python create_sub_nl.py --dataset CODEX --gen_train
python create_sub_nl.py --dataset CODEX --gen_valid
python create_sub_nl.py --dataset CODEX --gen_test
```
You can simply run generate.sh to process all datasets or select a specific dataset.
    
    ./generate.sh

    ./generate_FB15k-237.sh
    ./generate_CODEX.sh
    ./generate_UMLS
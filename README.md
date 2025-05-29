Aqua-QA
=============
This repository contains official implementation of paper "Enhancing Complex Reasoning in Knowledge Graph Question Answering Through Query Graph Approximation".

This repository is based on [Q2T](https://github.com/YaooXu/Q2T). 

For pretraining of Align module, run the following commands:

    cd src_final/align_kg_w_lm_new_neg/src

    ./align_umls.sh

    ./align_codex.sh
Then, to reproduce the final results of CFKGQA for CoDEx and UMLS, simply run the following commands:

    cd src_final

    ./run_umls.sh

    ./run_codex.sh

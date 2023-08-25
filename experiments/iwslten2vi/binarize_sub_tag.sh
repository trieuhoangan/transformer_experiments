#!/bin/bash

# Created by:         Emanuele Bugliarello (@e-bug)
# Date created:       9/4/2019
# Date last modified: 9/4/2019

src=en
tgt=vi
PROJ=$HOME/pascal
INPUT=$PROJ/data/iwslten2vi/sub_tags
OUTPUT=$INPUT/prime_en2vi_sub_tags

# activate environment
# source activate pascal

# Binarize the dataset:
# cd $HOME/fairseq/fairseq_cli
# python preprocess.py \
fairseq-preprocess --source-lang $src \
	--target-lang $tgt \
	--trainpref $INPUT/prime_indices_train \
	--validpref $INPUT/prime_indices_valid \
	--testpref $INPUT/prime_indices_test \
	--destdir $OUTPUT \
	--workers 32 \
	--only-source
	
# deactivate environment
# conda deactivate

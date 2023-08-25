#!/bin/bash

# Created by:         Emanuele Bugliarello (@e-bug)
# Date created:       9/4/2019
# Date last modified: 9/4/2019

src=vi
tgt=en
PROJ=$HOME/pascal
INPUT=$PROJ/data/iwslten2vi/pos_tags
OUTPUT=$INPUT/iwslt${src}2${tgt}_filtered_002

# activate environment
# source activate pascal

# Binarize the dataset:
# cd $HOME/fairseq/fairseq_cli
# python preprocess.py \
fairseq-preprocess --source-lang $src \
	--target-lang $tgt \
	--trainpref $INPUT/new_train_002 \
	--validpref $INPUT/new_valid_002 \
	--testpref $INPUT/new_test_002 \
	--destdir $OUTPUT \
	--workers 32 \
	--only-source
	
# deactivate environment
# conda deactivate

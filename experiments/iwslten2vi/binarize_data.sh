#!/bin/bash

# Created by:         Emanuele Bugliarello (@e-bug)
# Date created:       9/4/2019
# Date last modified: 9/4/2019

src=en
tgt=vi
PROJ=$HOME/pascal
INPUT=$PROJ/data/iwslten2vi/corpus_no_vi_bpe
OUTPUT=$INPUT/iwslt${src}2${tgt}_no_vi_bpe

# activate environment
# source activate pascal

# Binarize the dataset:
# cd $HOME/fairseq/fairseq_cli
# python preprocess.py \
fairseq-preprocess --source-lang $src \
	--target-lang $tgt \
	--trainpref $INPUT/clear_train.tok.bpe.10000 \
	--validpref $INPUT/clear_valid.tok.bpe.10000 \
	--testpref $INPUT/clear_test.tok.bpe.10000 \
	--destdir $OUTPUT \
	--workers 32 \
	--joined-dictionary \
	--thresholdtgt 0 --thresholdsrc 0 \
# deactivate environment
# conda deactivate

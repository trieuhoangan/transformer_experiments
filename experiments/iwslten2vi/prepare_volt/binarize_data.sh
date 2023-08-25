#!/bin/bash

# Created by:         Emanuele Bugliarello (@e-bug)
# Date created:       9/4/2019
# Date last modified: 9/4/2019

src=en
tgt=vi
PROJ=$HOME/pascal
INPUT=$PROJ/data/iwslten2vi/VOLT_en2vi
OUTPUT=$INPUT/iwslt_VOLT_${src}2${tgt}

# activate environment
# source activate pascal

# Binarize the dataset:
# cd $HOME/fairseq/fairseq_cli
# python preprocess.py \
fairseq-preprocess --source-lang $src \
	--target-lang $tgt \
	--trainpref $INPUT/train \
	--validpref $INPUT/valid \
	--testpref $INPUT/test \
	--destdir $OUTPUT \
	--workers 32 \
	--joined-dictionary \
	--thresholdtgt 0 --thresholdsrc 0 \
# deactivate environment
# conda deactivate

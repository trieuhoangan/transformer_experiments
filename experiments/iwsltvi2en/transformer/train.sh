#!/bin/bash

# Created by:         Emanuele Bugliarello (@e-bug)
# Date created:       9/4/2019
# Date last modified: 9/4/2019

DATADIR="$HOME/pascal/data/iwslten2vi/corpus/iwsltvi2en10k"
CKPTS="$HOME/checkpoint/dep_transformer/iwsltvi2en/transformer"
# export CUDA_VISIBLE_DEVICES=0,1,2,3
CUDA_VISIBLE_DEVICES=0
# NODES=$1
# GPUS=$1
# WORLD_SIZE=$[NODES * GPUS]
# MASTER=$(head -n 1 ./hosts)
# hosts=`cat ./hosts`
# h=0
# n=0

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES fairseq-train $DATADIR \
        --save-dir $CKPTS \
        --arch transformer_iwslt_de_en  \
        --dropout 0.3 \
        --share-all-embeddings \
        --optimizer adam \
        -s vi -t en \
        --adam-betas '(0.9,0.98)' \
        --clip-norm 0.0 \
        --lr-scheduler inverse_sqrt \
        --tensorboard-logdir $CKPTS \
        --keep-last-epochs 5\
        --warmup-updates 6000 \
        --lr 7e-4 \
        --weight-decay 0.0001 \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --max-tokens 4096 \
        --max-epoch 100 \
	--no-progress-bar \
	--log-format json \
	--log-interval 100 \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-detok moses \
        --eval-bleu-remove-bpe \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric




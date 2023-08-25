#!/bin/bash

# Created by:         Emanuele Bugliarello (@e-bug)
# Date created:       9/4/2019
# Date last modified: 9/4/2019

data_bin="$HOME/pascal/data/iwslten2vi/corpus/fairseq0102iwslten2vi10k"
TAGSDIR="$HOME/pascal/data/iwslten2vi/tags_root/fairseq0102iwslten2vi"
SUBTAGSDIR="$HOME/pascal/data/iwslten2vi/sub_tags/reordered_en2vi_sub_tags"
CKPTS="$HOME/checkpoint/dep_transformer/iwslten2vi/change_embed_method"
# DICTIONARYPATH="$HOME/pascal/data/iwslten2vi/pairs/dictionary2.en"
CUDA_VISIBLE_DEVICES=0
user_dir="$HOME/pascal/NMT/LISA_Change_embed_method/"

h=0
n=0
mkdir -p $CKPTS
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES CUDA_LAUNCH_BLOCKING=1 fairseq-train $data_bin \
        --user-dir $user_dir --criterion lisa_cross_entropy --task tags_translation --arch lisa_transformer_wmt_en_de \
        --tags-data $TAGSDIR \
        --sub-tags $SUBTAGSDIR \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        --lr 7e-4 --lr-scheduler inverse_sqrt --warmup-updates 6000 \
        --ddp-backend no_c10d \
        -s en -t vi --save-dir $CKPTS \
        --encoder-lisa-layer 0 \
        --parse-penalty 0.5 \
        --sub-penalty 0.3 \
        --dropout 0.3 \
        --tensorboard-logdir $CKPTS \
        --parent-ignoring 0.3 \
        --share-all-embeddings \
        --relu-dropout 0.1 --weight-decay 0.0001 --attention-dropout 0.1 \
        --label-smoothing 0.1 \
        --max-tokens 4096 --max-epoch 100 --seed 1\
        --no-progress-bar \
        --log-format json \
        --log-interval 100 \
        --keep-last-epochs 5 \
        --reset-dataloader \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-detok moses \
        --eval-bleu-remove-bpe \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
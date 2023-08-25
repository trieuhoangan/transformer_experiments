#!/bin/bash

# Created by:         Emanuele Bugliarello (@e-bug)
# Date created:       9/4/2019
# Date last modified: 9/4/2019
SRC_LANG=de
TGT_LANG=en
data_bin="$HOME/pascal/data/iwsltdeen/corpus3/iwsltde2en0102"
CKPTS="$HOME/checkpoint/dep_transformer/iwsltde2en/trans_add_tfidf"
DICTIONARYPATH="$HOME/pascal/data/iwsltdeen/pairs/tfidf.de"
CUDA_VISIBLE_DEVICES=0
user_dir="$HOME/pascal/NMT/transformer_add_frequency_refined/"

h=0
n=0
mkdir -p $CKPTS
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES CUDA_LAUNCH_BLOCKING=1 fairseq-train $data_bin \
        --user-dir $user_dir --criterion label_smoothed_cross_entropy --task translation --arch lisa_transformer_wmt_en_de \
        --pair-file $DICTIONARYPATH \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        --lr 7e-4 --lr-scheduler inverse_sqrt --warmup-updates 6000 \
        --ddp-backend no_c10d \
        -s $SRC_LANG -t $TGT_LANG --save-dir $CKPTS \
        --encoder-frequency-layer 0 \
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
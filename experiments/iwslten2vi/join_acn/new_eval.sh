#!/bin/bash

# Created by:         Emanuele Bugliarello (@e-bug)
# Date created:       9/4/2019
# Date last modified: 9/4/2019

export CUDA_VISIBLE_DEVICES=0
SRC_LANG=en
TGT_LANG=de
PROJ_PATH=$HOME/pascal/experiments/iwslt${SRC_LANG}2${TGT_LANG}
DATA_PATH="$HOME/pascal/data/iwsltdeen/corpus/tmp/iwslt${SRC_LANG}2${TGT_LANG}0102"
TEST_FILE="$DATA_PATH/test.$TGT_LANG.depbe"
TAGS_PATH="$HOME/pascal/data/iwsltdeen/tags_root/iwslt${SRC_LANG}2${TGT_LANG}0102"
CKPT_PATH="$HOME/checkpoint/dep_transformer/iwslt${SRC_LANG}2${TGT_LANG}/join_model"
user_dir="$HOME/pascal/code/join_model/"
MODEL_DIR=$PROJ_PATH/join_acn
OUTPUT_FN=$MODEL_DIR/res.txt
MOSES_LIB="$HOME/pascal/tools/mosesdecoder"


fairseq-generate ${DATA_PATH}  \
            --user-dir $user_dir \
            --tags-data $TAGS_PATH \
            --source-lang $SRC_LANG --target-lang $TGT_LANG \
            --criterion join_label_smoothed_cross_entropy --task acn_tags_translation \
            --path ${CKPT_PATH}/checkpoint_best.pt  --beam 5 --remove-bpe > ${MODEL_DIR}/out.avg.log 
    cat ${MODEL_DIR}/out.avg.log  | grep '^H' | sed 's/^H\-//g' | sort -t ' ' -k1,1 -n | cut -f 3- > ${MODEL_DIR}/generated.result
    cat ${MODEL_DIR}/out.avg.log  | grep '^T' | sed 's/^T\-//g' | sort -t ' ' -k1,1 -n | cut -f 2- > ${TEST_FILE} 

cat ${MODEL_DIR}/generated.result | ${MOSES_LIB}/scripts/generic/multi-bleu.perl ${TEST_FILE} > ${MODEL_DIR}/log_avg_multi-bleu.log
cat ${MODEL_DIR}/log_avg_multi-bleu.log
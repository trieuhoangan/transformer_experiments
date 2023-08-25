RUN_PREDICT=${1:-true}


SRC_LANG=vi
TGT_LANG=en
PATH_DATA="$HOME/checkpoint/dep_transformer/iwsltvi2en/true_frequency"
PATH_DATA_BIN="$HOME/pascal/data/iwslten2vi/corpus/iwsltvi2en10k"
MOSES_LIB="$HOME/pascal/scripts/tools/mosesdecoder"
TAGS_PATH="$HOME/pascal/data/iwslten2vi/tags_root/fairseq0102iwsltvi2en3"
SUBTAGSDIR="$HOME/pascal/data/iwslten2vi/sub_tags/sub_tags2"
TEST_FILE="$PATH_DATA/test.$TGT_LANG.debpe"
USER_DIR="$HOME/pascal/NMT/LISA_add_frequency/"
PROJ_PATH=$HOME/pascal/experiments/iwslt${SRC_LANG}2${TGT_LANG}
AVE_SCRIPT_PATH=$HOME/pascal/NMT/script
 

# # run avg last 5 checkpoint 
# if $RUN_PREDICT ; then
python $AVE_SCRIPT_PATH/avg_last_checkpoint.py --inputs ${PATH_DATA} --num-epoch-checkpoints 5 --output ${PATH_DATA}/averaged.pt
fairseq-generate ${PATH_DATA_BIN} \
        --tags-data $TAGS_PATH \
        --sub-tags $SUBTAGSDIR \
        --source-lang $SRC_LANG --target-lang $TGT_LANG \
        --task tags_translation --user-dir $USER_DIR \
        --criterion lisa_cross_entropy \
        --path ${PATH_DATA}/averaged.pt  --beam 5 --remove-bpe   > ${PATH_DATA}/generated.result.raw.log 
cat ${PATH_DATA}/generated.result.raw.log | grep '^H' | sed 's/^H\-//g' | sort -t ' ' -k1,1 -n | cut -f 3- > ${PATH_DATA}/generated.result
cat ${PATH_DATA}/generated.result.raw.log | grep '^T' | sed 's/^T\-//g' | sort -t ' ' -k1,1 -n | cut -f 2- > ${TEST_FILE} 
# fi
cat ${PATH_DATA}/generated.result | ${MOSES_LIB}/scripts/generic/multi-bleu.perl ${TEST_FILE} > ${PATH_DATA}/log_avg_multi-bleu.log
cat  ${PATH_DATA}/log_avg_multi-bleu.log
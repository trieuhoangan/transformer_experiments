PROJ_DIR="$HOME/pascal"
CORPUS_DIR="$PROJ_DIR/data/iwslten2vi/corpus"
BPE_CODE="$CORPUS_DIR/bpe.10000"

RAW_TRAIN_FILE="$CORPUS_DIR/clear_train"
RAW_TEST_FILE="$CORPUS_DIR/clear_train"
RAW_VALID_FILE="$CORPUS_DIR/clear_train"
BPE_TRAIN_FILE="$CORPUS_DIR/clear_train.tok.bpe.10000"
BPE_VALID_FILE="$CORPUS_DIR/clear_valid.tok.bpe.10000"
BPE_TEST_FILE="$CORPUS_DIR/clear_test.tok.bpe.10000"
VOLT_DIR="$PROJ_DIR/data/iwslten2vi/VOLT_vi2en"
VOLT_TRAIN_OUTPUT="$VOLT_DIR/train"
VOLT_VALID_OUTPUT="$VOLT_DIR/valid"
VOLT_TEST_OUTPUT="$VOLT_DIR/test"
VOCAB_FILE="$VOLT_DIR/vocab"
SIZE_FILE="$VOLT_DIR/size"
TMP_TEXT="$VOLT_DIR/tmp"
src=vi
tgt=en
mkdir $VOLT_DIR
cat $BPE_TRAIN_FILE.$src $BPE_VALID_FILE.$src > $TMP_TEXT.$src
cat $BPE_TRAIN_FILE.$tgt $BPE_VALID_FILE.$tgt > $TMP_TEXT.$tgt

# touch $VOLT_TEST_OUTPUT.$tgt
python3 ../../../VOLT/ot_run.py --source_file $TMP_TEXT.$src --target_file $TMP_TEXT.$tgt \
        --token_candidate_file $BPE_CODE \
        --vocab_file $VOCAB_FILE --max_number 10000 --interval 1000  --loop_in_ot 500 --tokenizer subword-nmt --size_file $SIZE_FILE


# echo "#version: 0.2" > $VOLT_DIR/vocab.seg # add version info
# echo $VOCAB_FILE  >> $VOLT_DIR/vocab.seg
subword-nmt apply-bpe -c $VOCAB_FILE < $RAW_TEST_FILE.$src > $VOLT_TEST_OUTPUT.$src
subword-nmt apply-bpe -c $VOCAB_FILE < $RAW_TEST_FILE.$tgt > $VOLT_TEST_OUTPUT.$tgt #optional if your task does not contain target texts

subword-nmt apply-bpe -c $VOCAB_FILE < $RAW_TRAIN_FILE.$src > $VOLT_TRAIN_OUTPUT.$src
subword-nmt apply-bpe -c $VOCAB_FILE < $RAW_TRAIN_FILE.$tgt > $VOLT_TRAIN_OUTPUT.$tgt #optional if your task does not contain target texts

subword-nmt apply-bpe -c $VOCAB_FILE < $RAW_VALID_FILE.$src > $VOLT_VALID_OUTPUT.$src
subword-nmt apply-bpe -c $VOCAB_FILE < $RAW_VALID_FILE.$tgt > $VOLT_VALID_OUTPUT.$tgt #optional if your task does not contain target texts
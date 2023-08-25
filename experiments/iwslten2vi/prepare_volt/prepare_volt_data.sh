PROJ_DIR="$HOME/pascal"
CORPUS_DIR="$PROJ_DIR/data/iwslten2vi/corpus"
size=30000
src=en
tgt=vi
RAW_TRAIN_FILE="$CORPUS_DIR/clear_train"
RAW_VALID_FILE="$CORPUS_DIR/clear_valid"
RAW_TEST_FILE="$CORPUS_DIR/clear_test"

VOLT_DIR="$PROJ_DIR/data/iwslten2vi/VOLT_en2vi"
BPE_CODE="$VOLT_DIR/bpe.$size"
# BPE_CODE="$VOLT_DIR/bpe.$size.$src"
# BPE_CODE="$VOLT_DIR/bpe.$size.$tgt"
BPE_TRAIN_FILE="$VOLT_DIR/clear_train.bpe.$size"
BPE_VALID_FILE="$VOLT_DIR/clear_valid.bpe.$size"
BPE_TEST_FILE="$VOLT_DIR/clear_test.bpe.$size"
VOLT_TRAIN_OUTPUT="$VOLT_DIR/train"
VOLT_VALID_OUTPUT="$VOLT_DIR/valid"
VOLT_TEST_OUTPUT="$VOLT_DIR/test"
VOCAB_FILE="$VOLT_DIR/vocab"
SIZE_FILE="$VOLT_DIR/size.txt"
TMP_TEXT="$VOLT_DIR/tmp.txt"

mkdir $VOLT_DIR
cat $RAW_TRAIN_FILE.$src $RAW_VALID_FILE.$src > training_data
cat $RAW_TRAIN_FILE.$tgt $RAW_VALID_FILE.$tgt >> training_data




# re apply BPE on raw data
subword-nmt learn-bpe -s $size  < training_data > $BPE_CODE
# subword-nmt learn-bpe -s $size  < $TMP_TEXT.$tgt > $BPE_CODE
subword-nmt apply-bpe -c $BPE_CODE < $RAW_TRAIN_FILE.$src > $BPE_TRAIN_FILE.$src
subword-nmt apply-bpe -c $BPE_CODE < $RAW_TRAIN_FILE.$tgt > $BPE_TRAIN_FILE.$tgt 

subword-nmt apply-bpe -c $BPE_CODE < $RAW_VALID_FILE.$src > $BPE_VALID_FILE.$src
subword-nmt apply-bpe -c $BPE_CODE < $RAW_VALID_FILE.$tgt > $BPE_VALID_FILE.$tgt 

# subword-nmt apply-bpe -c $BPE_CODE < $RAW_TEST_FILE.$src > $BPE_TEST_FILE.$src
# subword-nmt apply-bpe -c $BPE_CODE < $RAW_TEST_FILE.$tgt > $BPE_TEST_FILE.$tgt 



# Source files hadling
python3 ../../../VOLT/ot_run.py --source_file $BPE_TRAIN_FILE.$src --target_file $BPE_TRAIN_FILE.$tgt \
        --token_candidate_file $BPE_CODE \
        --vocab_file $VOCAB_FILE --max_number 10000 --interval 1000  --loop_in_ot 500 --tokenizer subword-nmt --size_file $SIZE_FILE

# python3 ../../../VOLT/ot_run.py --source_file $BPE_VALID_FILE.$src --target_file $BPE_VALID_FILE.$tgt \
#         --token_candidate_file $BPE_CODE \
#         --vocab_file $VOCAB_FILE --max_number 10000 --interval 1000  --loop_in_ot 500 --tokenizer subword-nmt --size_file $SIZE_FILE

# python3 ../../../VOLT/ot_run.py --source_file $BPE_VALID_FILE.$src \
#         --token_candidate_file $BPE_CODE \
#         --vocab_file $VOCAB_FILE --max_number 10000 --interval 1000  --loop_in_ot 500 --tokenizer subword-nmt --size_file $SIZE_FILE.$src

# target files hadling
# python3 ../../../VOLT/ot_run.py --source_file $BPE_TRAIN_FILE.$tgt \
#         --token_candidate_file $BPE_CODE \
#         --vocab_file $VOCAB_FILE --max_number 10000 --interval 1000  --loop_in_ot 500 --tokenizer subword-nmt --size_file $SIZE_FILE.$tgt

# python3 ../../../VOLT/ot_run.py --source_file $BPE_VALID_FILE.$tgt \
#         --token_candidate_file $BPE_CODE \
#         --vocab_file $VOCAB_FILE --max_number 10000 --interval 1000  --loop_in_ot 500 --tokenizer subword-nmt --size_file $SIZE_FILE.$tgt

# echo "#version: 0.2" > $VOLT_DIR/vocab.seg # add version info
# echo $VOCAB_FILE  >> $VOLT_DIR/vocab.seg
subword-nmt apply-bpe -c $VOCAB_FILE < $RAW_TEST_FILE.$src > $VOLT_TEST_OUTPUT.$src
subword-nmt apply-bpe -c $VOCAB_FILE < $RAW_TEST_FILE.$tgt > $VOLT_TEST_OUTPUT.$tgt #optional if your task does not contain target texts

subword-nmt apply-bpe -c $VOCAB_FILE < $RAW_TRAIN_FILE.$src > $VOLT_TRAIN_OUTPUT.$src
subword-nmt apply-bpe -c $VOCAB_FILE < $RAW_TRAIN_FILE.$tgt > $VOLT_TRAIN_OUTPUT.$tgt #optional if your task does not contain target texts

subword-nmt apply-bpe -c $VOCAB_FILE < $RAW_VALID_FILE.$src > $VOLT_VALID_OUTPUT.$src
subword-nmt apply-bpe -c $VOCAB_FILE < $RAW_VALID_FILE.$tgt > $VOLT_VALID_OUTPUT.$tgt #optional if your task does not contain target texts
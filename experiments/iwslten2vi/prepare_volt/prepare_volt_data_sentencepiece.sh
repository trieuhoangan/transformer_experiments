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
cat $RAW_TRAIN_FILE.$src $RAW_VALID_FILE.$src > $TMP_TEXT
cat $RAW_TRAIN_FILE.$tgt $RAW_VALID_FILE.$tgt >> $TMP_TEXT

cd /home/s2320037/pascal/VOLT/examples
mkdir spmout
python3 /home/s2320037/pascal/VOLT/examples/spm/spm_train.py --input=$TMP_TEXT --model_prefix=spm --vocab_size=$size --character_coverage=1.0 --model_type=bpe
sed -i 's/\t/ /g' spm.vocab
python3 spm/spm_encoder.py --model spm.model --inputs $RAW_TRAIN_FILE.$src --outputs spmout/train.$src --output_format piece
python3 spm/spm_encoder.py --model spm.model --inputs $RAW_TRAIN_FILE.$tgt --outputs spmout/train.$tgt --output_format piece

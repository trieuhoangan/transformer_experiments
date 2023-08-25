PROJ_DIR="$HOME/pascal"
OUTPUT_DIR="$PROJ_DIR/data/iwslten2vi/corpus"
MOSES_DIR="$PROJ_DIR/tools/mosesdecoder"
SCRIPTS_DIR="$PROJ_DIR/scripts"


# Tokenize data
# for f in ${OUTPUT_DIR}/*.vi; do
#   echo "Tokenizing $f..."
#   ${MOSES_DIR}/scripts/tokenizer/tokenizer.perl -q -l vi -threads 8 < $f > ${f%.*}.mtok.vi
# done
# for f in ${OUTPUT_DIR}/*.en; do
#   echo "Tokenizing $f..."
#   ${MOSES_DIR}/scripts/tokenizer/tokenizer.perl -q -l en -threads 8 < $f > ${f%.*}.mtok.en
# done

# Clean train corpus
# f=${OUTPUT_DIR}/train.tok.en
# f=${OUTPUT_DIR}/train.en
# fbase=${f%.*}
# echo "Cleaning ${fbase}..."
# ${MOSES_DIR}/scripts/training/clean-corpus-n.perl $fbase vi en "${fbase}.clean" 1 80

CoreNLP tokenization
for f in ${OUTPUT_DIR}/clear_train.en ${OUTPUT_DIR}/clear_valid.en ${OUTPUT_DIR}/clear_test.en; do
  fbase=${f%.*}
  echo "CoreNLP tokenizing ${fbase}..."
  python ${SCRIPTS_DIR}/custom_corenlp_tok.py $fbase
done

for merge_ops in 10000; do
  echo "Learning BPE with merge_ops=${merge_ops}. This may take a while..."
  cat "${OUTPUT_DIR}/train.vi" "${OUTPUT_DIR}/clear_train.tok.en" | \
    subword-nmt learn-bpe -s $merge_ops > "${OUTPUT_DIR}/bpe.${merge_ops}"

#   echo "Apply BPE with merge_ops=${merge_ops} to tokenized files..."
  # for lang in en vi; do
  #   for f in ${OUTPUT_DIR}/*.tok.${lang} ${OUTPUT_DIR}/train.tok.clean.${lang}; do
  #     outfile="${f%.*}.bpe.${merge_ops}.${lang}"
  #     subword-nmt apply-bpe -c "${OUTPUT_DIR}/bpe.${merge_ops}" < $f > "${outfile}"
  #   done
  # done
  # for f in ${OUTPUT_DIR}/train.tok.clean.tok.en ${OUTPUT_DIR}/train.tok.clean.vi; do
  #   outfile="${f%.*}.bpe.${merge_ops}.en"
  #   subword-nmt apply-bpe -c "${OUTPUT_DIR}/bpe.${merge_ops}" < $f > "${outfile}"
  # done
  # for f in ${OUTPUT_DIR}/train.tok.clean.vi; do
  #   outfile="${f%.*}.bpe.${merge_ops}.vi"
  #   subword-nmt apply-bpe -c "${OUTPUT_DIR}/bpe.${merge_ops}" < $f > "${outfile}"
  # done
  for lang in vi; do
    for f in in ${OUTPUT_DIR}/clear_valid.${lang} ${OUTPUT_DIR}/clear_test.${lang} ${OUTPUT_DIR}/clear_train.${lang}; do
      outfile="${f%.*}.tok.bpe.${merge_ops}.${lang}"
      subword-nmt apply-bpe -c "${OUTPUT_DIR}/bpe.${merge_ops}" < $f > "${outfile}"
    done
  done
  for f in in ${OUTPUT_DIR}/clear_valid.tok.en ${OUTPUT_DIR}/clear_test.tok.en ${OUTPUT_DIR}/clear_train.tok.en; do
      outfile="${f%.*}.bpe.${merge_ops}.en"
      subword-nmt apply-bpe -c "${OUTPUT_DIR}/bpe.${merge_ops}" < $f > "${outfile}"
    done
done
src=vi
tgt=en
PROJ_PATH=$HOME/pascal/experiments/iwslt${src}2${tgt}
test_path="$HOME/pascal/data/iwslten2vi/corpus/fairseq0102iwsltvi2en10k"
user_dir="$HOME/Attention-calibration-NMT/mycode-gate/"
model_path="$HOME/checkpoint/dep_transformer/iwsltvi2en/ACN/checkpoint_best.pt"
MODEL_DIR=$PROJ_PATH/attention_calibration_nmt
OUTPUT_FN=$MODEL_DIR/res.txt
fairseq-generate $test_path \
        --user-dir $user_dir \
        --criterion my_label_smoothed_cross_entropy --task attack_translation_task --path $model_path \
        --remove-bpe -s $src -t $tgt --beam 4 --lenpen 0.6 \
        > $OUTPUT_FN
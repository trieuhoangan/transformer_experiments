user_dir="$HOME/pascal/code/join_model/"
data_bin="$HOME/pascal/data/iwslten2vi/corpus/fairseq0102iwslten2vi10k"
TAGSDIR="$HOME/pascal/data/iwslten2vi/tags_root/fairseq0102iwslten2vi"
model_dir="$HOME/checkpoint/dep_transformer/iwslten2vi/join_model"

mkdir -p $model_dir
export CUDA_VISIBLE_DEVICES=0
# nohup fairseq-train $data_bin \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES fairseq-train $data_bin \
        --user-dir $user_dir --criterion join_label_smoothed_cross_entropy --task acn_tags_translation --arch join_lisa_acn_transformer \
        --tags-data $TAGSDIR \
        --optimizer myadam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt \
        --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 5e-04 --min-lr 1e-09 \
        --ddp-backend no_c10d \
        -s en -t vi --save-dir $model_dir \
        --encoder-lisa-layer 0 \
        --dropout 0.3 \
        --parent-ignoring 0.3 \
        --share-all-embeddings \
        --relu-dropout 0.1 \
        --weight-decay 0.1 \
        --label-smoothing 0.1 \
        --max-tokens 8000 \
        --max-update 120000 \
        --no-progress-bar \
        --log-format json \
        --log-interval 100 \
        --save-interval 500000 \
        --save-interval-updates 500 \
        --keep-interval-updates 1 \
        --reset-dataloader \
        --dataset-impl "mmap" \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-detok moses \
        --eval-bleu-remove-bpe \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
        # --mask-loss-weight 0.03 > log.train-chen-9 \
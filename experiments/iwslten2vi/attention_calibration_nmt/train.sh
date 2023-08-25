user_dir="$HOME/Attention-calibration-NMT/mycode-gate/"
data_bin="$HOME/pascal/data/iwslten2vi/corpus/fairseq0102iwslten2vi10k"
model_dir="$HOME/checkpoint/dep_transformer/iwslten2vi/ACN"
mkdir -p $model_dir
export CUDA_VISIBLE_DEVICES=0
# nohup fairseq-train $data_bin \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES fairseq-train $data_bin \
        --user-dir $user_dir --criterion my_label_smoothed_cross_entropy --task attack_translation_task --arch my_arch \
        --optimizer myadam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt \
        --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0005 --min-lr 1e-09 \
        --weight-decay 0.0 --label-smoothing 0.1 \
        --max-tokens 8192 --no-progress-bar --max-update 150000 \
        --log-interval 100 --save-interval-updates 1000 --keep-interval-updates 10 --save-interval 10000 --seed 1111 \
        --ddp-backend no_c10d \
        --dropout 0.3 \
        -s en -t vi --save-dir $model_dir \
        --mask-loss-weight 0.03 > log.train-chen-9 \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-detok moses \
        --eval-bleu-remove-bpe \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
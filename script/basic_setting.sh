TRAN_CMD="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES fairseq-train $DATADIR \
                --save-dir $CKPTS --tensorboard-logdir $CKPTS \
                -s $SRC_LANG -t $TGT_LANG --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 \
                --lr 7e-4 --lr-scheduler inverse_sqrt --warmup-updates 6000 \
                --ddp-backend no_c10d \
                --dropout 0.3 \
                --share-all-embeddings \
                --relu-dropout 0.1 --weight-decay 0.0001 --attention-dropout 0.1 \
                --label-smoothing 0.1 \
                --max-tokens 4096 --seed 1\
                --no-progress-bar \
                --log-format json \
                --log-interval 100 \
                --keep-last-epochs 5 \
                --reset-dataloader \
                --eval-bleu \
                --eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' \
                --eval-bleu-detok moses \
                --eval-bleu-remove-bpe \
                --best-checkpoint-metric bleu --maximize-best-checkpoint-metric"
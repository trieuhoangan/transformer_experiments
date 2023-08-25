SRC_LANG=vi
TGT_LANG=en
DATADIR="$HOME/pascal/data/iwslten2vi/VOLT_vi2en/iwslt_VOLT_vi2en"
CKPTS="$HOME/checkpoint/dep_transformer/iwsltvi2en/transformer_VOLT"
arch="--arch transformer_iwslt_de_en  --criterion label_smoothed_cross_entropy \
                --task translation "
CUDA_VISIBLE_DEVICES=0

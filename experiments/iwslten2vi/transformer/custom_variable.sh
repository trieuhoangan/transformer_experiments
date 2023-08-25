SRC_LANG=en 
TGT_LANG=vi
DATADIR="$HOME/pascal/data/iwslten2vi/VOLT_en2vi/iwslt_VOLT_en2vi"
CKPTS="$HOME/checkpoint/dep_transformer/iwslten2vi/transformer_VOLT"
arch="--arch transformer_wmt_en_de_big  --criterion label_smoothed_cross_entropy \
                --task translation "
CUDA_VISIBLE_DEVICES=0
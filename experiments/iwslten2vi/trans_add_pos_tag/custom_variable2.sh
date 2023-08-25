SRC_LANG=de
TGT_LANG=en
DATADIR="$HOME/pascal/data/iwsltdeen/corpus3/iwsltde2en0102"
CKPTS="$HOME/checkpoint/dep_transformer/iwsltde2en/trans_add_fre_product_method"
user_dir="$HOME/pascal/NMT/transformer_add_frequency_refined/"
DICTIONARYPATH="$HOME/pascal/data/iwsltdeen/pairs/dictionary2.de"
CUDA_VISIBLE_DEVICES=0
arch="--arch lisa_transformer_iwslt_de_en --pair-file $DICTIONARYPATH --user-dir $user_dir --encoder-lisa-layer 0"

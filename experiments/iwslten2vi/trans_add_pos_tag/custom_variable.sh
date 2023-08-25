SRC_LANG=en
TGT_LANG=vi
DATADIR="$HOME/pascal/data/iwslten2vi/corpus/fairseq0102iwslten2vi10k"
CKPTS="$HOME/checkpoint/dep_transformer/iwslten2vi/trans_add_pos_tag_ave"
user_dir="$HOME/pascal/NMT/additional_pos_tag/"
# DICTIONARYPATH="$HOME/pascal/data/iwsltdeen/pairs/tfidf.de"
TAGSDIR="$HOME/pascal/data/iwslten2vi/pos_tags/iwslten2vi"
WORDPATH="$HOME/pascal/data/iwsltdeen/pairs/tfidf_word.$SRC_LANG"
CUDA_VISIBLE_DEVICES=0
arch="  --arch lisa_transformer_iwslt_de_en \
        --user-dir $user_dir --tags-data $TAGSDIR --encoder-lisa-layer 0 --criterion label_smoothed_cross_entropy \
        --task tags_translation "

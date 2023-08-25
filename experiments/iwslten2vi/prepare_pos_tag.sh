#!/bin/bash

# Created by:         Emanuele Bugliarello (@e-bug)
# Date created:       9/4/2019
# Date last modified: 9/4/2019

PROJDIR=$HOME/pascal
INDIR=$PROJDIR/data/iwslten2vi/corpus
OUTDIR=$PROJDIR/data/iwslten2vi/pos_tags
lang=en
TRAIN=clear_train.tok.bpe.10000.$lang
VALID=clear_valid.tok.bpe.10000.$lang
TEST=clear_test.tok.bpe.10000.$lang
SCRIPTSDIR=$PROJDIR/NMT/script

mkdir -p $OUTDIR

size=10000
train_size=150000
i=0

python $SCRIPTSDIR/pos_tag.py $lang $INDIR/$TEST $OUTDIR/test.$lang $size $i
python $SCRIPTSDIR/pos_tag.py $lang $INDIR/$VALID $OUTDIR/valid.$lang $size $i

# python $SCRIPTSDIR/pos_tag.py $lang $INDIR/$TRAIN $OUTDIR/train.$lang $train_size $i &

for i in {0..23}; do
  python $SCRIPTSDIR/pos_tag.py $lang $INDIR/$TRAIN $OUTDIR/train.$lang.$i $size $i
done


wait
rm $OUTDIR/train.$lang
for i in {0..23}; do
  cat $OUTDIR/train.$lang.$i >> $OUTDIR/train.$lang
done
rm $OUTDIR/train.$lang.*
wait


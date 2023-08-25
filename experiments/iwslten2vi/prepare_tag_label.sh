PROJDIR=$HOME/pascal
INDIR=$PROJDIR/data/iwslten2vi/corpus
OUTDIR=$PROJDIR/data/iwslten2vi/tags_label
lang=en
TRAIN=train.clean.bpe.16000.$lang
VALID=valid.tok.bpe.16000.$lang
TEST=test.tok.bpe.16000.$lang
SCRIPTSDIR=$PROJDIR/scripts

mkdir -p $OUTDIR

size=10000
i=0

python $SCRIPTSDIR/bpe_tags_label.py $lang $INDIR/$TEST $OUTDIR/test.$lang $size $i &
python $SCRIPTSDIR/bpe_tags_label.py $lang $INDIR/$VALID $OUTDIR/valid.$lang $size $i &

for i in {0..23}; do
  python $SCRIPTSDIR/bpe_tags_label.py $lang $INDIR/$TRAIN $OUTDIR/train.$lang.$i $size $i &
done

wait

rm $OUTDIR/train.$lang
for i in {0..23}; do
  cat $OUTDIR/train.$lang.$i >> $OUTDIR/train.$lang
done
rm $OUTDIR/train.$lang.*
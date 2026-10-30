# Greedy eval on the held-out test split: one vLLM process per GPU, then merge + score with the training
# reward and the paper metrics. Called by scripts/run05/run06 after scripts/_common.sh. Reads:
#   NAME   results/<NAME>/   MODEL  HF checkpoint dir   TEST (test.jsonl)   N (1)   TEMPERATURE (0.0; not TEMP, which clusters set to a temp dir)   MAXTOK (16384)
NAME=${NAME:?set NAME}
MODEL=${MODEL:?set MODEL}
TEST=${TEST:-$DATA/test.jsonl}
N=${N:-1}; TEMPERATURE=${TEMPERATURE:-0.0}; MAXTOK=${MAXTOK:-16384}
WORKERS=$(( NCPUS / NGPUS > 2 ? NCPUS / NGPUS - 1 : 1 ))
echo "[eval] name=$NAME model=$MODEL test=$TEST n=$N temp=$TEMPERATURE max_tokens=$MAXTOK gpus=$NGPUS"

for i in $(seq 0 $((NGPUS - 1))); do
  CUDA_VISIBLE_DEVICES=$i python scripts/eval.py --model "$MODEL" --name "$NAME" --test "$TEST" --n "$N" \
      --temperature "$TEMPERATURE" --max-tokens "$MAXTOK" --shard "$i" --num-shards "$NGPUS" --workers "$WORKERS" &
done
wait
python scripts/eval.py --name "$NAME" --merge --num-shards "$NGPUS"
echo "[eval] results/$NAME/summary.md"

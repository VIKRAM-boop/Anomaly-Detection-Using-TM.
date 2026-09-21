#!/bin/bash
# Three branches at seed 42 first (completes spec Step 4 + enables the
# multi-scale ensemble), then two extra medium seeds for variance.
PY=/opt/anaconda3/bin/python3
LOG=/tmp/tmwork2
mkdir -p $LOG
for job in "short 42" "medium 42" "long 42" "medium 7" "medium 1234"; do
    set -- $job
    echo "=== START $1 seed$2 $(date +%H:%M:%S) ==="
    $PY -u 02c_train_with_scores.py $1 $2 2>&1 | tee -a $LOG/tm_$1_seed$2.log | grep -E "TRAIN accuracy|plain argmax|epoch|scores saved"
    echo "=== DONE $1 seed$2 $(date +%H:%M:%S) ==="
done
echo "=== ALL TM RUNS COMPLETE ==="

#!/usr/bin/env bash
# Runs the leave-one-class-out open-set sweep: for each attack class, rebuilds
# the dataset with that class hidden, trains the clause banks, and applies the
# decision hierarchy. Worms is excluded because it has only 52 test rows.
set -u
cd "$(dirname "$0")"
CLASSES=(Analysis Backdoor DoS Exploits Fuzzers Generic Reconnaissance Shellcode)
mkdir -p logs
for c in "${CLASSES[@]}"; do
  echo "################ hold out: $c ################"
  date
  python 09_openset_preprocess.py  "$c" 2>&1 | tee "logs/sweep_${c}_09.log" || { echo "09 FAILED for $c"; continue; }
  python 10_train_family_banks.py  "$c" 2>&1 | tee "logs/sweep_${c}_10.log" || { echo "10 FAILED for $c"; continue; }
  python 11_openset_decision.py    "$c" 2>&1 | tee "logs/sweep_${c}_11.log" || { echo "11 FAILED for $c"; continue; }
  echo "################ done: $c ################"
done
python 13_openset_summary.py

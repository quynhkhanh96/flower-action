#!/bin/usr/env bash
set -e
DATA_DIR=$1
WORK_DIR=$2

cd ../../../simulation

initial=0.1
end=0.95
inc=0.05

for p in $(seq $initial $inc $end)
do
    python -m stc_sim --work_dir="$DATA_DIR/stc_sim" \
        --data_dir=$DATA_DIR --server_device="cuda:1" \
        --aggregation="mean" --compression="stc_up" \
        --p_up=$p \
        --cfg_path="../examples/hmdb51/configs/hmdb51_fedbn_sim.yaml"
done
#!/bin/bash

USAGE="$0 <hyperparameter_config_file> <output_dir>"
if [ $# -ne 2 ]; then
    echo "Usage: $USAGE"
    exit 1
fi
CFG=$1
OUT=$2

HERE=$(dirname $0)
SCRIPT="$HERE/../run_sft_trainer.py"
MODEL="Qwen/Qwen3-8B-Base"

DATA_ARGS="--dataset-version=4 \
--hps-cfg=$CFG \
--ctp=$CTP \
--output-dir=$OUT"
TRAIN_ARGS="--epochs=1 \
--eval-steps=0.1 \
--eval-acc=7 \
--grad-acc=38 \
--batch-size=8 \
--eval-batch-size=14 \
--schedule=constant_with_warmup \
--rank-dimension=6 \
--lora-alpha=2"

BAR='------------------------------------------------------------------'
CMD="python $SCRIPT $MODEL $DATA_ARGS $TRAIN_ARGS"
echo $BAR
echo $CMD
echo $BAR
PYTORCH_ALLOC_CONF=expandable_segments:True $CMD
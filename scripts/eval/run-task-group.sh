#!/bin/bash

if [ $# -ge 7 ] || [ $# -le 3 ]; then
    echo "Usage: $0 <model_path> <task_group> <output_path> <accelerate_config> <library>[hf/vllm] <max_gen_toks>[default 2048]"
    echo 'Example task groups: general_en, general_fr, medical_fr, legal_en, etc.'
    exit 1
fi
MODEL_PATH=$1
TASK_GROUP=$2
OUTPUT_PATH=$3
ACCELERATE_CONFIG=$4
LIB=$5
MAX_GEN_TOKS=${6:-2048}

ENV_BIN="$WORK/miniconda3/envs/partages-dev/bin"
TASK_GROUP_FILE="$(dirname $0)/../../configs/lm-eval/task-groups-flat.yaml"
YQ_CMD="$ENV_BIN/yq eval .$TASK_GROUP $TASK_GROUP_FILE"
echo "Retrieving task names: $YQ_CMD"
YAML_TASKS=$($YQ_CMD)
if [ "$YAML_TASKS" = 'null' ]; then
    echo "Unrecognised task group: $TASK_GROUP"
    exit 1
fi
TASKS="${YAML_TASKS//'- task: '/''}"

LM_EVAL_ARGS_BASE="--log_samples \
--model $LIB \
--model_args pretrained=$MODEL_PATH \
--output_path $OUTPUT_PATH \
--gen_kwargs max_gen_toks=$MAX_GEN_TOKS"
if [[ "$LIB" == 'hf' ]]; then
    LM_EVAL_ARGS_BASE="$LM_EVAL_ARGS_BASE --limit 1000 --batch_size auto"
else
    export VLLM_CACHE_ROOT="$WORK/vllm-cache"
fi

echo "Running $MODEL_PATH on task group $TASK_GROUP"
export HF_HOME="$WORK/hf-home"
for t in $TASKS; do
    lm_eval_args_task="$LM_EVAL_ARGS_BASE --tasks $t"
    if [[ ( "$t" == *"med"* && "$t" != 'med_global_mmlu' ) || "$TASK_GROUP" == "med_mmlu"* || "$TASK_GROUP" == 'med_fr' || "$TASK_GROUP" == 'med_glianorex' ]]; then
        lm_eval_args_task="$lm_eval_args_task --num_fewshot 3"
    elif [[ "$TASK_GROUP" == 'maths_en' ]]; then
        lm_eval_args_task="$lm_eval_args_task --num_fewshot 4"
    else
        lm_eval_args_task="$lm_eval_args_task --num_fewshot 5"
    fi
    CMD="$ENV_BIN/accelerate launch --config_file=$ACCELERATE_CONFIG -m lm_eval $lm_eval_args_task"
    echo "Command: $CMD"
    $CMD
done

#!/bin/bash

# set defaults
GPU_TYPE='a'
NUM_GPUS=1
MODEL_DIR="$WORK/partages-models"
OUTPUT_DIR="$WORK/partages-data/eval-results"
MAX_GEN_TOKS=2048

# parse command line
CL_HELP='-g <gpu_type>[v|a|h] -n <num_gpus>[1|2|3|4] -m <model_dir> -o <output_dir> -t <max_gen_toks>'
USAGE="Usage: $0 $CL_HELP"
while getopts "g:n:m:o:t:" opt; do
    case $opt in
        g) GPU_TYPE="$OPTARG" ;;
        n) NUM_GPUS="$OPTARG" ;;
        m) MODEL_DIR="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        t) MAX_GEN_TOKS="$OPTARG" ;;
        *) echo $USAGE && exit 1 ;;
    esac
done
echo "MODEL_DIR $MODEL_DIR"
echo "OUTPUT_DIR $OUTPUT_DIR"
echo "MAX_GEN_TOKS $MAX_GEN_TOKS"

# GPU configuration
ACCELERATE_DIR="$WORK/hf-home/accelerate"
if [ $GPU_TYPE = 'v' ]
then
    CLUSTER_ARGS='-A bbo@v100 -C v100-32g'
elif [ $GPU_TYPE = 'a' ]
then
    CLUSTER_ARGS='-A bbo@a100 -C a100'
elif [ $GPU_TYPE = 'h' ]
then
    CLUSTER_ARGS='-A bbo@h100 -C h100'
else
    echo "Invalid value for gpu_type: $GPU_TYPE"
    echo 'Should be v, a, or h (V100/A100/H100)'
    exit 1
fi
if [ $NUM_GPUS = 1 ]
then
    CLUSTER_ARGS="$CLUSTER_ARGS --gres=gpu:1"
    ACCELERATE_CONFIG="$ACCELERATE_DIR/config_single_gpu.yaml"
elif [ $NUM_GPUS = 2 ]
then
    CLUSTER_ARGS="$CLUSTER_ARGS --gres=gpu:2"
    ACCELERATE_CONFIG="$ACCELERATE_DIR/config_multi_gpu_2.yaml"
elif [ $NUM_GPUS = 3 ]
then
    CLUSTER_ARGS="$CLUSTER_ARGS --gres=gpu:3"
    ACCELERATE_CONFIG="$ACCELERATE_DIR/config_multi_gpu_3.yaml"
elif [ $NUM_GPUS = 4 ]
then
    CLUSTER_ARGS="$CLUSTER_ARGS --gres=gpu:4"
    ACCELERATE_CONFIG="$ACCELERATE_DIR/config_multi_gpu_4.yaml"
else
    echo 'Accelerate configs only available for 1-4 GPUs'
    exit 1
fi
echo "Using accelerate config $ACCELERATE_CONFIG"

MODELS=(
    #################
    ### BASELINES ###
    #################
    # Qwen3-4B-Base
    # Qwen3-1.7B-Base
    # Qwen3-0.6B-Base
    # Qwen3-4B
    # Qwen3-1.7B
    # Qwen3-0.6B
    # Qwen3-14B-Base
    # Qwen3-14B-FP8
    # Qwen3-32B-FP8
    # Apertus-8B-Instruct-2509
    # Apertus-8B-2509
    # Gaperon-1125-1B
    # Gaperon-1125-8B
    # Gaperon-1125-8B-SFT
    # Gaperon-1125-24B
    # gemma-3-270m
    # gemma-3-1b-pt
    # gemma-3-4b-pt
    # gemma-3-270m-it
    # gemma-3-1b-it
    # gemma-3-4b-it
    # SmolLM3-3B
    # gpt2
    # Baguettotron
    # Olmo-3-1025-7B
    # Olmo-3-1125-32B
    # medgemma-4b-pt
    # medgemma-4b-it
    # medgemma-27b-text-it
    # Llama-3.2-1B-Instruct
    # Mistral-7B-Instruct-v0.3
    #################

    ##############
    ### PDAPT2 ###
    ##############
    # Apertus-8B-PDAPT2-1080
    # EuroLLM-9B-PDAPT2-720
    # Mistral-7B-PDAPT2-180
    # Apertus-8B-PDAPT2-1440
    # EuroLLM-9B-PDAPT2-890
    # Mistral-7B-PDAPT2-360
    # Apertus-8B-PDAPT2-1692
    # Gaperon-8B-PDAPT2-180
    # Mistral-7B-PDAPT2-540
    # Apertus-8B-PDAPT2-360
    # Gaperon-8B-PDAPT2-360
    # Mistral-7B-PDAPT2-720
    # Apertus-8B-PDAPT2-720
    # Gaperon-8B-PDAPT2-540
    # Mistral-7B-PDAPT2-900
    # EuroLLM-9B-PDAPT2-180
    # Gaperon-8B-PDAPT2-720
    # Qwen3-8B-PDAPT2-1440
    # EuroLLM-9B-PDAPT2-360
    # Gaperon-8B-PDAPT2-902
    # Qwen3-8B-PDAPT2-1628
    # EuroLLM-9B-PDAPT2-540
    # Mistral-7B-PDAPT2-1034
    # Qwen3-8B-PDAPT2-720
    ##############

    #####################
    ### PDAPT2 MERGES ###
    #####################
    # apertus8-pdapt2-slerp-26-01-20
    # eurollm-pdapt2-slerp-26-01-08
    # gaperon8-pdapt2-slerp-26-01-08
    # mistral7-pdapt2-slerp-26-01-20
    # qwen8-pdapt2-slerp-26-01-08
    #####################

    ##########################
    ### mix-v6 checkpoints ###
    ##########################
    # EuroLLM-9B_26-02-03-20-12-2040726/checkpoint-360
    # EuroLLM-9B_26-02-03-20-12-2040726/checkpoint-720
    # EuroLLM-9B_26-02-03-20-12-2040726/checkpoint-1080
    # EuroLLM-9B_26-02-03-20-12-2040726/checkpoint-1440
    # EuroLLM-9B_26-02-03-20-12-2040726/checkpoint-1800
    # EuroLLM-9B_26-02-03-20-12-2040726/checkpoint-2160
    # EuroLLM-9B_26-02-03-20-12-2040726/checkpoint-2520
    # EuroLLM-9B_26-02-03-20-12-2040726/checkpoint-2880
    # EuroLLM-9B_26-02-03-20-12-2040726/checkpoint-3240
    # Gaperon-1125-8B_26-02-05-11-54-2081619/checkpoint-360
    # Gaperon-1125-8B_26-02-05-11-54-2081619/checkpoint-720
    # Gaperon-1125-8B_26-02-05-11-54-2081619/checkpoint-1080
    # Gaperon-1125-8B_26-02-05-11-54-2081619/checkpoint-1440
    # Gaperon-1125-8B_26-02-05-11-54-2081619/checkpoint-1800
    # Gaperon-1125-8B_26-02-05-11-54-2081619/checkpoint-2160
    # Gaperon-1125-8B_26-02-05-11-54-2081619/checkpoint-2520
    # Gaperon-1125-8B_26-02-05-11-54-2081619/checkpoint-2880
    # Gaperon-1125-8B_26-02-05-11-54-2081619/checkpoint-3240
    # Gaperon-1125-8B_26-02-05-11-54-2081619/checkpoint-3600
    # Gaperon-1125-8B_26-02-05-11-54-2081619/checkpoint-3960
    # Mistral-7B-v0.3_26-02-05-11-54-2081616/checkpoint-360
    # Mistral-7B-v0.3_26-02-05-11-54-2081616/checkpoint-720
    # Mistral-7B-v0.3_26-02-05-11-54-2081616/checkpoint-1080
    # Mistral-7B-v0.3_26-02-05-11-54-2081616/checkpoint-1440
    # Mistral-7B-v0.3_26-02-05-11-54-2081616/checkpoint-1800
    # Mistral-7B-v0.3_26-02-05-11-54-2081616/checkpoint-2160
    # Mistral-7B-v0.3_26-02-05-11-54-2081616/checkpoint-2520
    # Mistral-7B-v0.3_26-02-05-11-54-2081616/checkpoint-2880
    # Mistral-7B-v0.3_26-02-05-11-54-2081616/checkpoint-3240
    # Mistral-7B-v0.3_26-02-05-11-54-2081616/checkpoint-3600
    # Mistral-7B-v0.3_26-02-05-11-54-2081616/checkpoint-3960
    # Qwen3-8B-Base_26-02-05-11-54-2081617/checkpoint-1440
    # Qwen3-8B-Base_26-02-05-11-54-2081617/checkpoint-2880
    # Qwen3-8B-Base_26-02-05-11-54-2081617/checkpoint-4320
    # Qwen3-8B-Base_26-02-05-11-54-2081617/checkpoint-5760
    ##########################

    ######################################
    ### com-clean-dedup-v2 checkpoints ###
    ######################################
    # Apertus-8B-2509_26-02-20-04-04-222665/checkpoint-540
    # Apertus-8B-2509_26-02-20-04-04-222665/checkpoint-1080
    # EuroLLM-9B_26-02-06-13-32-2107601/checkpoint-540
    # EuroLLM-9B_26-02-06-13-32-2107601/checkpoint-1080
    # EuroLLM-9B_26-02-06-13-32-2107601/checkpoint-1620
    # EuroLLM-9B_26-02-06-13-32-2107601/checkpoint-2160
    # EuroLLM-9B_26-02-06-13-32-2107601/checkpoint-2700
    # EuroLLM-9B_26-02-06-13-32-2107601/checkpoint-3240
    # Gaperon-1125-8B_26-02-06-13-32-2107597/checkpoint-540
    # Gaperon-1125-8B_26-02-06-13-32-2107597/checkpoint-1080
    # Gaperon-1125-8B_26-02-06-13-32-2107597/checkpoint-1620
    # Gaperon-1125-8B_26-02-06-13-32-2107597/checkpoint-2160
    # Gaperon-1125-8B_26-02-06-13-32-2107597/checkpoint-2700
    # Gaperon-1125-8B_26-02-06-13-32-2107597/checkpoint-3240
    # Gaperon-1125-8B_26-02-06-13-32-2107597/checkpoint-3640
    # Mistral-7B-v0.3_26-02-06-13-34-2107602/checkpoint-540
    # Mistral-7B-v0.3_26-02-06-13-34-2107602/checkpoint-1080
    # Mistral-7B-v0.3_26-02-06-13-34-2107602/checkpoint-1620
    # Mistral-7B-v0.3_26-02-06-13-34-2107602/checkpoint-2160
    # Mistral-7B-v0.3_26-02-06-13-34-2107602/checkpoint-2700
    # Mistral-7B-v0.3_26-02-06-13-34-2107602/checkpoint-3240
    # Mistral-7B-v0.3_26-02-06-13-34-2107602/checkpoint-3780
    # Qwen3-8B-Base_26-02-06-14-28-2107603/checkpoint-1440
    # Qwen3-8B-Base_26-02-06-14-28-2107603/checkpoint-2880
    # Qwen3-8B-Base_26-02-06-14-28-2107603/checkpoint-4320
    # Qwen3-8B-Base_26-02-06-14-28-2107603/checkpoint-5760
    ######################################

    #####################################
    ### research-clean-v2 checkpoints ###
    #####################################
    # EuroLLM-9B_26-02-12-08-03-16879/checkpoint-720
    # EuroLLM-9B_26-02-12-08-03-16879/checkpoint-1440
    # EuroLLM-9B_26-02-12-08-03-16879/checkpoint-2160
    # EuroLLM-9B_26-02-12-08-03-16879/checkpoint-2880
    # EuroLLM-9B_26-02-12-08-03-16879/checkpoint-3224
    # Gaperon-1125-8B_26-02-12-10-13-16880/checkpoint-720
    # Gaperon-1125-8B_26-02-12-10-13-16880/checkpoint-1440
    # Gaperon-1125-8B_26-02-12-10-13-16880/checkpoint-2160
    # Gaperon-1125-8B_26-02-12-10-13-16880/checkpoint-2880
    # Gaperon-1125-8B_26-02-12-10-13-16880/checkpoint-3288
    # Mistral-7B-v0.3_26-02-14-00-21-63097/checkpoint-720
    # Mistral-7B-v0.3_26-02-14-00-21-63097/checkpoint-1440
    # Mistral-7B-v0.3_26-02-14-00-21-63097/checkpoint-2160
    # Mistral-7B-v0.3_26-02-14-00-21-63097/checkpoint-2880
    # Mistral-7B-v0.3_26-02-14-00-21-63097/checkpoint-3600
    # Mistral-7B-v0.3_26-02-14-00-21-63097/checkpoint-3776
    #####################################
)
TASK_GROUPS=(
    # med_glianorex
    # global_mmlu_en
    # global_mmlu_fr
    # general_x_en
    # general_x_fr
    # mmlu_prox_en_bio
    # mmlu_prox_en_bus
    # mmlu_prox_en_chem
    # mmlu_prox_en_cs
    # mmlu_prox_en_econ
    # mmlu_prox_en_eng
    # mmlu_prox_en_hist
    # mmlu_prox_en_math
    # mmlu_prox_en_philo~20
    # mmlu_prox_en_phys
    # mmlu_prox_en_law~20
    # mmlu_prox_en_psy~20
    # mmlu_prox_fr_bio
    # mmlu_prox_fr_bus
    # mmlu_prox_fr_chem
    # mmlu_prox_fr_cs
    # mmlu_prox_fr_econ~20
    # mmlu_prox_fr_eng
    # mmlu_prox_fr_hist~08
    # mmlu_prox_fr_law~20
    # mmlu_prox_fr_math
    # mmlu_prox_fr_philo~20
    # mmlu_prox_fr_phys
    # mmlu_prox_fr_psy~20
    # mmlu_prox_en_health~20
    # mmlu_prox_fr_health
    # med_global_mmlu
    # med_mmlu_fr~01
    # med_mmlu_en~01
)

EVAL_SCRIPT="$WORK/partages-llm/scripts/eval/run-task-group.sh"
echo "Submitting batch jobs with $CLUSTER_ARGS"
for task_group in "${TASK_GROUPS[@]}"; do
    if [[ "$task_group" == *"~"* ]]; then
        IFS='~' read -ra task_group_params <<< $task_group
        task_group_name=${task_group_params[0]}
        hrs=${task_group_params[1]}
    else
        task_group_name=$task_group
        hrs='20'
    fi
    sbatch_args="$CLUSTER_ARGS --time=${hrs}:00:00"
    for model_name in "${MODELS[@]}"; do
        model_path="$MODEL_DIR/$model_name"
        job_name="eval-$model_name-$task_group_name"
        log_path="$OUTPUT_DIR/logs-slurm/$job_name-%j.log"
        sbatch_cmd="sbatch $sbatch_args --output=$log_path --job-name=$job_name"
        eval_script_args="$model_path $task_group_name $OUTPUT_DIR $ACCELERATE_CONFIG vllm $MAX_GEN_TOKS"
        echo ''
        echo "Eval run: $model_path --> $task_group_name"
        $sbatch_cmd --wrap="bash $EVAL_SCRIPT $eval_script_args"
    done
done
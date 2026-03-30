#!/bin/bash

# set defaults
HERE=$(dirname $0)
PROJECT_HOME="$HERE/../../.."
CONFIG_FILE="$PROJECT_HOME/config/sft/peft-hyperparam-grid.json"
HPS_SCRIPT="$PROJECT_HOME/scripts/train/sft/hps-run-scripts/run-hps-qwen8.sh"
GPU_TYPE='a'
HRS='12'
OUTPUT_DIR="$WORK/partages-models/sft/hps-runs"

# parse command line
CL_HELP='-c <hps_config_directory> -n <num_grid_chunks> -s <hps_script> -g <gpu_type>[a|h] -H <submission_time_hours> -o <output_dir>'
USAGE="Usage: $0 $CL_HELP"
while getopts "c:n:s:g:H:o:" opt; do
    case $opt in 
        c) CONFIG_FILE="$OPTARG" ;;
        n) NUM_GRID_CHUNKS="$OPTARG" ;;
        s) HPS_SCRIPT="$OPTARG" ;;
        g) GPU_TYPE="$OPTARG" ;;
        H) HRS="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        *) echo $USAGE && exit 1 ;;
    esac
done
echo "CONFIG_FILE: $CONFIG_FILE"
echo "HPS_SCRIPT: $HPS_SCRIPT"
echo "OUTPUT_DIR: $OUTPUT_DIR"

# unroll grid search configuration
UNROLL_SCRIPT="$PROJECT_HOME/scripts/preprocess/unroll_sft_config.py"
UNROLLED_CONFIG_DIR=$(python $UNROLL_SCRIPT $CONFIG_FILE -c $NUM_GRID_CHUNKS)
echo "Using unrolled configuration files in $UNROLLED_CONFIG_DIR"

# Resource configuration
if [[ $GPU_TYPE == 'a' || $GPU_TYPE == 'h' ]]
then
    GPU_ARGS="-A bbo@${GPU_TYPE}100 -C ${GPU_TYPE}100 --gres=gpu:1 --time=${HRS}:00:00"
    CPU_ARGS='--cpus-per-task=8 --hint=nomultithread'
else
    echo "Invalid value for gpu_type: $GPU_TYPE"
    echo 'Should be a or h (A100/H100)'
    exit 1
fi
echo "RESOURCE ARGS: $GPU_ARGS $CPU_ARGS"

# SLURM submissions
CONFIG_FILENAME=$(basename $HPS_SCRIPT)
JOB_NAME="SFT-${CONFIG_FILENAME%.*}"
echo "JOB_NAME: $JOB_NAME"
for filename in $UNROLLED_CONFIG_DIR/*; do
    echo ''
    echo "CONFIG FILE: $(basename $filename)"
    log_path_out="$OUTPUT_DIR/logs-slurm/$JOB_NAME-%j.out"
    log_path_err="$OUTPUT_DIR/logs-slurm/$JOB_NAME-%j.err"
    job_args="--output=$log_path_out --error=$log_path_err --job-name=$JOB_NAME"
    sbatch_cmd="sbatch $GPU_ARGS $CPU_ARGS $job_args"
    $sbatch_cmd --wrap="bash $HPS_SCRIPT $filename $OUTPUT_DIR"
done


#!/bin/bash
# eval_highway_deit_simplified.sh
# Created by: Jonathan Mikler - 14/Nov/24

SCRIPT_PATH=$(realpath $0)
source $(dirname $SCRIPT_PATH)/shell_utils.sh

check_conda_env

# arguments
# Check if --verbose or -v is given in args
VERBOSE=false
DRY_RUN=false
for arg in "$@"; do
  if [ "$arg" == "--verbose" ] || [ "$arg" == "-v" ]; then
    VERBOSE=true
  fi
  if [ "$arg" == "--dry-run" ] || [ "$arg" == "-d" ]; then
    DRY_RUN=true
  fi
done

##### Parameters
BASE_PATH='/home/iony/DTU/f24/thesis/code/lgvit/lgvit_repo'
CHECKPOINT_PATH="/home/iony/DTU/f24/thesis/code/lgvit/LGViT-ViT-Cifar100"
IGNORE_MISMATCHED_SIZES=False

model_path="${BASE_PATH}/models/deit_highway"

export PYTHONPATH=$BASE_PATH:$PYTHONPATH # Add path to the beginning of the search path
export PYTHONPATH="$PYTHONPATH:$model_path" # Add the model path to the end of the search path

BACKBONE=ViT # ViT, DeiT
MODEL_TYPE=${BACKBONE}-base
MODEL_NAME=facebook/deit-base-distilled-patch16-224

DATASET=uoft-cs/cifar100      # uoft-cs/cifar100, Food101, Maysee/tiny-imagenet, imagenet-1k
if [ $DATASET = 'Maysee/tiny-imagenet' ]; then
  DATANAME=tiny-imagenet
else
  DATANAME=$DATASET
fi

EXIT_STRATEGY=confidence # entropy, confidence, patience, patient_and_confident
HIGHWAY_TYPE=LGViT # linear, LGViT, vit, self_attention, conv_normal
PAPER_NAME=LGViT  # base, SDN, PABEE, PCEE, BERxiT, ViT-EE, LGViT

# export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_VISIBLE_DEVICES=0
report_to='none' # none, wandb

#export WANDB_PROJECT=${BACKBONE}_${DATANAME}_eval

##### Program run

args="--run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --model_name_or_path ${CHECKPOINT_PATH} \
    --dataset_name $DATASET \
    --output_dir $BASE_PATH/outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval True\
    --per_device_eval_batch_size 1 \
    --seed 777 \
    --report_to $report_to \
    --use_auth_token False \
    --ignore_mismatched_sizes $IGNORE_MISMATCHED_SIZES \
    --highway_type $HIGHWAY_TYPE \
    "

if [ "$VERBOSE" = true ]; then
  echo -e "Arguments:\n$args"
fi
if [ "$DRY_RUN" = true ]; then
  echo -e "Dry run, exiting..."
  exit 0
fi

python "${BASE_PATH}/examples/run_highway_deit.py" $args

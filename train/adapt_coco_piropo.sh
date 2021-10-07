#!/bin/bash
# dirs
TOOL_DIR=mmdetection/tools
CONFIG_FILE=mmdetection/configs/adaptive/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person_adaptive.py
WORK_DIR=~/WORK_DIRS/GPA/tuning
WORK_ROOT=coco_piropo
RES_DIR=results/GPA/tuning
RES_ROOT=coco_piropo

# model
# LOAD_FROM=mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth
LOAD_FROM=mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227_split.pth

# data
SRC_DIR=/data/COCO/train2017
SRC_FILE=/data/COCO/annotations/person_train2017.json
TGT_DIR=/data/PIROPO
TGT_ROOT=omni_training
TEST_FILE=/data/PIROPO/omni_test2.json

# GPUs
N_GPU=4 # if this is changed, the number of iterations also needs to be adapted
VIS_GPU=0,1,2,3
GPU_PORT=29500

# TRAINING PARAMETERS
BATCH=16

# POSITIONAL ARGS
EPOCHS=$1; shift
n=$1; shift
x=$1; shift
MODEL=$1; shift
NAME=$1; shift
# OPTIONAL ARGS
PRETRAIN=0
for arg in "$@"; do
    case $arg in
        -p|--pretrain)
        PRETRAIN=1
        shift
        ;;
    esac
done

# WORK DIRECTORY AND OUTPUT
if [[ "$PRETRAIN" == 1 ]]; then
    OUT=${MODEL}_${NAME}_sp
else
    OUT=${MODEL}_${NAME}_s
fi
_WORK_DIR=${WORK_DIR}/${WORK_ROOT}_${n}${x}_${OUT}

mkdir -p ~/WORK_DIRS
mkdir -p ~/WORK_DIRS/GPA
mkdir -p ${WORK_DIR}
mkdir -p ${RES_DIR}
mkdir -p ${_WORK_DIR}

# PRETEAIN DOMAIN CLASSIFIERS IF NECESSARY
# - no early-stopping
# - higher lr
# - all lrs except for domain classifiers set to 0
# - only 20 epochs
if [[ "$PRETRAIN" == 1 ]]; then
    CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_DIR}/dist_train_adaptive.sh \
    ${CONFIG_FILE} \
    ${N_GPU} \
    --work-dir ${_WORK_DIR} \
    --seed 42 \
    --cfg-options data.samples_per_gpu=$(($BATCH/$N_GPU)) \
        data.train_src.ann_file=${SRC_FILE} \
        data.train_src.img_prefix=${SRC_DIR} \
        data.train_tgt.ann_file=${TGT_DIR}/${TGT_ROOT}_${n}${x}.json \
        data.train_tgt.img_prefix=None \
        data.val.ann_file=${TEST_FILE} \
        data.val.img_prefix=None \
        runner.max_epochs=20 \
        evaluation._delete_=True \
        evaluation.interval=1 \
        evaluation.metric=bbox \
        optimizer.lr=0.01 \
        optimizer.paramwise_cfg.custom_keys.backbone.lr_mult=0 \
        optimizer.paramwise_cfg.custom_keys.neck.lr_mult=0 \
        optimizer.paramwise_cfg.custom_keys.rpn_head.lr_mult=0 \
        optimizer.paramwise_cfg.custom_keys.roi_head.lr_mult=0 \
        model.type=${MODEL} \
        load_from=${LOAD_FROM} \
        "$@" \
        2>&1 | tee ${RES_DIR}/${RES_ROOT}_${n}${x}_${OUT}_pre.log

    # use pretrained model for actual training
    mv ${_WORK_DIR}/epoch_20.pth ${_WORK_DIR}/pre.pth
    LOAD_FROM=${_WORK_DIR}/pre.pth
fi

# TRAIN
NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_DIR}/dist_train_adaptive.sh \
${CONFIG_FILE} \
${N_GPU} \
--work-dir ${_WORK_DIR} \
--seed 42 \
--cfg-options data.samples_per_gpu=$(($BATCH/$N_GPU)) \
    data.train_src.ann_file=${SRC_FILE} \
    data.train_src.img_prefix=${SRC_DIR} \
    data.train_tgt.ann_file=${TGT_DIR}/${TGT_ROOT}_${n}${x}.json \
    data.train_tgt.img_prefix=None \
    data.val.ann_file=${TEST_FILE} \
    data.val.img_prefix=None \
    runner.max_epochs=${EPOCHS} \
    model.type=${MODEL} \
    load_from=${LOAD_FROM} \
    "$@" \
    2>&1 | tee ${RES_DIR}/${RES_ROOT}_${n}${x}_${OUT}.log

# CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_DIR}/dist_test.sh \
# ${CONFIG_FILE} \
# ${_WORK_DIR}/latest.pth \
# ${N_GPU} \
# --eval bbox \
# --cfg-options data.samples_per_gpu=$(($BATCH/$N_GPU)) \
#     data.test.ann_file=${TEST_FILE} \
#     data.test.img_prefix=None \
# --out ${_WORK_DIR}/piropo_test2.pkl 2>&1 | tee ${RES_DIR}/${RES_ROOT}_${n}${x}_${OUT}.txt

# free up space
# rm ${_WORK_DIR}/*.log
# rm ${_WORK_DIR}/*.pkl
# rm ${_WORK_DIR}/*.json

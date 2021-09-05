# dirs
TOOL_DIR=mmdetection/tools
CONFIG_FILE=mmdetection/configs/adaptive/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person_adaptive.py
WORK_DIR=~/WORK_DIRS/GPA/tuning
WORK_ROOT=coco_piropo
RES_DIR=results/GPA/tuning
RES_ROOT=coco_piropo

# data
SRC_DIR=/data/COCO/train2017
SRC_FILE=/data/COCO/annotations/person_train2017.json
TGT_DIR=/data/PIROPO
TGT_ROOT=omni_training
TEST_FILE=/data/PIROPO/omni_test2.json

# GPUs
N_GPU=8 # if this is changed, the number of iterations also needs to be adapted
VIS_GPU=0,1,2,3,4,5,6,7
GPU_PORT=29502

mkdir -p ~/WORK_DIRS
mkdir -p ~/WORK_DIRS/GPA
mkdir -p ${WORK_DIR}
mkdir -p ${RES_DIR}

# TRAINING PARAMETERS
BATCH=16
EPOCHS=$1
n=$2
x=$3
# with all images, we train 12 epochs:
#   ceil(2357/16) = 148 iterations --> 148 * 12 = 1776
# (( ITER_PER_EPOCH=($n+$BATCH-1)/$BATCH )) # this is a custom implementation of the ceiling function
# (( EPOCHS=(1776+$ITER_PER_EPOCH-1)/$ITER_PER_EPOCH ))
# with 100 images, we need smaller lr after 6 epochs:
#   ceil(100/16) = 7 --> 7 * 6 = 42 --> lr update after 42 iterations
# (( LR_UPDATE_STEP=(42+$ITER_PER_EPOCH-1)/$ITER_PER_EPOCH ))
# we want to evaluate ca. 10 times per model --> every 178 iterations
# (( VAL_INTERVAL=(178+$ITER_PER_EPOCH-1)/$ITER_PER_EPOCH ))

# only fine-tuning
MODEL=$4
SUFFIX=$5

OUT=${MODEL}_seed_${SUFFIX}

_WORK_DIR=${WORK_DIR}/${WORK_ROOT}_${n}${x}_${OUT}
mkdir -p ${_WORK_DIR}

# use this for iter-based runner:
# runner._delete_=True \
# runner.type='IterBasedRunnerAdaptive' \
# runner.max_iters=140 \
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
    runner.max_epochs=${EPOCHS} \
    model.type=${MODEL} \
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

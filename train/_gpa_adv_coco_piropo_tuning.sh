# dirs
TOOL_DIR=mmdetection/tools
CONFIG_FILE=mmdetection/configs/adaptive/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person_adaptive.py
CONFIG_FILE_ADV=mmdetection/configs/adaptive/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person_adaptive_only_adv.py
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
ROI_INTRA=$5
ROI_INTER=$6
RCNN_INTRA=$7
RCNN_INTER=$8
GRAPH=$9
FC=${10}
SUFFIX=${11}

OUT=${MODEL}_${ROI_INTRA//.}_${ROI_INTER//.}_${RCNN_INTRA//.}_${RCNN_INTER//.}_${FC//[_-.]}_g${GRAPH}_seed${SUFFIX}

_WORK_DIR=${WORK_DIR}/${WORK_ROOT}_${n}${x}_${OUT}
_WORK_DIR_ADV=${WORK_DIR}/${WORK_ROOT}_${n}${x}_${OUT}_adv
mkdir -p ${_WORK_DIR}
mkdir -p ${_WORK_DIR_ADV}

# use this for iter-based runner:
# runner._delete_=True \
# runner.type='IterBasedRunnerAdaptive' \
# runner.max_iters=140 \

# first train only domain classifiers
# CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_DIR}/dist_train_adaptive.sh \
# ${CONFIG_FILE_ADV} \
# ${N_GPU} \
# --work-dir ${_WORK_DIR_ADV} \
# --seed 42 \
# --cfg-options data.samples_per_gpu=$(($BATCH/$N_GPU)) \
#     data.train_src.ann_file=${SRC_FILE} \
#     data.train_src.img_prefix=${SRC_DIR} \
#     data.train_tgt.ann_file=${TGT_DIR}/${TGT_ROOT}_${n}${x}.json \
#     data.train_tgt.img_prefix=None \
#     data.val.ann_file=${TEST_FILE} \
#     data.val.img_prefix=None \
#     evaluation.interval=1 \
#     checkpoint_config.interval=200 \
#     model.type=${MODEL} \
#     model.train_cfg.gpa.loss_roi_intra=${ROI_INTRA} \
#     model.train_cfg.gpa.loss_roi_inter=${ROI_INTER} \
#     model.train_cfg.gpa.loss_rcnn_intra=${RCNN_INTRA} \
#     model.train_cfg.gpa.loss_rcnn_inter=${RCNN_INTER} \
#     model.train_cfg.gpa.use_graph=${GRAPH} \
#     model.train_cfg.gpa.fc_layer=${FC} \
#     2>&1 | tee ${RES_DIR}/${RES_ROOT}_${n}${x}_${OUT}_adv.log

# then train normally
# LOAD_FROM=$(ls ${_WORK_DIR_ADV}/latest.pth)
# LOAD_FROM=/home/thaddaus/WORK_DIRS/GPA/tuning/coco_piropo_20a_TwoStageDetectorAdaptiveAdversarial_1_1_1_1_none_gTrue_seed_direct5_adv/latest.pth
LOAD_FROM=mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth

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
    lr_config.step=[200] \
    evaluation.interval=1 \
    checkpoint_config.interval=200 \
    model.type=${MODEL} \
    model.train_cfg.gpa.loss_roi_intra=${ROI_INTRA} \
    model.train_cfg.gpa.loss_roi_inter=${ROI_INTER} \
    model.train_cfg.gpa.loss_rcnn_intra=${RCNN_INTRA} \
    model.train_cfg.gpa.loss_rcnn_inter=${RCNN_INTER} \
    model.train_cfg.gpa.use_graph=${GRAPH} \
    model.train_cfg.gpa.fc_layer=${FC} \
    load_from=${LOAD_FROM} \
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

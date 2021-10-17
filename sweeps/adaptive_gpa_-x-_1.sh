TOOL_DIR=mmdetection/tools

# special training for models containing the Δxywh-GPA head for training sets with only 1 sample
# the gt_iou_thrs is decreased to 0.25 from 0.5 for this case

# MODEL and OUTPUT
# CONFIG_FILE=mmdetection/configs/sweeps/gpa_pxg.py
# WORK_DIR=WORK_DIRS/sweeps/gpa_pxg_
# RES_DIR=results/sweeps/gpa_pxg_
CONFIG_FILE=mmdetection/configs/sweeps/gpa_-xg.py
WORK_DIR=WORK_DIRS/sweeps/gpa_-xg_
RES_DIR=results/sweeps/gpa_-xg_
mkdir -p ${WORK_DIR}
mkdir -p ${RES_DIR}

# DATA
SRC_ANNS=/data/COCO/annotations/person_train2017.json
SRC_PRE=/data/COCO/train2017
TGT_ANNS=/data/PIROPO/omni_training
TGT_PRE=None
TEST_ANNS=/data/PIROPO/omni_test2.json
TEST_PRE=None

# GPUs
N_GPU=4
VIS_GPU=0,1,2,3
GPU_PORT=29503

# TRAINING
EPOCHS=40
BATCH=16

# training on n samples per dataset, with x datasets labeled a through j each
n=1
for x in {a,c}
do
    mkdir -p ${WORK_DIR}/${n}${x}

    # train
    CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_DIR}/dist_train_adaptive.sh \
    ${CONFIG_FILE} \
    ${N_GPU} \
    --work-dir ${WORK_DIR}/${n}${x} \
    --cfg-options data.samples_per_gpu=$(($BATCH/$N_GPU)) \
        data.train_src.ann_file=${SRC_ANNS} \
        data.train_src.img_prefix=${SRC_PRE} \
        data.train_tgt.ann_file=${TGT_ANNS}_${n}${x}.json \
        data.train_tgt.img_prefix=${TGT_PRE} \
        data.val.ann_file=${TEST_ANNS} \
        data.val.img_prefix=${TEST_PRE} \
        runner.max_epochs=${EPOCHS} \
        model.train_cfg.da.0.gt_iou_thrs=0 \
        2>&1 | tee ${RES_DIR}/${n}${x}.log

    # free up space
    rm ${WORK_DIR}/${n}${x}/*.log # we already save the log in the results file
    rm ${WORK_DIR}/${n}${x}/*.json # we don't need the log as .json file
    rm ${WORK_DIR}/${n}${x}/epoch_40.pth # we only want the best, not the last model
    rm ${WORK_DIR}/${n}${x}/latest.pth # we only want the best, not the last model
    rm ${WORK_DIR}/${n}${x}/*.py # we already save the model config in the log file
done
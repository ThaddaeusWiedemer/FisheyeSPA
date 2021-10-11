TOOL_DIR=mmdetection/tools

# MODEL and OUTPUT
# CONFIG_FILE=mmdetection/configs/sweeps/gpa_ppp.py
# WORK_DIR=WORK_DIRS/sweeps/gpa_ppp
# RES_DIR=results/sweeps/gpa_ppp
# CONFIG_FILE=mmdetection/configs/sweeps/gpa_p-p.py
# WORK_DIR=WORK_DIRS/sweeps/gpa_p-p
# RES_DIR=results/sweeps/gpa_p-p
# CONFIG_FILE=mmdetection/configs/sweeps/gpa_pxg.py
# WORK_DIR=WORK_DIRS/sweeps/gpa_pxg
# RES_DIR=results/sweeps/gpa_pxg
# CONFIG_FILE=mmdetection/configs/sweeps/gpa_-xg.py
# WORK_DIR=WORK_DIRS/sweeps/gpa_-xg
# RES_DIR=results/sweeps/gpa_-xg
CONFIG_FILE=mmdetection/configs/sweeps/adv_0.py
WORK_DIR=WORK_DIRS/sweeps/adv_0_p
RES_DIR=results/sweeps/adv_0_p
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
GPU_PORT=29502

# TRAINING
EPOCHS=40
EPOCHS_PRE=10
BATCH=16

# training on n samples per dataset, with x datasets labeled a through j each
for n in {1,2,5,10,20,50,100}
do
    for x in {a..c}
    do
        mkdir -p ${WORK_DIR}/${n}${x}

        # pretrain domain classifiers
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
            runner.max_epochs=${EPOCHS_PRE} \
            evaluation._delete_=True \
            evaluation.interval=1 \
            evaluation.metric=bbox \
            optimizer.lr=0.01 \
            optimizer.paramwise_cfg.custom_keys.backbone.lr_mult=0 \
            optimizer.paramwise_cfg.custom_keys.neck.lr_mult=0 \
            optimizer.paramwise_cfg.custom_keys.rpn_head.lr_mult=0 \
            optimizer.paramwise_cfg.custom_keys.roi_head.lr_mult=0 \
            2>&1 | tee ${RES_DIR}/${n}${x}_pre.log

        # rename pretrained model
        mv ${WORK_DIR}/${n}${x}/epoch_10.pth ${WORK_DIR}/${n}${x}/pretrained.pth

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
            load_from=${WORK_DIR}/${n}${x}/pretrained.pth \
            2>&1 | tee ${RES_DIR}/${n}${x}.log
        
        # free up space
        rm ${WORK_DIR}/${n}${x}/*.log # we already save the log in the results file
        rm ${WORK_DIR}/${n}${x}/*.json # we don't need the log as .json file
        rm ${WORK_DIR}/${n}${x}/epoch_40.pth # we only want the best, not the last model
        rm ${WORK_DIR}/${n}${x}/latest.pth # we only want the best, not the last model
        rm ${WORK_DIR}/${n}${x}/*.py # we already save the model config in the log file

    done
done

# train once on whole dataset
# mkdir ${WORK_DIR}/all

# CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_DIR}/dist_train_adaptive.sh \
# ${CONFIG_FILE} \
# ${N_GPU} \
# --work-dir ${WORK_DIR}/all \
# --cfg-options data.samples_per_gpu=$(($BATCH/$N_GPU)) \
#     data.train_src.ann_file=${SRC_ANNS} \
#     data.train_src.img_prefix=${SRC_PRE} \
#     data.train_tgt.ann_file=${TGT_ANNS}.json \
#     data.train_tgt.img_prefix=${TGT_PRE} \
#     data.val.ann_file=${TEST_ANNS} \
#     data.val.img_prefix=${TEST_PRE} \
#     runner.max_epochs=${EPOCHS} \
#     2>&1 | tee ${RES_DIR}/all.log

# dirs
TOOL_DIR=mmdetection/tools
CONFIG_FILE=mmdetection/configs/adaptive/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person_adaptive.py
WORK_DIR=work_dirs/GPA
WORK_ROOT=coco_piropo
RES_DIR=results/GPA
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
GPU_PORT=29501

# mkdir $WORK_DIR
# mkdir $RES_DIR

# no fine-tuning
# mkdir ${WORK_DIR}/${WORK_ROOT}_0
# CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} python ${TOOL_DIR}/test.py \
#     ${CONFIG_FILE} \
#     mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth \
#     --eval bbox \
#     --cfg-options data.samples_per_gpu=$((16/$N_GPU)) \
#         data.test.ann_file=${TEST_FILE} \
#         data.text.img_prefix=None \
#     --out ${WORK_DIR}/${WORK_ROOT}_0/piropo_test2.pkl 2>&1 | tee ${RES_DIR}/${RES_ROOT}_0.txt

# finetuning on n samples per dataset, with 10 datasets labeled a through j per n
# for n in {1,2,5,10,20,50,100,200,500,1000}; do
# for n in {100,200,500,1000}; do
#     for x in {a..j}; do
        n=100
        x=a
        _WORK_DIR=${WORK_DIR}/${WORK_ROOT}_${n}${x}
        mkdir ${_WORK_DIR}

        # train
        CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_DIR}/dist_train_adaptive.sh \
        ${CONFIG_FILE} \
        ${N_GPU} \
        --work-dir ${_WORK_DIR} \
        --cfg-options data.samples_per_gpu=$((16/$N_GPU)) \
            data.train_src.ann_file=${SRC_FILE} \
            data.train_src.img_prefix=${SRC_DIR} \
            data.train_tgt.ann_file=${TGT_DIR}/${TGT_ROOT}_${n}${x}.json \
            data.train_tgt.img_prefix=None \
            data.val.ann_file=${TEST_FILE} \
            data.val.img_prefix=None 2>&1 | tee ${RES_DIR}/${RES_ROOT}_${n}${x}.log

        # test
        CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_DIR}/dist_test.sh \
        ${CONFIG_FILE} \
        ${_WORK_DIR}/latest.pth \
        ${N_GPU} \
        --eval bbox \
        --cfg-options data.samples_per_gpu=$((16/$N_GPU)) \
            data.test.ann_file=${TEST_FILE} \
            data.test.img_prefix=None \
        --out ${_WORK_DIR}/piropo_test2.pkl 2>&1 | tee ${RES_DIR}/${RES_ROOT}_${n}${x}.txt

        # free up space
        # rm ${_WORK_DIR}/*.log
        rm ${_WORK_DIR}/epoch_[1-9].pth
        rm ${_WORK_DIR}/epoch_10.pth
        rm ${_WORK_DIR}/epoch_11.pth
#     done
# done

# finetune once on whole dataset
# _WORK_DIR=${WORK_DIR}/${WORK_ROOT}_all
# mkdir ${_WORK_DIR}

# # train
# CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_DIR}/dist_train_adaptive.sh \
# ${CONFIG_FILE} \
# ${N_GPU} \
# --work-dir ${_WORK_DIR} \
# --cfg-options data.samples_per_gpu=$((16/$N_GPU)) \
#     data.train_src.ann_file=${SRC_FILE} \
#     data.train_src.img_prefix=${SRC_DIR} \
#     data.train_tgt.ann_file=${TGT_DIR}/${TGT_ROOT}.json \
#     data.train_tgt.img_prefix=None \
#     data.val.ann_file=${TEST_FILE} \
#     data.val.img_prefix=None

# # test
# CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_DIR}/dist_test.sh \
# ${CONFIG_FILE} \
# ${_WORK_DIR}/latest.pth \
# ${N_GPU} \
# --eval bbox \
# --cfg-options data.samples_per_gpu=$((16/$N_GPU)) \
#     data.test.ann_file=${TEST_FILE} \
#     data.test.img_prefix=None \
# --out ${_WORK_DIR}/piropo_test2.pkl 2>&1 | tee ${RES_DIR}/${RES_ROOT}_all.txt

# # free up space
# rm ${_WORK_DIR}/*.log
# rm ${_WORK_DIR}/epoch_[1-9].pth
# rm ${_WORK_DIR}/epoch_10.pth
# rm ${_WORK_DIR}/epoch_11.pth



# mkdir ${WORK_DIR}_all

# CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_PATH}/dist_train.sh \
#     ${CONFIG_FILE} \
#     ${N_GPU} \
#     --work-dir ${WORK_DIR}_all \
#     --cfg-options data.train.ann_file=${TRAIN_ROOT}.json

# CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_PATH}/dist_test.sh \
#     ${CONFIG_FILE} \
#     ${WORK_DIR}_all/latest.pth \
#     ${N_GPU} \
#     --eval bbox \
#     --out ${WORK_DIR}_all/test2.pkl 2>&1 | tee ${RES_ROOT}_all.txt

# rm ${WORK_DIR}_all/*.log
# rm ${WORK_DIR}_all/epoch_[1-9].pth
# rm ${WORK_DIR}_all/epoch_10.pth
# rm ${WORK_DIR}_all/epoch_11.pth
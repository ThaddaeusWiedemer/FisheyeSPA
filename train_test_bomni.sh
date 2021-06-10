TOOL_PATH=mmdetection/tools
CONFIG_FILE=mmdetection/configs/fisheye/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person_piropo.py
WORK_DIR=work_dirs/BOMNI/training
TRAIN_ROOT=/data/Bomni-DB/train
TEST_FILE=/data/Bomni-DB/test.json
RES_ROOT=results/BOMNI/test
N_GPU=2
VIS_GPU=1,3
GPU_PORT=29501

# no finetuning
# mkdir work_dirs/BOMNI
# mkdir ${WORK_DIR}_0
# mkdir results/BOMNI

CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_PATH}/dist_test.sh \
    ${CONFIG_FILE} \
    mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth \
    ${N_GPU} \
    --eval bbox \
    --cfg-options data.test.ann_file=${TEST_FILE} \
    --out ${WORK_DIR}_0/test_0.pkl 2>&1 | tee ${RES_ROOT}_0.txt

# finetuning on n samples per dataset, with 10 datasets labeled a through j per n
# for n in {1,2,5,10,20,50,100}; do
#     for x in {a..j}; do
#         mkdir ${WORK_DIR}_${n}${x}

#         # train
#         CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_PATH}/dist_train.sh \
#             ${CONFIG_FILE} \
#             ${N_GPU} \
#             --work-dir ${WORK_DIR}_${n}${x} \
#             --cfg-options data.train.ann_file=${TRAIN_ROOT}_${n}${x}.json data.val.ann_file=${TEST_FILE}

#         # test
#         CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_PATH}/dist_test.sh \
#             ${CONFIG_FILE} \
#             ${WORK_DIR}_${n}${x}/latest.pth \
#             ${N_GPU} \
#             --eval bbox \
#             --cfg-options data.test.ann_file=${TEST_FILE} \
#             --out ${WORK_DIR}_${n}${x}/test_${n}${x}.pkl 2>&1 | tee ${RES_ROOT}_${n}${x}.txt

#         # free up space
#         rm ${WORK_DIR}_${n}${x}/*.log
#         rm ${WORK_DIR}_${n}${x}/epoch_[1-9].pth
#         rm ${WORK_DIR}_${n}${x}/epoch_10.pth
#         rm ${WORK_DIR}_${n}${x}/epoch_11.pth
#     done
# done

# finetune once on whole dataset
# CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_PATH}/dist_train.sh \
#     ${CONFIG_FILE} \
#     ${N_GPU} \
#     --work-dir ${WORK_DIR}_all \
#     --cfg-options data.train.ann_file=${TRAIN_ROOT}.json data.val.ann_file=${TEST_FILE}

# CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_PATH}/dist_test.sh \
#     ${CONFIG_FILE} \
#     ${WORK_DIR}_all/latest.pth \
#     ${N_GPU} \
#     --eval bbox \
#     --cfg-options data.test.ann_file=${TEST_FILE} \
#     --out ${WORK_DIR}_all/test.pkl 2>&1 | tee ${RES_ROOT}_all.txt

# rm ${WORK_DIR}_all/*.log
# rm ${WORK_DIR}_all/epoch_[1-9].pth
# rm ${WORK_DIR}_all/epoch_10.pth
# rm ${WORK_DIR}_all/epoch_11.pth
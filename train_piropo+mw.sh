TOOL_PATH=mmdetection/tools
CONFIG_FILE=mmdetection/configs/fisheye/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person_piropo.py
WORK_DIR=work_dirs/PIROPO+MW
MODEL_DIR=work_dirs/PIROPO/training
RES_ROOT=results/PIROPO+MW/sweep
N_GPU=4
VIS_GPU=4,5,6,7
GPU_PORT=29501

mkdir ${WORK_DIR}
mkdir ${RES_ROOT}

# train on combined dataset and test on both individual sets
# CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_PATH}/dist_train.sh \
#     ${CONFIG_FILE} \
#     ${N_GPU} \
#     --work-dir ${WORK_DIR} \
#     --cfg-options data.train.ann_file=data/piropo+mw_train.json

CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_PATH}/dist_test.sh \
    ${CONFIG_FILE} \
    ${WORK_DIR}/latest.pth \
    ${N_GPU} \
    --eval bbox \
    --cfg-options data.test.ann_file=data/PIROPO/omni_test2.json \
    --out ${WORK_DIR}/PIROPO_omni_test2.pkl 2>&1 | tee ${RES_ROOT}/PIROPO_omni_test2.txt

CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_PATH}/dist_test.sh \
    ${CONFIG_FILE} \
    ${WORK_DIR}/latest.pth \
    ${N_GPU} \
    --eval bbox \
    --cfg-options data.test.ann_file=data/MW-18Mar/test.json \
    --out ${WORK_DIR}/MW-18Mar_test.pkl 2>&1 | tee ${RES_ROOT}/MW-18Mar_test.txt

# rm ${WORK_DIR}/*.log
# rm ${WORK_DIR}/epoch_[1-9].pth
# rm ${WORK_DIR}/epoch_10.pth
# rm ${WORK_DIR}/epoch_11.pth

TOOL_DIR=mmdetection/tools

# MODEL and OUTPUT
# CONFIG_FILE=mmdetection/configs/sweeps/fine-tune.py
# RES_FILE=results/sweeps/baseline.log
CONFIG_FILE=mmdetection/configs/sweeps/fine-tune.py
RES_FILE=results/sweeps_mw/baseline.log

# DATA
# TEST_ANNS=/data/PIROPO/omni_test2.json
# TEST_PRE=None
TEST_ANNS=/data/MW-18Mar/test.json
TEST_PRE=None

# GPUs
N_GPU=2
VIS_GPU=1,2
GPU_PORT=29503

# TEST
CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_DIR}/dist_test.sh \
    ${CONFIG_FILE} \
    'mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth' \
    ${N_GPU} \
    --eval bbox \
    --cfg-options data.test.ann_file=${TEST_ANNS} data.test.img_prefix=${TEST_PRE} \
    2>&1 | tee ${RES_FILE}

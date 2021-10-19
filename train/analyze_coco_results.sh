# dirs
TOOL_DIR=mmdetection/tools
CONFIG_FILE=mmdetection/configs/sweeps/fine-tune.py
TEST_FILE=/data/PIROPO/omni_test2.json
MODEL_DIR=/data/thaddaus/WORK_DIRS/sweeps/fine-tune/20a
RES_DIR=results/analysis/final_fine-tune

mkdir -p ${RES_DIR}

# GPUs
N_GPU=4 # if this is changed, the number of iterations also needs to be adapted
VIS_GPU=0,1,2,3
GPU_PORT=29503
BATCH=16

# run test with best epoch and save result
MODEL=$(ls ${MODEL_DIR}/best_bbox_mAP_50_epoch_*.pth)
# MODEL=mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth

CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_DIR}/dist_test.sh \
${CONFIG_FILE} \
${MODEL} \
${N_GPU} \
--format-only \
--options "jsonfile_prefix=${RES_DIR}/" \
--cfg-options data.samples_per_gpu=$(($BATCH/$N_GPU)) \
    data.val.ann_file=${TEST_FILE} \
    data.val.img_prefix=None \
    data.test.ann_file=${TEST_FILE} \
    data.test.img_prefix=None \

# run analyis on result file
python results/coco_error_analysis_fisheye.py \
    ${RES_DIR}/.bbox.json \
    ${RES_DIR} \
    --ann=${TEST_FILE}

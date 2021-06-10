TOOL_PATH=mmdetection/tools
CONFIG_FILE=mmdetection/configs/fisheye/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person_piropo.py
WORK_DIR=work_dirs/blub
MODEL_DIR=work_dirs/MW-18Mar/training
TEST_FILE=/data/Bomni-DB/test.json
RES_ROOT=results/MW_BOMNI/test
N_GPU=1
VIS_GPU=7
GPU_PORT=29502

mkdir results/MW_BOMNI

# finetuning on n samples per dataset, with 10 datasets labeled a through j per n
for n in {1,2,5,10,20,50,100,200}; do
    for x in {a..j}; do
        # test
        CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_PATH}/dist_test.sh \
            ${CONFIG_FILE} \
            ${MODEL_DIR}_${n}${x}/latest.pth \
            ${N_GPU} \
            --eval bbox \
            --cfg-options data.test.ann_file=${TEST_FILE} \
            --out ${WORK_DIR}/test_${n}${x}.pkl 2>&1 | tee ${RES_ROOT}_${n}${x}.txt
    done
done

# finetune once on whole dataset
CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_PATH}/dist_test.sh \
    ${CONFIG_FILE} \
    ${MODEL_DIR}_all/latest.pth \
    ${N_GPU} \
    --eval bbox \
    --cfg-options data.test.ann_file=${TEST_FILE} \
    --out ${WORK_DIR}/test.pkl 2>&1 | tee ${RES_ROOT}_all.txt

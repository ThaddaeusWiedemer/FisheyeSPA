TOOL_DIR=mmdetection/tools

# MODEL
SETTING=adv_0_gpa_-xg
CONFIG_FILE=mmdetection/configs/sweeps/${SETTING}.py

# GPUs
N_GPU=2
VIS_GPU=1,2
GPU_PORT=29503

# T R A I N   O N   M I R R O R W O R L D   T E S T   O N   P I R O P O
# PATHS
MODEL_DIR= # path to model checkpoint
RES_DIR=results/sweeps/${SETTING}_mw
mkdir -p ${RES_DIR}

# DATA
TEST_ANNS=/data/PIROPO/omni_test2.json
TEST_PRE=None

for n in {1,2,5,10,20,50,100}
do
    for x in {a..c}
    do
        model=$(ls ${MODEL_DIR}/$n$x/best_bbox_mAP_50_epoch_*.pth)

        # TEST
        CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_DIR}/dist_test.sh \
            ${CONFIG_FILE} \
            ${model} \
            ${N_GPU} \
            --eval bbox \
            --cfg-options data.test.ann_file=${TEST_ANNS} data.test.img_prefix=${TEST_PRE} \
            2>&1 | tee ${RES_DIR}/$n$x.log

    done
done



# T R A I N   O N   P I R O P O   T E S T   O N   M I R R O R W O R L D
# PATHS
MODEL_DIR= # path to model checkpoint
RES_DIR=results/sweeps_mw/${SETTING}_piropo
mkdir -p ${RES_DIR}

# DATA
TEST_ANNS=/data/MW-18Mar/test.json
TEST_PRE=None

for n in {1,2,5,10,20,50,100}
do
    for x in {a..c}
    do
        model=$(ls ${MODEL_DIR}/$n$x/best_bbox_mAP_50_epoch_*.pth)

        # TEST
        CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_DIR}/dist_test.sh \
            ${CONFIG_FILE} \
            ${model} \
            ${N_GPU} \
            --eval bbox \
            --cfg-options data.test.ann_file=${TEST_ANNS} data.test.img_prefix=${TEST_PRE} \
            2>&1 | tee ${RES_DIR}/$n$x.log

    done
done
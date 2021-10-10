TOOL_DIR=mmdetection/tools

# MODEL and OUTPUT
CONFIG_FILE=mmdetection/configs/sweeps/fine-tune.py
WORK_DIR=WORK_DIRS/sweeps/fine-tune
RES_DIR=results/sweeps/fine-tune
mkdir -p ${WORK_DIR}
mkdir -p ${RES_DIR}

# DATA
DATA_ANNS=/data/PIROPO/omni_training
DATA_PRE=None
TEST_ANNS=/data/PIROPO/omni_test2.json
TEST_PRE=None

# GPUs
N_GPU=4
VIS_GPU=0,1,2,3
GPU_PORT=29502

# TRAINING
EPOCHS=40
BATCH=16

# fine-tuning on n samples per dataset, with x datasets labeled a through j each
for n in {20,50,100,200,500,1000}
do
    for x in {a..c}
    do
        mkdir -p ${WORK_DIR}/${n}${x}

        # train
        CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_DIR}/dist_train.sh \
        ${CONFIG_FILE} \
        ${N_GPU} \
        --work-dir ${WORK_DIR}/${n}${x} \
        --cfg-options data.samples_per_gpu=$(($BATCH/$N_GPU)) \
            data.train.ann_file=${DATA_ANNS}_${n}${x}.json \
            data.train.img_prefix=${DATA_PRE} \
            data.val.ann_file=${TEST_ANNS} \
            data.val.img_prefix=${TEST_PRE} \
            runner.max_epochs=${EPOCHS} \
            2>&1 | tee ${RES_DIR}/${n}${x}.log
    done
done

# fine-tune once on whole dataset
mkdir ${WORK_DIR}/all

CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_PATH}/dist_train.sh \
    ${CONFIG_FILE} \
    ${N_GPU} \
    --work-dir ${WORK_DIR}/all \
    --cfg-options data.samples_per_gpu=$(($BATCH/$N_GPU)) \
        data.train.ann_file=${DATA_ANNS}.json \
        data.train.img_prefix=${DATA_PRE} \
        data.val.ann_file=${TEST_ANNS} \
        data.val.img_prefix=${TEST_PRE} \
        runner.max_epochs=${EPOCHS} \
        2>&1 | tee ${RES_DIR}/all.log
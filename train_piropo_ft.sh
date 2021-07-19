TOOL_PATH=mmdetection/tools
CONFIG_FILE=mmdetection/configs/fisheye/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person_piropo_ft.py
WORK_DIR=work_dirs/PIROPO/sweep_ft/training
TRAIN_ROOT=/data/PIROPO/omni_training
RES_ROOT=results/PIROPO/sweep_ft/test2
N_GPU=4 # if this is changed, the number of iterations also needs to be adapted
VIS_GPU=4,5,6,7
GPU_PORT=29501

# no finetuning
# python ${TOOL_PATH}/test.py \
#     ${CONFIG_FILE} \
#     mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth \
#     --eval bbox \
#     --out ${WORK_DIR}_0/test2_0.pkl 2>&1 | tee ${RES_ROOT}_0.txt

# finetuning on n samples per dataset, with 10 datasets labeled a through j per n
for n in {1,2,5,10,20,50,100,200,500,1000}
do
    for x in {a..j}
    do
        mkdir ${WORK_DIR}_${n}${x}

        # train
        CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_PATH}/dist_train.sh \
            ${CONFIG_FILE} \
            ${N_GPU} \
            --work-dir ${WORK_DIR}_${n}${x} \
            --cfg-options data.train.ann_file=${TRAIN_ROOT}_${n}${x}.json

        # test
        CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_PATH}/dist_test.sh \
            ${CONFIG_FILE} \
            ${WORK_DIR}_${n}${x}/latest.pth \
            ${N_GPU} \
            --eval bbox \
            --out ${WORK_DIR}_${n}${x}/test2_${n}${x}.pkl 2>&1 | tee ${RES_ROOT}_${n}${x}.txt

        # free up space
        # rm ${WORK_DIR}_${n}${x}/*.log
        # rm ${WORK_DIR}_${n}${x}/epoch_[1-9].pth
        # rm ${WORK_DIR}_${n}${x}/epoch_10.pth
        # rm ${WORK_DIR}_${n}${x}/epoch_11.pth
    done
done

# finetune once on whole dataset
mkdir ${WORK_DIR}_all

CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_PATH}/dist_train.sh \
    ${CONFIG_FILE} \
    ${N_GPU} \
    --work-dir ${WORK_DIR}_all \
    --cfg-options data.train.ann_file=${TRAIN_ROOT}.json

CUDA_VISIBLE_DEVICES=${VIS_GPU} PORT=${GPU_PORT} ./${TOOL_PATH}/dist_test.sh \
    ${CONFIG_FILE} \
    ${WORK_DIR}_all/latest.pth \
    ${N_GPU} \
    --eval bbox \
    --out ${WORK_DIR}_all/test2.pkl 2>&1 | tee ${RES_ROOT}_all.txt

# rm ${WORK_DIR}_all/*.log
# rm ${WORK_DIR}_all/epoch_[1-9].pth
# rm ${WORK_DIR}_all/epoch_10.pth
# rm ${WORK_DIR}_all/epoch_11.pth
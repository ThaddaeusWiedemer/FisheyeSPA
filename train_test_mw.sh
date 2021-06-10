TOOL_PATH=mmdetection/tools
CONFIG_FILE=mmdetection/configs/fisheye/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person_piropo.py
WORK_DIR=work_dirs/MW-18Mar/training
TRAIN_ROOT=/data/MW-18Mar/train
TEST_FILE=/data/MW-18Mar/test.json
RES_ROOT=results/MW-18Mar/test
N_GPU=4

# no finetuning
# mkdir ${WORK_DIR}_0

# python ${TOOL_PATH}/test.py \
#     ${CONFIG_FILE} \
#     mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth \
#     --eval bbox \
#     --cfg-options data.test.ann_file=${TEST_FILE} \
#     --out ${WORK_DIR}_0/test_0.pkl 2>&1 | tee ${RES_ROOT}_0.txt

# finetuning on n samples per dataset, with 10 datasets labeled a through j per n
for n in {1,2,5,10,20,50,100,200}; do
    for x in {a..j}; do
        mkdir ${WORK_DIR}_${n}${x}

        # train
        bash ${TOOL_PATH}/dist_train.sh \
            ${CONFIG_FILE} \
            ${N_GPU} \
            --work-dir ${WORK_DIR}_${n}${x} \
            --cfg-options data.train.ann_file=${TRAIN_ROOT}_${n}${x}.json data.val.ann_file=${TEST_FILE}

        # test
        bash ${TOOL_PATH}/dist_test.sh \
            ${CONFIG_FILE} \
            ${WORK_DIR}_${n}${x}/latest.pth \
            ${N_GPU} \
            --eval bbox \
            --cfg-options data.test.ann_file=${TEST_FILE} \
            --out ${WORK_DIR}_${n}${x}/test2_${n}${x}.pkl 2>&1 | tee ${RES_ROOT}_${n}${x}.txt

        # free up space
        rm ${WORK_DIR}_${n}${x}/*.log
        rm ${WORK_DIR}_${n}${x}/epoch_[1-9].pth
        rm ${WORK_DIR}_${n}${x}/epoch_10.pth
        rm ${WORK_DIR}_${n}${x}/epoch_11.pth
    done
done

# finetune once on whole dataset
bash ${TOOL_PATH}/dist_train.sh \
    ${CONFIG_FILE} \
    ${N_GPU} \
    --work-dir ${WORK_DIR}_all \
    --cfg-options data.train.ann_file=${TRAIN_ROOT}.json data.val.ann_file=${TEST_FILE}

bash ${TOOL_PATH}/dist_test.sh \
    ${CONFIG_FILE} \
    ${WORK_DIR}_all/latest.pth \
    ${N_GPU} \
    --eval bbox \
    --cfg-options data.test.ann_file=${TEST_FILE} \
    --out ${WORK_DIR}_all/test2.pkl 2>&1 | tee ${RES_ROOT}_all.txt

rm ${WORK_DIR}_all/*.log
rm ${WORK_DIR}_all/epoch_[1-9].pth
rm ${WORK_DIR}_all/epoch_10.pth
rm ${WORK_DIR}_all/epoch_11.pth
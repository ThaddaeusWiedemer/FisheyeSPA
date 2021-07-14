# PIROPO
## download
Download complete dataset, but only unzip data from omnidirectional cameras, as we're only using that.

```bash
wget -O /data/PIROPO/conv_5A.zip 'https://drive.upm.es/index.php/s/YF2JUrw33wtRMIj/download?path=%2F&files=conv_5A.zip'
wget -O /data/PIROPO/conv_6A.zip 'https://drive.upm.es/index.php/s/YF2JUrw33wtRMIj/download?path=%2F&files=conv_6A.zip'
wget -O /data/PIROPO/conv_7A.zip 'https://drive.upm.es/index.php/s/YF2JUrw33wtRMIj/download?path=%2F&files=conv_7A.zip'
wget -O /data/PIROPO/conv_8A.zip 'https://drive.upm.es/index.php/s/YF2JUrw33wtRMIj/download?path=%2F&files=conv_8A.zip'
wget -O /data/PIROPO/omni_1A.zip 'https://drive.upm.es/index.php/s/YF2JUrw33wtRMIj/download?path=%2F&files=omni_1A.zip'
wget -O /data/PIROPO/omni_2A.zip 'https://drive.upm.es/index.php/s/YF2JUrw33wtRMIj/download?path=%2F&files=omni_2A.zip'
wget -O /data/PIROPO/omni_3A.zip 'https://drive.upm.es/index.php/s/YF2JUrw33wtRMIj/download?path=%2F&files=omni_3A.zip'
wget -O /data/PIROPO/conv_2B.zip 'https://drive.upm.es/index.php/s/YoWW0gkemWNZ3AL/download?path=%2F&files=conv_2B.zip'
wget -O /data/PIROPO/omni_1B.zip 'https://drive.upm.es/index.php/s/YoWW0gkemWNZ3AL/download?path=%2F&files=omni_1B.zip'

cd /data/PIROPO
unzip omni_1A.zip
unzip omni_1B.zip
unzip omni_2A.zip
unzip omni_3A.zip
```

## download annotations
Download annotations (only available for `test2`, `test3`, and `training` sequence) and unzip them in their own folder,
since they also contain annotations for other datasets.

```bash
cd /data
git clone git@github.com:hitachi-rd-cv/omnidet-rotinv.git
mv omnidet-rotinv hitachi_annotations

cd /data/hitachi_annotations
tar -xzvf normal.tar.gz
```

## convert annotations to COCO-style
The converter script needs a list of all images with available annotation, which we generate from the annotation files.
Then we can run the converter script (in `mmdetection/tools/dataset_converters`) for each subset (e.g. 
`PIROPO/omni_1A/omni1A_test2`).

```bash
declare -a dirs=("1A" "1B" "2A" "3A")
declare -a splits=("training" "test2" "test3")

# save list of images with available annotation as 'images.txt'
for dir in ${dirs[@]}; do
    for split in ${splits[@]}; do
        ls /data/hitachi_annotations/normal/piropo/Room_${dir:1:1}/omni_$dir/omni_${dir}_$split | sed -e 's/\.xml$//' > /data/PIROPO/omni_$dir/omni${dir}_$split/images.txt
    done
done

# use converter script to generate COCO annotations
for dir in ${dirs[@]}; do
    for split in ${splits[@]}; do
        python mmdetection/tools/dataset_converters/hitachi_to_coco.py \
            /data/PIROPO/omni_$dir/omni${dir}_$split \
            /data/hitachi_annotations/normal/piropo/Room_${dir:1:1}/omni_$dir/omni_${dir}_$split \
            --out-dir /data/PIROPO/omni_$dir/omni${dir}_$split\
            --out-format coco
    done
done
```

Finally, we can use the script in `mmdetection/tools/misc/coco_concat.py` to concatenate several subsequences to larger
datasets. These are just saved as `PIROPO/omni_<sequence name>.json` and link to the images.

The script can also be used to generate multiple splits of different sizes. These are saved as 
`PIROPO/omni_<sequence name>_<size><suffix>.json`, where `<suffix>` is alphabetically enumerating the different random 
splits.

## testing
A COCO-pretrained model for person detection is run to test the datasets:

```bash
python mmdetection/tools/test.py \
    mmdetection/configs/fisheye/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person_piropo.py \
    mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth \
    --eval bbox \
    --out results/piropo_test.pkl
```

# Mirror Worlds
## download
Then unzip complete data.

```bash
wget -O /data/MW-18Mar/MWAll.zip 'http://www2.icat.vt.edu/mirrorworlds-videos/MW-18Mar/MWAll.zip'
wget -O /data/MW-18Mar/MWLabels_MOT.zip 'http://www2.icat.vt.edu/mirrorworlds-videos/MW-18Mar/MWLabels(MOT).zip'
wget -O /data/MW-18Mar/RawTrainLabels.zip 'http://www2.icat.vt.edu/mirrorworlds-videos/MW-18Mar/RawTrainLabels.zip'
wget -O /data/MW-18Mar/RawVideos.zip 'http://www2.icat.vt.edu/mirrorworlds-videos/MW-18Mar/RawVideos.zip'

cd /data/MW-18Mar
unzip MWAll.zip
```

## annotations
While the datasets comes with horizontal bounding boxes annotated per frame, there is very little movement of objects
between adjacents frames. The Hitachi Annotations seem to have sampled the available images and additionally provide
annotations for the test sequences, which the original dataset don't. While this results in less annotated training
images, we go with these annotations for consistency.

```bash
declare -a dirs=("Test/MW-18Mar-1" "Test/MW-18Mar-4" "Test/MW-18Mar-5" "Test/MW-18Mar-6" "Test/MW-18Mar-9" \
                 "Test/MW-18Mar-11" "Test/MW-18Mar-15" "Test/MW-18Mar-16" "Test/MW-18Mar-20" "Test/MW-18Mar-28" \
                 "Test/MW-18Mar-29" "Test/MW-18Mar-30" "Train/MW-18Mar-2" "Train/MW-18Mar-3" "Train/MW-18Mar-7" \
                 "Train/MW-18Mar-8" "Train/MW-18Mar-10" "Train/MW-18Mar-12" "Train/MW-18Mar-13" "Train/MW-18Mar-14" \
                 "Train/MW-18Mar-17" "Train/MW-18Mar-18" "Train/MW-18Mar-19" "Train/MW-18Mar-21" "Train/MW-18Mar-22" \
                 "Train/MW-18Mar-23" "Train/MW-18Mar-24" "Train/MW-18Mar-25" "Train/MW-18Mar-26" "Train/MW-18Mar-27" \
                 "Train/MW-18Mar-31")

# save list of images with available annotation as 'images.txt'
for dir in ${dirs[@]}; do
    ls /data/hitachi_annotations/normal/mw_18mar/${dir}/gt | sed -e 's/\.xml$//' > /data/MW-18Mar/MWAll/${dir}/img1/images.txt
done

# use converter script to generate COCO annotations
for dir in ${dirs[@]}; do
    python mmdetection/tools/dataset_converters/hitachi_to_coco.py \
        /data/MW-18Mar/MWAll/${dir}/img1 \
        /data/hitachi_annotations/normal/mw_18mar/${dir}/gt \
        --out-dir /data/MW-18Mar/MWAll/${dir}/img1 \
        --ending .png \
        --out-format coco
done
```

## splits
We use `mmdetection/tools/misc/coco_concat.py` to concat the subsets to a `train` and `test` sequence in 
`MW-18Mar/<sequence name>.json`. Different splits of the `train` sequence are also saved as
`<sequence name>_<size><suffix>.json`.

# Bomni-DB
## download
```bash
cd /data/Bomni-DB
unzip bomni-5841.zip
```

## annotations
All video sequences are annotated in the original dataset. However, since these annotations don't vary a lot per frame,
the sampling of every 10th frame as in the hitachi annotations seems useful. For now, only the annotations from there
are used. These only include sequences with top views.

```bash
# make folders for images
for i in {0..3}; do
    mkdir /data/Bomni-DB/scenario1/top-${i}
done

# save list of images with available annotation as 'images.txt'
for i in {0..3}; do
    ls /data/hitachi_annotations/normal/bomni/normal/scenario1/top-${i} | sed -e 's/\.xml$//' > /data/Bomni-DB/scenario1/top-${i}/images.txt
done

# iterate over image list and extract corresponding frames as lossless images from the mp4 video
# the timestamps for each specific frame is listed in the corresponding .info file
SCEN1_ROOT=data/BOMNI/scenario1
for i in {0..3}; do
    while read p; do
        # make sure image id is interpreted as decimal, not octal, despite leading 0, and bring to 4 digits with
        # leading zeroes. then subtract one because bomni frames are 0 based
        printf -v img %04i $(( 10#$p-1 ))
        # extract time stamp for frame
        ts=$(awk '$1==a {printf "%09.6f", $2}' a=$img ${SCEN1_ROOT}/top-${i}.info)
        # extract frame from video as bitmap (since it is lossless)
        printf -v img %04i $(( 10#$p ))
        ffmpeg -ss ${ts} -i ${SCEN1_ROOT}/top-${i}.mp4 -frames:v 1 ${SCEN1_ROOT}/top-${i}/${img}.bmp
    done < ${SCEN1_ROOT}/top-${i}/images.txt
done

i=0
img=1000
ts=$(awk '$1==a {printf "%09.6f", $2}' a=$img ${SCEN1_ROOT}/top-${i}.info)
img=1001
ffmpeg -ss ${ts} -i ${SCEN1_ROOT}/top-${i}.mp4 -frames:v 1 ${SCEN1_ROOT}/top-${i}/${img}.bmp

# use converter script to generate COCO annotations
for i in {0..3}; do
    python mmdetection/tools/dataset_converters/hitachi_to_coco.py \
        /data/Bomni-DB/scenario1/top-${i} \
        /data/hitachi_annotations/normal/bomni/normal/scenario1/top-${i} \
        --out-dir /data/Bomni-DB/scenario1/top-${i} \
        --ending .bmp \
        --out-format coco
done
```

## splits
We use `mmdetection/tools/misc/coco_concat.py` to concat the subsets to a `train` sequence (`top-1`, `top-2`, `top-3`)
and `test` sequence (`top-0`) in `Bomni-DB/<sequence name>.json`. Different splits of the `train` sequence are also
saved as `<sequence name>_<size><suffix>.json`.

# COCO
```bash
wget -O /data/COCO/train2017.zip 'http://images.cocodataset.org/zips/train2017.zip'
wget -O /data/COCO/val2017.zip 'http://images.cocodataset.org/zips/val2017.zip'
wget -O /data/COCO/test2017.zip 'http://images.cocodataset.org/zips/test2017.zip'
wget -O /data/COCO/trainval2017.zip 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'

cd /data/COCO
unzip train2017.zip
unzip test2017.zip
unzip trainval2017.zip
```

## COCO person
Create an annotation file with all ground-truths other than `person` removed and remove all annotations which don't
contain the `person` class to begin with.
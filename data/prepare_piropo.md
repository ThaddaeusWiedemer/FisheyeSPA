# download PIROPO
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

# download annotations
Download annotations (only available for `test2`, `test3`, and `training` sequence) and unzip them in their own folder,
since they also contain annotations for other datasets.

```bash
cd /data
git clone git@github.com:hitachi-rd-cv/omnidet-rotinv.git
mv omnidet-rotinv hitachi_annotations

cd /data/hitachi_annotations
tar -xzvf normal.tar.gz
```

# convert annotations to COCO-style
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
        python mmdetection/tools(dataset_converters/piropo_to_coco.py \
            /data/PIROPO/omni_$dir/omni${dir}_$split \
            /data/hitachi_annotations/normal/piropo/Room_${dir:1:1}/omni_$dir/omni_${dir}_$split \
            --out-dir /data/PIROPO/omni_$dir/omni${dir}_$split\
            --out-format coco
    done
done
```

Finally, we can use the script in `mmdetection/tools/misc/coco_concat.py` to concatenate several subsequences to larger
datasets. These are just saved as `PIROPO/omni_<sequence name>.json` and link to the images.

# testing
A COCO-pretrained model for person detection is run to test the datasets:

```bash
python mmdetection/tools/test.py \
    mmdetection/configs/fisheye/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person_piropo.py \
    mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth \
    --eval bbox
```
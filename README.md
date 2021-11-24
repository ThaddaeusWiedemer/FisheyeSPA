# Few-Shot Supervised Prototype Alignment for Pedestrian Detection on Fisheye Images

## Folder structure
```
.
├─ data: Download the datasets to here. Annotations are already provided.
├─ results: Scripts and Jupyter notebooks to analyze results.
├─ mmdetection: All models and config files.
├─ sweeps: Scripts to run train/test sweeps over multiple dataset sizes.
└─ train_test: Scripts to train/test on individual dataset with different parameters.
```

## Setting up
Download and unzip data into `data` folder. Unzip the corresponding annotation files into respective folders.

### PIROPO
```bash
wget -O data/PIROPO/omni_1A.zip 'https://drive.upm.es/index.php/s/YF2JUrw33wtRMIj/download?path=%2F&files=omni_1A.zip'
wget -O data/PIROPO/omni_2A.zip 'https://drive.upm.es/index.php/s/YF2JUrw33wtRMIj/download?path=%2F&files=omni_2A.zip'
wget -O data/PIROPO/omni_3A.zip 'https://drive.upm.es/index.php/s/YF2JUrw33wtRMIj/download?path=%2F&files=omni_3A.zip'
wget -O data/PIROPO/omni_1B.zip 'https://drive.upm.es/index.php/s/YoWW0gkemWNZ3AL/download?path=%2F&files=omni_1B.zip'

cd data/PIROPO
unzip omni_1A.zip
unzip omni_1B.zip
unzip omni_2A.zip
unzip omni_3A.zip
```

### Mirror Worlds
```bash
wget -O data/MW-18Mar/MWAll.zip 'http://www2.icat.vt.edu/mirrorworlds-videos/MW-18Mar/MWAll.zip'
wget -O data/MW-18Mar/MWLabels_MOT.zip 'http://www2.icat.vt.edu/mirrorworlds-videos/MW-18Mar/MWLabels(MOT).zip'
wget -O data/MW-18Mar/RawTrainLabels.zip 'http://www2.icat.vt.edu/mirrorworlds-videos/MW-18Mar/RawTrainLabels.zip'
wget -O data/MW-18Mar/RawVideos.zip 'http://www2.icat.vt.edu/mirrorworlds-videos/MW-18Mar/RawVideos.zip'

cd data/MW-18Mar
unzip MWAll.zip
```

### COCO
```bash
wget -O data/COCO/train2017.zip 'http://images.cocodataset.org/zips/train2017.zip'

cd /data/COCO
unzip train2017.zip
```
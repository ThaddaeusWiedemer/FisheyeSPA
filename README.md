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

### Pre-Trained Model
Download the COCO-person pre-trained model from MMDetection:

```
https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth
```

Then run `train_test/split_config.py` to generate a version where the shared head is split into classification and bounding-box regression.

## Running the Model
Run `fine-tune.sh` and `adaptive.sh` in `sweeps/` to train models on training sets with sizes 1 to 100 and log the results.
Uncomment the corresponding sections in these scripts to switch datasets or model configuration.

Run `baseline.sh` to test the COCO pre-trained model on the fisheye datasets.

Additionally, `cross_test.sh` can be used to replicate the experiments with training on PIROPO and testing on Mirror Worlds (and vice-versa)

### Experiments on Individual Datasets
`train_test/adapt_coco_piropo.sh` can be used to train a model on a single training set. The script takes 5 arguments:
number of epochs, training set size, training set split, model definition, and output name. Additionally, individual configuration
parameters can be overwritten.

For example, run

```
train_test/adapt_coco_piropo.sh 80 20 a TwoStageDetectorDA more_epochs
```

to train and test the final model on PIROPO-20a for 80 epochs (instead of the 40 used in other experiments). To overwrite
configuration parameters, pass their name and value as additional parameters. E.g., to change the sample size for adversarial
adaptation, run

```
train_test/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA smaller_sample_size model.train_cfg.da.0.sample_shape=9
```

## Analyzing Results
You can use the `paper.ipynb` Jupyter notebook to recreate plots as shown in the paper based on the training/testing logs.
Similarly, running `analysis.sh` with the training/testing logs will reproduce the analyis of results by object characteristics size, distant, and angle.
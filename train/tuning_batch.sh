# parameters are: EPOCHS - SPLITSIZE - SPLIT - DA - DA_ROI - DA_RCNN - DA_INTRA - DA_INTER - DISTANCE - NORMALIZE - GT - SUFFIX
cd ~/MasterthesisCode
# train/_gpa_coco_piropo_tuning.sh 40 100 a 0.1   0   1   1   0 cosine       # test previous best setting
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1   1    1    1   1 mean_squared False True  # test ground truth prototypes
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1   1    1    1   1 mean_squared True  False # test prototype normalization
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1   1    1    1   1 mean_squared False False # test standard
# train/_gpa_coco_piropo_tuning.sh 40 100 a 0.1   1    1    1   1 mean_squared False False # test standard x 0.1
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1   1    1    1   1 euclidean    False False # test euclidean
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1   1    1    1   1 cosine       False False # test cosine
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1   1    0    1   1 mean_squared False False # test ROI
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1   0    1    1   1 mean_squared False False # test RCNN
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1   1    1    1   0 mean_squared False False # test intra
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1   1    1    0   1 mean_squared False False # test inter
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1   0   10    1   1 mean_squared False False # test RCNN x 10
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1   0  100    1   1 mean_squared False False # test RCNN x 100
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1   0 1000    1   1 mean_squared False False # test RCNN x 1000
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1   0    1   10   1 mean_squared False False # test intra x 10
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1   0    1  100   1 mean_squared False False # test intra x 100
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1   0    1 1000   1 mean_squared False False # test intra x 1000
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1   0    1   10   0 mean_squared False False # test intra x 10 no inter

# intra x 100 across splits
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1   0    1  100   1 mean_squared False False 1
# train/_gpa_coco_piropo_tuning.sh 40 100 b   1   0    1  100   1 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh 40 100 c   1   0    1  100   1 mean_squared False False 

# check transferability of results to splits of different sizes
# shouldn't need more epochs on larger split
# train/_gpa_coco_piropo_tuning.sh  40 500 a   0   0    0    0   0 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh  40 500 a   1   0    1  100   1 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh  40 500 b   0   0    0    0   0 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh  40 500 b   1   0    1  100   1 mean_squared False False 
# use 7 times as many epochs for roughly same amount of iterations
# train/_gpa_coco_piropo_tuning.sh 280   1 a   0   0    0    0   0 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh 280   1 a   1   0    1  100   1 mean_squared False False
# train/_gpa_coco_piropo_tuning.sh  80  10 a   0   0    0    0   0 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh  80  10 a   1   0    1  100   1 mean_squared False False

# re-run baseline: seeded? random?
# # train/_gpa_coco_piropo_tuning.sh 280   1 a   0   0    0    0   0 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh 280    1 b   0   0    0    0   0 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh 280    1 c   0   0    0    0   0 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh 280    2 a   0   0    0    0   0 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh 280    2 b   0   0    0    0   0 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh 280    2 c   0   0    0    0   0 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh 280    5 a   0   0    0    0   0 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh 280    5 b   0   0    0    0   0 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh 280    5 c   0   0    0    0   0 mean_squared False False 
# # train/_gpa_coco_piropo_tuning.sh 280   10 a   0   0    0    0   0 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh 280   10 b   0   0    0    0   0 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh 280   10 c   0   0    0    0   0 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh 280   20 a   0   0    0    0   0 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh 280   20 b   0   0    0    0   0 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh 280   20 c   0   0    0    0   0 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh 280   50 a   0   0    0    0   0 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh 280   50 b   0   0    0    0   0 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh 280   50 c   0   0    0    0   0 mean_squared False False 
# # train/_gpa_coco_piropo_tuning.sh 280  100 a   0   0    0    0   0 mean_squared False False 
# # train/_gpa_coco_piropo_tuning.sh 280  100 b   0   0    0    0   0 mean_squared False False 
# # train/_gpa_coco_piropo_tuning.sh 280  100 c   0   0    0    0   0 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh 280  200 a   0   0    0    0   0 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh 280  200 b   0   0    0    0   0 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh 280  200 c   0   0    0    0   0 mean_squared False False 
# # train/_gpa_coco_piropo_tuning.sh 280  500 a   0   0    0    0   0 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh 280  500 b   0   0    0    0   0 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh 280  500 c   0   0    0    0   0 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh 280 1000 a   0   0    0    0   0 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh 280 1000 b   0   0    0    0   0 mean_squared False False 
# train/_gpa_coco_piropo_tuning.sh 280 1000 c   0   0    0    0   0 mean_squared False False 

# keep training on source domain - !CHANGE two_stage_adaptive.py BACK TO ONLY USE TARGET LOSS!
# train/_gpa_coco_piropo_tuning.sh 40 100 a 0.1   1    1    1   1 mean_squared False False _trainsrc # test standard x 0.1
# train/_gpa_coco_piropo_tuning.sh 40 100 a 0.1   0    1  100   1 mean_squared False False _trainsrc # test RCNN intra x 100

# replace fc-layer with pooling
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1   0    1  100   1 mean_squared False False apool # replace fc with average pool
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1   0    1  100   1 mean_squared False False mpool # replace fc with average pool
# train/_gpa_coco_piropo_tuning.sh 60 100 a   1   0    1  100   1 mean_squared False False mpool # replace fc with mx pool
# train/_gpa_coco_piropo_tuning.sh 60 100 a   1   0    1  100   1 mean_squared False False _nofc # remove fc

# more combinations based on best reasonable result so far
# train/_gpa_coco_piropo_tuning.sh 40 100 a 0.1   0    1  100    1 mean_squared False False # absolute weight × 0.1
# train/_gpa_coco_piropo_tuning.sh 40 100 a  10   0    1  100    1 mean_squared False False # absolute weight × 10
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1   1    1  100    1 mean_squared False False # ROI + RCNN
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1   1  0.1  100    1 mean_squared False False # ROI + 0.1 RCNN
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1 0.1    1  100    1 mean_squared False False # 0.1 ROI + RCNN
# train/_gpa_coco_piropo_tuning.sh 40 100 a 0.1   1    1  100    1 mean_squared False False # ROI + RCNN × 0.1
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1   1    0  100    1 mean_squared False False # only ROI
# train/_gpa_coco_piropo_tuning.sh 40 100 a 0.1   1    0  100    1 mean_squared False False # only ROI × 0.1
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1   1    0   10  0.1 mean_squared False False # keep relation
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1   1    0    1 0.01 mean_squared False False # keep relation
# train/_gpa_coco_piropo_tuning.sh 60 100 a   1   0    1  100    1 euclidean    False False # euclidean
# train/_gpa_coco_piropo_tuning.sh 60 100 a   1   0    1  100    1 cosine       False False # cosine
# train/_gpa_coco_piropo_tuning.sh 40 100 a 0.1   0    1  100    1 mean_squared True  False # normalized
# train/_gpa_coco_piropo_tuning.sh 40 100 a 0.1   0    1  100    1 mean_squared False True  # ground truth

# # more combinations based on best result with only inter loss
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1   1    0    0   1 mean_squared False False # only ROI
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1   0    1    0   1 mean_squared False False # only RCNN
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1   1    1    0 0.1 mean_squared False False # × 10
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1   1    1    0  10 mean_squared False False # × 0.1

# experiments with new best result
train/_gpa_coco_piropo_tuning.sh 40 100 a 0.1 0.1    1  100    1 mean_squared False False # 0.1 ROI + RCNN × 0.1
train/_gpa_coco_piropo_tuning.sh 40 100 a   1 0.1    1   10  0.1 mean_squared False False # 0.1 ROI + RCNN keep relation
train/_gpa_coco_piropo_tuning.sh 40 100 a   1 0.1    1    1 0.01 mean_squared False False # 0.1 ROI + RCNN keep relation



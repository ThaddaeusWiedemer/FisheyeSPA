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
# train/_gpa_coco_piropo_tuning.sh 40 100 a 0.1 0.1    1  100    1 mean_squared False False # 0.1 ROI + RCNN × 0.1
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1 0.1    1   10  0.1 mean_squared False False # 0.1 ROI + RCNN keep relation
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1 0.1    1    1 0.01 mean_squared False False # 0.1 ROI + RCNN keep relation
# train/_gpa_coco_piropo_tuning.sh 40 100 a   1 0.1    1  100  0.1 mean_squared False False # 0.1 ROI + RCNN + 0.1 inter

# check deterministic=True - !CHANGE _tuning.sh script back to run normally!
# train/_gpa_coco_piropo_tuning.sh 40 100 a 0 0 0 0 0 mean_squared False False 0
# train/_gpa_coco_piropo_tuning.sh 40 100 a 0 0 0 0 0 mean_squared False False 1
# train/_gpa_coco_piropo_tuning.sh 40 100 a 0 0 0 0 0 mean_squared False False 2

# check automatic loss-balancing
# train/_gpa_coco_piropo_tuning.sh 80 100 a   1 1 1 1 1 mean_squared False False _auto # auto-balance DA losses
# train/_gpa_coco_piropo_tuning.sh 80 100 a 0.1 1 1 1 1 mean_squared False False _auto # auto-balance DA losses but weigh × 0.1
# train/_gpa_coco_piropo_tuning.sh 80 100 a   1 1 1 1 1 mean_squared False False _autoall # auto-balance all losses
# train/_gpa_coco_piropo_tuning.sh 80 100 a   1 0.1 1 100 1 mean_squared False False _autoall # auto-balance all losses but weigh with best result so far
# train/_gpa_coco_piropo_tuning.sh 80 100 a   1 1 1 1 0.001 mean_squared False False _autoall # auto-balance all losses but weigh inter-loss × 0.001
# train/_gpa_coco_piropo_tuning.sh 80 100 a   1 1 1 1 1 mean_squared False False _autoallinit # auto-balance all losses and initialize coefficients to 1/num_tasks = 0.2
# train/_gpa_coco_piropo_tuning.sh 80 100 a   1 1 1 1 1 cosine False False _autoallinit # auto-balance all losses and initialize coefficients to 1/num_tasks = 0.2
# train/_gpa_coco_piropo_tuning.sh 80 100 a   1 1 1 1 1 euclidean False False _autoallinit # auto-balance all losses and initialize coefficients to 1/num_tasks = 0.2

# methodically balancing inter <-> intra
# train/_gpa_coco_piropo_tuning.sh 40 100 a 1 0   10    10 1 mean_squared False False # 10:1
# train/_gpa_coco_piropo_tuning.sh 40 100 a 1 0   10   100 1 mean_squared False False # 100:1
# train/_gpa_coco_piropo_tuning.sh 40 100 a 1 0   10  1000 1 mean_squared False False # 1000:1
# train/_gpa_coco_piropo_tuning.sh 40 100 a 1 0  0.1    10 1 mean_squared False False # 10:1
# train/_gpa_coco_piropo_tuning.sh 40 100 a 1 0  0.1   100 1 mean_squared False False # 100:1
# train/_gpa_coco_piropo_tuning.sh 40 100 a 1 0  0.1  1000 1 mean_squared False False # 1000:1
# train/_gpa_coco_piropo_tuning.sh 40 100 a 1 0  0.1 10000 1 mean_squared False False # 1000:1
# train/_gpa_coco_piropo_tuning.sh 40 100 a 1 0 0.01    10 1 mean_squared False False # 10:1
# train/_gpa_coco_piropo_tuning.sh 40 100 a 1 0 0.01   100 1 mean_squared False False # 100:1
# train/_gpa_coco_piropo_tuning.sh 40 100 a 1 0 0.01  1000 1 mean_squared False False # 1000:1
# train/_gpa_coco_piropo_tuning.sh 40 100 a 1 0 0.01 10000 1 mean_squared False False # 10000:1

# methodically balancing ROI <-> RCNN
# train/_gpa_coco_piropo_tuning.sh 40 100 a 1 0.01 1 100 0.1 mean_squared False False # 0.01:1
# train/_gpa_coco_piropo_tuning.sh 40 100 a 1    1 1 100 0.1 mean_squared False False #    1:1
# train/_gpa_coco_piropo_tuning.sh 40 100 a 1   10 1 100 0.1 mean_squared False False #   10:1
# train/_gpa_coco_piropo_tuning.sh 40 100 a 0 0 0 0 0 mean_squared False False _custom # custom weights

# from here, arguments are
# EPOCHS SPLITSIZE SPLIT MODEL ROI_INTRA ROI_INTER RCNN_INTRA RCNN_INTER GRAPH FC SUFFIX

# test auto-balancing again
# train/_gpa_coco_piropo_tuning.sh 40 100 a TwoStageDetectorAdaptiveAutoBalance 10 0.1 100 0.1 True fc_layer

# test ROI individually
# train/_gpa_coco_piropo_tuning.sh 40 100 a TwoStageDetectorAdaptive   0.1 0.01   0 0   True fc_layer
# train/_gpa_coco_piropo_tuning.sh 40 100 a TwoStageDetectorAdaptive  10   0.01   0 0   True fc_layer
# train/_gpa_coco_piropo_tuning.sh 40 100 a TwoStageDetectorAdaptive   1   0.1    0 0   True fc_layer
# train/_gpa_coco_piropo_tuning.sh 40 100 a TwoStageDetectorAdaptive 100   0.1    0 0   True fc_layer
# train/_gpa_coco_piropo_tuning.sh 40 100 a TwoStageDetectorAdaptive  10   1      0 0   True fc_layer

# test fc-layer
# train/_gpa_coco_piropo_tuning.sh 40 100 a TwoStageDetectorAdaptive  10   0.1  100 0.1 True none
# train/_gpa_coco_piropo_tuning.sh 40 100 a TwoStageDetectorAdaptive  10   0.01 100 0.1 True none
# train/_gpa_coco_piropo_tuning.sh 40 100 a TwoStageDetectorAdaptive  10   0.1  100 0.1 True fc_layer .1
# train/_gpa_coco_piropo_tuning.sh 40 100 a TwoStageDetectorAdaptive  10   0.1  100 0.1 True fc_layer _0
# train/_gpa_coco_piropo_tuning.sh 40 100 a TwoStageDetectorAdaptive  10   0.1  100 0.1 True fc_layer _01
# train/_gpa_coco_piropo_tuning.sh 40 100 a TwoStageDetectorAdaptive  10   0.1  100 0.1 True fc_layer_roi
# train/_gpa_coco_piropo_tuning.sh 40 100 a TwoStageDetectorAdaptive  10   0.1  100 0.1 True fc_layer_rcnn
# train/_gpa_coco_piropo_tuning.sh 40 100 a TwoStageDetectorAdaptive  10   0.1  30 0.1 True fc_layer_roi

# test graph-based aggregation
# train/_gpa_coco_piropo_tuning.sh 40 100 a TwoStageDetectorAdaptive  10   0.1  100 0.1 False fc_layer
# train/_gpa_coco_piropo_tuning.sh 40 100 a TwoStageDetectorAdaptive  10   0.01  100 0.1 False fc_layer

# fine-tuning without random rotation
# train/_gpa_coco_piropo_tuning.sh 40 100 a TwoStageDetectorAdaptive 0 0 0 0 True fc_layer
# train/_gpa_coco_piropo_tuning.sh 40 100 a TwoStageDetectorAdaptive 0 0 0 0 True fc_layer _norot # removed random rotation
# train/_gpa_coco_piropo_tuning.sh 40    1 a TwoStageDetectorAdaptive 0 0 0 0 True fc_layer _norot # removed random rotation
# train/_gpa_coco_piropo_tuning.sh 40   10 a TwoStageDetectorAdaptive 0 0 0 0 True fc_layer _norot # removed random rotation
# train/_gpa_coco_piropo_tuning.sh 40   20 a TwoStageDetectorAdaptive 0 0 0 0 True fc_layer _norot # removed random rotation
# train/_gpa_coco_piropo_tuning.sh 40   50 a TwoStageDetectorAdaptive 0 0 0 0 True fc_layer _norot # removed random rotation
# train/_gpa_coco_piropo_tuning.sh 40  100 b TwoStageDetectorAdaptive 0 0 0 0 True fc_layer _norot # removed random rotation
# train/_gpa_coco_piropo_tuning.sh 40  100 c TwoStageDetectorAdaptive 0 0 0 0 True fc_layer _norot # removed random rotation
# train/_gpa_coco_piropo_tuning.sh 40  500 a TwoStageDetectorAdaptive 0 0 0 0 True fc_layer _norot # removed random rotation
# train/_gpa_coco_piropo_tuning.sh 40 1000 a TwoStageDetectorAdaptive 0 0 0 0 True fc_layer _norot # removed random rotation

# fine-tuning with random 90° rotation
# train/_gpa_coco_piropo_tuning.sh 40    1 a TwoStageDetectorAdaptive 0 0 0 0 True fc_layer _rot90 # removed random rotation
# train/_gpa_coco_piropo_tuning.sh 40   10 a TwoStageDetectorAdaptive 0 0 0 0 True fc_layer _rot90 # removed random rotation
# train/_gpa_coco_piropo_tuning.sh 40   20 a TwoStageDetectorAdaptive 0 0 0 0 True fc_layer _rot90 # removed random rotation
# train/_gpa_coco_piropo_tuning.sh 40   50 a TwoStageDetectorAdaptive 0 0 0 0 True fc_layer _rot90 # removed random rotation
# train/_gpa_coco_piropo_tuning.sh 40  100 a TwoStageDetectorAdaptive 0 0 0 0 True fc_layer _rot90 # removed random rotation
# train/_gpa_coco_piropo_tuning.sh 40  500 a TwoStageDetectorAdaptive 0 0 0 0 True fc_layer _rot90 # removed random rotation
# train/_gpa_coco_piropo_tuning.sh 40 1000 a TwoStageDetectorAdaptive 0 0 0 0 True fc_layer _rot90 # removed random rotation


# GPA without random rotation
# train/_gpa_coco_piropo_tuning.sh 40 100 a TwoStageDetectorAdaptive 10 0.1 100 0.1 True fc_layer _norot # removed random rotation
# train/_gpa_coco_piropo_tuning.sh 40 100 a TwoStageDetectorAdaptive 1 1 1 1 True fc_layer _norot # removed random rotation
# train/_gpa_coco_piropo_tuning.sh 40 1 a TwoStageDetectorAdaptive 10 0.1 100 0.1 True fc_layer _norot # removed random rotation
# train/_gpa_coco_piropo_tuning.sh 40 10 a TwoStageDetectorAdaptive 10 0.1 100 0.1 True fc_layer _norot # removed random rotation
# train/_gpa_coco_piropo_tuning.sh 40 20 a TwoStageDetectorAdaptive 10 0.1 100 0.1 True fc_layer _norot # removed random rotation
# train/_gpa_coco_piropo_tuning.sh 40 50 a TwoStageDetectorAdaptive 10 0.1 100 0.1 True fc_layer _norot # removed random rotation

# GPA + adversarial
# all trainings removed random rotation
# train/_gpa_coco_piropo_tuning.sh 40 100 a TwoStageDetectorAdaptiveAdversarial 1 1 1 1 True fc_layer _norot
# train/_gpa_coco_piropo_tuning.sh 40 100 a TwoStageDetectorAdaptiveAdversarial 0.1 0.1 0.1 0.1 True fc_layer _norot
# train/_gpa_coco_piropo_tuning.sh 40 100 a TwoStageDetectorAdaptiveAdversarial 1 1 1 1 True fc_layer _norot_ann # increasing lambda
# train/_gpa_coco_piropo_tuning.sh 40 20 a TwoStageDetectorAdaptiveAdversarial   1   1   1   1 True fc_layer 
# train/_gpa_coco_piropo_tuning.sh 40 20 a TwoStageDetectorAdaptiveAdversarial 0.1 0.1 0.1 0.1 True fc_layer
# train/_gpa_coco_piropo_tuning.sh 40 20 a TwoStageDetectorAdaptiveAdversarial   1   1   1   1 True fc_layer _incr
# train/_gpa_coco_piropo_tuning.sh 40 20 a TwoStageDetectorAdaptiveAdversarial   1   1   1   1 True fc_layer _incr1
# train/_gpa_coco_piropo_tuning.sh 40 20 a TwoStageDetectorAdaptiveAdversarial   1   1   1   1 True fc_layer _incr2
# train/_gpa_coco_piropo_tuning.sh 40 20 a TwoStageDetectorAdaptiveAdversarial   1   1   1   1 True fc_layer _incr3
# train/_gpa_coco_piropo_tuning.sh 40 20 a TwoStageDetectorAdaptiveAdversarial   1   1   1   1 True fc_layer _direct1
# train/_gpa_coco_piropo_tuning.sh 40 20 a TwoStageDetectorAdaptiveAdversarial   1   1   1   1 True fc_layer _direct2
# train/_gpa_coco_piropo_tuning.sh 40 20 a TwoStageDetectorAdaptiveAdversarial   1   1   1   1 True none _direct3
# train/_gpa_coco_piropo_tuning.sh 40 20 a TwoStageDetectorAdaptiveAdversarial   1   1   1   1 True none _direct4
# train/_gpa_adv_coco_piropo_tuning.sh 40 20 a TwoStageDetectorAdaptiveAdversarial   1   1   1   1 True none _direct5
# train/_gpa_adv_coco_piropo_tuning.sh 40 20 a TwoStageDetectorAdaptiveAdversarial   1   1   1   1 True none _direct6
# train/_gpa_adv_coco_piropo_tuning.sh 40 20 a TwoStageDetectorAdaptiveAdversarial   1   1   1   1 True none _direct7
# train/_gpa_adv_coco_piropo_tuning.sh 40 20 a TwoStageDetectorAdaptiveAdversarial   1   1   1   1 True none _direct8
# train/_gpa_coco_piropo_tuning.sh 40 20 a TwoStageDetectorAdaptiveAdversarial   1   1   1   1 True none _direct9
# train/_gpa_coco_piropo_tuning.sh 80 20 a TwoStageDetectorAdaptiveAdversarial   1   1   1   1 True none _direct10

# improved framework
# train/_gpa_coco_piropo_tuning.sh 40 20 a TwoStageDetectorDA GPAtest # check validity of framework
# train/_gpa_coco_piropo_tuning.sh 40 20 a TwoStageDetectorDA GPAtest1 # check validity of framework
# train/_gpa_coco_piropo_tuning.sh 40 20 a TwoStageDetectorDA GPAratio # don't keep ratio
# train/_gpa_coco_piropo_tuning.sh 40 20 a TwoStageDetectorDA GPAratiotest # don't keep ratio, even durng testing
# train/_gpa_coco_piropo_tuning.sh 40 20 a TwoStageDetectorDA ADVneck4 # test adversarial on backbone features
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck4 -p # test adversarial on backbone features
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck4_p2 # test pooling2d on backbone features
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck4_c # test conv on backbone features
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck4_ch # test channel mode on backbone features
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck4_s # test sample mode on backbone features
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck4_s16 # test sample mode on backbone features
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck3_s16 model.train_cfg.da.0.feat='neck_3' # test sample mode on backbone features
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck2_s16 model.train_cfg.da.0.feat='neck_2' # test sample mode on backbone features
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck1_s16 model.train_cfg.da.0.feat='neck_1' # test sample mode on backbone features
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck0_s16 model.train_cfg.da.0.feat='neck_0' # test sample mode on backbone features
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneckall_s16 # test sample mode on backbone features
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck024_s16 # test sample mode on backbone features
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPAnogt # test sample mode on backbone features

# split head !!! CHANGE BACK IN config.py AND adapt_coco_piropo.sh !!!
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPAsplit # only on rcnn_cls
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPAsplit1 # rcnn_cls and rcnn_bbox
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPAsplit2 # rcnn_cls and rcnn_bbox
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPAcls # only on rcnn_cls
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPAclsgt91 # try different IoU threshols
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPAclsstep # only on rcnn_cls
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPAclsinv model.train_cfg.da.0.thr_mode='invers' # only on rcnn_cls
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPAp__  # only on roi pred, cls gt
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPAg__ model.train_cfg.da.0.mode='ground_truth' # only on roi gt, cls gt
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPAggg  # only on roi gt, cls gt


# A D V E R S A R I A L
# use a shared head to be consistent with earlier experiments
# EFFECT OF SAMPLE SIZE
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck4_s16x9 model.train_cfg.da.0.sample_shape=9
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck4_s16x7 model.train_cfg.da.0.sample_shape=7
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck4_s16x5 model.train_cfg.da.0.sample_shape=5
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck4_s16x3 model.train_cfg.da.0.sample_shape=3
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck4_s16x1 model.train_cfg.da.0.sample_shape=1
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck0_s16x15 model.train_cfg.da.0.sample_shape=15
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck0_s16x13 model.train_cfg.da.0.sample_shape=13
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck0_s16x11 model.train_cfg.da.0.sample_shape=11
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck0_s16x9 model.train_cfg.da.0.sample_shape=9
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck0_s16x7 model.train_cfg.da.0.sample_shape=7
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck0_s16x5 model.train_cfg.da.0.sample_shape=5
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck0_s16x3 model.train_cfg.da.0.sample_shape=3
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck0_s16x1 model.train_cfg.da.0.sample_shape=1
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck0_s16x17 model.train_cfg.da.0.sample_shape=17
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck0_s16x19 model.train_cfg.da.0.sample_shape=19
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck0_s16x38 model.train_cfg.da.0.sample_shape=38
# EFFECT OF SAMPLE COUNT
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck4_s20x7 model.train_cfg.da.0.sample_shape=7 model.train_cfg.da.0.feat=feat_neck_4 model.train_cfg.da.0.n_sample=20
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck4_s30x7 model.train_cfg.da.0.sample_shape=7 model.train_cfg.da.0.feat=feat_neck_4 model.train_cfg.da.0.n_sample=30
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck4_s50x7 model.train_cfg.da.0.sample_shape=7 model.train_cfg.da.0.feat=feat_neck_4 model.train_cfg.da.0.n_sample=50
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck4_s100x7 model.train_cfg.da.0.sample_shape=7 model.train_cfg.da.0.feat=feat_neck_4 model.train_cfg.da.0.n_sample=100
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck0_s20x3 model.train_cfg.da.0.sample_shape=3 model.train_cfg.da.0.feat=feat_neck_0 model.train_cfg.da.0.n_sample=20
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck0_s30x3 model.train_cfg.da.0.sample_shape=3 model.train_cfg.da.0.feat=feat_neck_0 model.train_cfg.da.0.n_sample=30
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck0_s50x3 model.train_cfg.da.0.sample_shape=3 model.train_cfg.da.0.feat=feat_neck_0 model.train_cfg.da.0.n_sample=50
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck0_s100x3 model.train_cfg.da.0.sample_shape=3 model.train_cfg.da.0.feat=feat_neck_0 model.train_cfg.da.0.n_sample=100
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck0_s20x7 model.train_cfg.da.0.sample_shape=7 model.train_cfg.da.0.feat=feat_neck_0 model.train_cfg.da.0.n_sample=20
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck0_s30x7 model.train_cfg.da.0.sample_shape=7 model.train_cfg.da.0.feat=feat_neck_0 model.train_cfg.da.0.n_sample=30
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck0_s50x7 model.train_cfg.da.0.sample_shape=7 model.train_cfg.da.0.feat=feat_neck_0 model.train_cfg.da.0.n_sample=50
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck0_s100x7 model.train_cfg.da.0.sample_shape=7 model.train_cfg.da.0.feat=feat_neck_0 model.train_cfg.da.0.n_sample=100
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck0_s20x13 model.train_cfg.da.0.sample_shape=13 model.train_cfg.da.0.feat=feat_neck_0 model.train_cfg.da.0.n_sample=20
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck0_s30x13 model.train_cfg.da.0.sample_shape=13 model.train_cfg.da.0.feat=feat_neck_0 model.train_cfg.da.0.n_sample=30
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck0_s50x13 model.train_cfg.da.0.sample_shape=13 model.train_cfg.da.0.feat=feat_neck_0 model.train_cfg.da.0.n_sample=50
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck0_s100x13 model.train_cfg.da.0.sample_shape=13 model.train_cfg.da.0.feat=feat_neck_0 model.train_cfg.da.0.n_sample=100
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck0_s20x19 model.train_cfg.da.0.sample_shape=19 model.train_cfg.da.0.feat=feat_neck_0 model.train_cfg.da.0.n_sample=20
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck0_s30x19 model.train_cfg.da.0.sample_shape=19 model.train_cfg.da.0.feat=feat_neck_0 model.train_cfg.da.0.n_sample=30
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck0_s50x19 model.train_cfg.da.0.sample_shape=19 model.train_cfg.da.0.feat=feat_neck_0 model.train_cfg.da.0.n_sample=50
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck0_s100x19 model.train_cfg.da.0.sample_shape=19 model.train_cfg.da.0.feat=feat_neck_0 model.train_cfg.da.0.n_sample=100
# GROUND-TRUTH
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVgtneck0
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVgtneck2 model.train_cfg.da.0.feat=feat_neck_2 model.train_cfg.da.0.sample_shape=13
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVgtneck4 model.train_cfg.da.0.feat=feat_neck_4 model.train_cfg.da.0.sample_shape=7
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVgtneck0_fg model.train_cfg.da.0.only_fg=True
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVgtneck4_fg model.train_cfg.da.0.feat=feat_neck_4 model.train_cfg.da.0.sample_shape=7 model.train_cfg.da.0.only_fg=True
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVgtneck2_fg model.train_cfg.da.0.feat=feat_neck_2 model.train_cfg.da.0.sample_shape=13 model.train_cfg.da.0.only_fg=True
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVgtneck0_d4
# COMBINATION
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneckall_02
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck04_05
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVneck024_033

# G P A
# use a split head to be consistent with earlier experiments
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPAclsinvtgt model.train_cfg.da.0.thr_mode='invers_tgt'
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPAcls_usecls model.train_cfg.da.0.gt_use_cls=True
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPAcls_useclsinv model.train_cfg.da.0.gt_use_cls=True
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPAbbox model.train_cfg.da.0.mode='prediction'
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPAbboxgt model.train_cfg.da.0.mode='ground_truth'
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPAbboxgt_x05 model.train_cfg.da.0.gt_iou_thrs=0.5
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPAbboxgt_x75 model.train_cfg.da.0.gt_iou_thrs=0.75
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPAbboxgt_x9 model.train_cfg.da.0.gt_iou_thrs=0.9
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPAbboxgt_y9 model.train_cfg.da.0.gt_bbox_dim='y'
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPAbboxgt_w9 model.train_cfg.da.0.gt_bbox_dim='w'
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPAbboxgt_h9 model.train_cfg.da.0.gt_bbox_dim='h'
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPAbboxgt_xywh1
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPApxg
train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPApxg033



# B R I N G I N G   I T   A L L   T O G E T H E R
# use a split head to be consistent throughout all combinations


# S W E E P S
# use a split head to be consistent throughout all combinations
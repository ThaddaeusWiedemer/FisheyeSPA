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


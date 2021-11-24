cd ~/MasterthesisCode

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
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVgtall
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADV012gt34 model.train_cfg.da.0.transform=sample model.train_cfg.da.1.transform=sample model.train_cfg.da.2.transform=sample
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADV01gt234
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADV234gt01
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADV34gt012 model.train_cfg.da.3.transform=sample model.train_cfg.da.4.transform=sample
# BETTER SCHEDULE
# train/adapt_coco_piropo.sh 80 20 a TwoStageDetectorDA ADVall_scdl_02
# train/adapt_coco_piropo.sh 80 20 a TwoStageDetectorDA ADVall_scdl model.train_cfg.da.0.lamd_weight=1 model.train_cfg.da.1.lamd_weight=1 model.train_cfg.da.2.lamd_weight=1 model.train_cfg.da.3.lamd_weight=1 model.train_cfg.da.4.lamd_weight=1  
# train/adapt_coco_piropo.sh 80 20 a TwoStageDetectorDA ADVall_scdl1_02
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVall_max 
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADV0123_max model.train_cfg.da.4.lambd_weight=0
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADV0124_max model.train_cfg.da.3.lambd_weight=0
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADV0134_max model.train_cfg.da.2.lambd_weight=0
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADV0234_max model.train_cfg.da.1.lambd_weight=0
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADV1234_max model.train_cfg.da.0.lambd_weight=0
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA cADVgt014
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA dADVgt0 #model.train_cfg.da.0.lambd_weight=1 model.train_cfg.da.1.lambd_weight=0 model.train_cfg.da.2.lambd_weight=0
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA cADVgt1 model.train_cfg.da.0.feat='feat_neck_1' model.train_cfg.da.0.sample_shape=28
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA cADVgt4
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA cADVgt01 model.train_cfg.da.0.lambd_weight=.5 model.train_cfg.da.1.lambd_weight=.5 model.train_cfg.da.2.lambd_weight=0
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA cADVgt04 model.train_cfg.da.0.lambd_weight=.5 model.train_cfg.da.1.lambd_weight=0 model.train_cfg.da.2.lambd_weight=.5
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA cADVgt14 model.train_cfg.da.0.lambd_weight=0 model.train_cfg.da.1.lambd_weight=.5 model.train_cfg.da.2.lambd_weight=.5
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA cADVgt04
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA cADV0gt4 model.train_cfg.da.0.transfrom='sample'
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA cADV4gt0 model.train_cfg.da.1.transfrom='sample'
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADV014_cyc4
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVbb0
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVbb1
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVbb3
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVbb0123
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA eADVgt014
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA eADVgt01 model.train_cfg.da.0.lambd_weight=.5 model.train_cfg.da.1.lambd_weight=.5 model.train_cfg.da.2.lambd_weight=0
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA eADVgt04 model.train_cfg.da.0.lambd_weight=.5 model.train_cfg.da.1.lambd_weight=0 model.train_cfg.da.2.lambd_weight=.5
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA eADVgt14 model.train_cfg.da.0.lambd_weight=0 model.train_cfg.da.1.lambd_weight=.5 model.train_cfg.da.2.lambd_weight=.5
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA eADVgt0 model.train_cfg.da.0.lambd_weight=1 model.train_cfg.da.1.lambd_weight=0 model.train_cfg.da.2.lambd_weight=0
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA eADVgt1 model.train_cfg.da.0.lambd_weight=0 model.train_cfg.da.1.lambd_weight=1 model.train_cfg.da.2.lambd_weight=0
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA eADVgt4 model.train_cfg.da.0.lambd_weight=0 model.train_cfg.da.1.lambd_weight=0 model.train_cfg.da.2.lambd_weight=1



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
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPApxg033
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPA-xg
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPApx-
# COMBINATIONS
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPAppp
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPAppp_033
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPApp- model.train_cfg.da.2.lambd=0
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPAp-p model.train_cfg.da.1.lambd=0
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPA-pp model.train_cfg.da.0.lambd=0
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPAp-- model.train_cfg.da.1.lambd=0 model.train_cfg.da.2.lambd=0
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPA-p- model.train_cfg.da.0.lambd=0 model.train_cfg.da.2.lambd=0
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPA--p model.train_cfg.da.0.lambd=0 model.train_cfg.da.1.lambd=0
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPA-x-_scdl_025
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPA-x-_scdl model.train_cfg.da.0.lambd=1 model.train_cfg.da.1.lambd=1 model.train_cfg.da.2.lambd=1 model.train_cfg.da.3.lambd=1
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPA-x-_comb
train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA GPApxg033_comb


# B R I N G I N G   I T   A L L   T O G E T H E R
# use a split head to be consistent throughout all combinations
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVallGPA-xg
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADV04GPAppp
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADV04GPApxg033
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVgt04GPApxg033
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVgt0GPApxg033
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVgt04GPAppp
# train/adapt_coco_piropo.sh 40 20 a TwoStageDetectorDA ADVgt0GPAppp


# S W E E P S
# use a split head to be consistent throughout all combinations
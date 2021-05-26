from mmdet.apis import init_detector, inference_detector
import mmcv

# specify path to model config and checkpoint file
config_file = 'mmdet/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'

# build model from config and checkpoint
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test on a single image and show result
img = mmcv.imread('/home/thaddaus/MasterthesisCode/data/PIROPO/omni_1A/omni1A_test6/2015-10-06T15-42-11.307Z.jpg')
result = inference_detector(model, img)

# visualize results in new window
# model.show_result(img, result)

# save visualization result to image file
model.show_result(img, result, out_file='result.jpg')
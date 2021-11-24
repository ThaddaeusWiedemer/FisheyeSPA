# Split shared head of pre-trained model into classification and bounding-box heads

import torch

in_path = 'mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'
out_path = 'mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227_split.pth'

model = torch.load(in_path)

# copy shared rcnn layers in separate branches
# then remove shared rcnn layers
model['state_dict'].update(
    {'roi_head.bbox_head.cls_fcs.0.weight': model['state_dict']['roi_head.bbox_head.shared_fcs.0.weight']})
model['state_dict'].update(
    {'roi_head.bbox_head.reg_fcs.0.weight': model['state_dict']['roi_head.bbox_head.shared_fcs.0.weight']})
del model['state_dict']['roi_head.bbox_head.shared_fcs.0.weight']

model['state_dict'].update(
    {'roi_head.bbox_head.cls_fcs.1.weight': model['state_dict']['roi_head.bbox_head.shared_fcs.1.weight']})
model['state_dict'].update(
    {'roi_head.bbox_head.reg_fcs.1.weight': model['state_dict']['roi_head.bbox_head.shared_fcs.1.weight']})
del model['state_dict']['roi_head.bbox_head.shared_fcs.1.weight']

model['state_dict'].update(
    {'roi_head.bbox_head.cls_fcs.0.bias': model['state_dict']['roi_head.bbox_head.shared_fcs.0.bias']})
model['state_dict'].update(
    {'roi_head.bbox_head.reg_fcs.0.bias': model['state_dict']['roi_head.bbox_head.shared_fcs.0.bias']})
del model['state_dict']['roi_head.bbox_head.shared_fcs.0.bias']

model['state_dict'].update(
    {'roi_head.bbox_head.cls_fcs.1.bias': model['state_dict']['roi_head.bbox_head.shared_fcs.1.bias']})
model['state_dict'].update(
    {'roi_head.bbox_head.reg_fcs.1.bias': model['state_dict']['roi_head.bbox_head.shared_fcs.1.bias']})
del model['state_dict']['roi_head.bbox_head.shared_fcs.1.bias']

# save
torch.save(model, out_path)
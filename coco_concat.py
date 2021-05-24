# concats multiple COCO like datasets to a single dataset

from pathlib import Path
import json
import pycocotools.coco as coco


def merge(images, annotations, next_img_id, next_ann_id, other, img_path=None):
    print(next_img_id)
    print(next_ann_id)
    other_images = other.loadImgs(other.getImgIds())
    img_id_mapping = {}
    for img in other_images:
        img_id_mapping[img['id']] = next_img_id
        img['id'] = next_img_id
        next_img_id += 1
        if img_path is not None:
            img['file_name'] = img_path + img['file_name']
        images.append(img)
    other_annotations = other.loadAnns(other.getAnnIds())
    for ann in other_annotations:
        ann['id'] = next_ann_id
        next_ann_id += 1
        ann['image_id'] = img_id_mapping[ann['image_id']]
        annotations.append(ann)
    return next_img_id, next_ann_id

def concat_sets(base_paths, sets, out_path):
    images = []
    annotations = []
    next_img_id = 0
    next_ann_id = 0
    for s in sets:
        if isinstance(s, tuple):
            json_file = s[0]
            image_path = s[1]
        else:
            json_file = s
            image_path = None
        c = coco.COCO(base_path / json_file)
        next_img_id, next_ann_id = merge(images, annotations, next_img_id, next_ann_id, c, image_path)
        print(len(images))
        print(len(annotations))
    categories = c.loadCats(c.getCatIds())
    result = {'categories': categories, 'images': images, 'annotations': annotations}
    with open(out_path, 'w') as f:
        json.dump(result, f)

base_path = Path('data/')
paths = [
        ('MW_18Mar/Test/MW-18Mar-1/annotations.json', 'MW_18Mar/Test/MW-18Mar-1/'),
        ('MW_18Mar/Test/MW-18Mar-4/annotations.json', 'MW_18Mar/Test/MW-18Mar-4/'),
        ('MW_18Mar/Test/MW-18Mar-5/annotations.json', 'MW_18Mar/Test/MW-18Mar-5/'),
        ('MW_18Mar/Test/MW-18Mar-6/annotations.json', 'MW_18Mar/Test/MW-18Mar-6/'),
        ('MW_18Mar/Test/MW-18Mar-9/annotations.json', 'MW_18Mar/Test/MW-18Mar-9/'),
        ('MW_18Mar/Test/MW-18Mar-11/annotations.json', 'MW_18Mar/Test/MW-18Mar-11/'),
        ('MW_18Mar/Test/MW-18Mar-15/annotations.json', 'MW_18Mar/Test/MW-18Mar-15/'),
        ('MW_18Mar/Test/MW-18Mar-16/annotations.json', 'MW_18Mar/Test/MW-18Mar-16/'),
        ('MW_18Mar/Test/MW-18Mar-20/annotations.json', 'MW_18Mar/Test/MW-18Mar-20/'),
        ('MW_18Mar/Test/MW-18Mar-28/annotations.json', 'MW_18Mar/Test/MW-18Mar-28/'),
        ('MW_18Mar/Test/MW-18Mar-29/annotations.json', 'MW_18Mar/Test/MW-18Mar-29/'),
        ('MW_18Mar/Test/MW-18Mar-30/annotations.json', 'MW_18Mar/Test/MW-18Mar-30/'),
        ('PIROPO/Room_A/omni_1A/omni1A_test2/annotations.json', 'PIROPO/Room_A/omni_1A/omni1A_test2/'),
        ('PIROPO/Room_A/omni_1A/omni1A_test3/annotations.json', 'PIROPO/Room_A/omni_1A/omni1A_test3/'),
        ('PIROPO/Room_A/omni_2A/omni2A_test2/annotations.json', 'PIROPO/Room_A/omni_2A/omni2A_test2/'),
        ('PIROPO/Room_A/omni_2A/omni2A_test3/annotations.json', 'PIROPO/Room_A/omni_2A/omni2A_test3/'),
        ('PIROPO/Room_A/omni_3A/omni3A_test2/annotations.json', 'PIROPO/Room_A/omni_3A/omni3A_test2/'),
        ('PIROPO/Room_A/omni_3A/omni3A_test3/annotations.json', 'PIROPO/Room_A/omni_3A/omni3A_test3/'),
        ('PIROPO/Room_B/omni_1B/omni1B_test2/annotations.json', 'PIROPO/Room_B/omni_1B/omni1B_test2/'),
        ('PIROPO/Room_B/omni_1B/omni1B_test3/annotations.json', 'PIROPO/Room_B/omni_1B/omni1B_test3/'),
]

concat_sets(base_path, paths,
        'data/MW18Mar_PIROPO_test.json')
 

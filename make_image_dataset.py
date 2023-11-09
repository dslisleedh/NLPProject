import torch
from mmdet.apis import DetInferencer
import matplotlib.pyplot as plt

from PIL import Image

import os
from argparse import ArgumentParser
from typing import Sequence


model = 'deformable-detr-refine_r50_16xb2-50e_coco' # model name
label = 0 # person label
threshold = 0.8 # threshold for certainty. If this is low, more images will be cropped but some of them may not be human or partial human.
enlarge_ratio = 0.2 # enlarge bbox by this ratio

img_extension = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']


def main(dirs: Sequence[str]):  # dirs must be absolute paths
    os.makedirs('./hpitp_dataset/', exist_ok=True)
    os.makedirs('./hpitp_dataset/images/', exist_ok=True)
    
    # Make image dataset first
    for file in os.listdir('./hpitp_dataset/images/'):
        os.remove(os.path.join('./hpitp_dataset/images/', file))
    
    imgs = []
    for dir_ in dirs:
        imgs.append(
            [os.path.join(dir_, img) for img in os.listdir(dir_) if os.path.splitext(img)[1] in img_extension]
        )
    imgs = sum(imgs, [])
    
    cuda_available = torch.cuda.is_available()
    inferencer = DetInferencer(model=model, device='cuda:0' if cuda_available else 'cpu')
    
    for n, img in enumerate(imgs):
        print(f'{n+1}/{len(imgs)}')
        res = inferencer(img)
        if len(res['predictions']) == 0:
            continue
        
        # get index where label is person
        labels = res['predictions'][0]['labels']
        idx = [i for i, x in enumerate(labels) if x == label]
        if len(idx) == 0:
            continue
        
        # crop image and save
        img_loaded = Image.open(img)
        img_name = os.path.splitext(os.path.basename(img))[0]
        for i in idx:
            bbox = res['predictions'][0]['bboxes'][i]
            
            # skip if bbox size is less than 224x224
            if bbox[2] - bbox[0] < 224 or bbox[3] - bbox[1] < 224:
                continue
            
            # skip if certainty is less than threshold
            if res['predictions'][0]['scores'][i] < threshold:
                continue
            
            w, h = img_loaded.size
            # Enlarge bbox
            # x1, y1, x2, y2
            x_size = bbox[2] - bbox[0]
            y_size = bbox[3] - bbox[1]
            
            bbox[0] = max(0, bbox[0] - x_size * enlarge_ratio)
            bbox[1] = max(0, bbox[1] - y_size * enlarge_ratio)
            bbox[2] = min(w, bbox[2] + x_size * enlarge_ratio)
            bbox[3] = min(h, bbox[3] + y_size * enlarge_ratio)
            
            cropped_img = img_loaded.crop(bbox)
            cropped_img.save(os.path.join('./hpitp_dataset/images/', f'{img_name}_{i}.jpg'))

    print('Number of human images cropped: ', len(os.listdir('./hpitp_dataset/images/')))
    

if __name__ == '__main__':
    dir_path = os.environ['IMAGEDIR']
    
    # dir1, dir2, ...
    dir_path = dir_path.split(',')
    dir_path = [os.path.abspath(dir_) for dir_ in dir_path]

    main(dir_path)
    
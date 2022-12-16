import json
import os
import math
import random
from random import random as rand

import torch
from torch.utils.data import Dataset

from torchvision.transforms.functional import hflip, resize

from PIL import Image
from dataset.utils import pre_caption
from refTools.refer_python3 import REFER


class grounding_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30, mode='train'):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.mode = mode

        if self.mode == 'train':
            self.img_ids = {}
            n = 0
            for ann in self.ann:
                img_id = ann['image'].split('/')[-1]
                if img_id not in self.img_ids.keys():
                    self.img_ids[img_id] = n
                    n += 1            
        
    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = pre_caption(ann['text'], self.max_words)

        if self.mode == 'train':
            img_id = ann['image'].split('/')[-1]

            return image, caption, self.img_ids[img_id]
        else:
            return image, caption, ann['ref_id']


class grounding_dataset_bbox(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30, mode='train', config=None, refer=None, remove_duplicated_refs=False):
        assert refer is not None, 'refer cannot be None!!!'
        self.refer = refer
        self.image_res = config['image_res']

        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))

        if remove_duplicated_refs:
            mp = {}
            new_ann = []
            for x in self.ann:
                if x['ref_id'] not in mp:
                    mp[x['ref_id']]=1
                    new_ann.append(x)
            self.ann=new_ann

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.mode = mode
        self.hflip_mode = config['hflip_mode']

        if self.mode == 'train':
            self.img_ids = {}
            n = 0
            for ann in self.ann:
                img_id = ann['image'].split('/')[-1]
                if img_id not in self.img_ids.keys():
                    self.img_ids[img_id] = n
                    n += 1

        assert config is not None
        self.image_res = config['image_res']
        self.patch_size = config['patch_size']
        assert self.image_res % self.patch_size == 0
        self.num_patch = int(self.image_res / self.patch_size)

        self.refid2img = {ann['ref_id']:ann['image'] for ann in self.ann}
        self.caption_key = 'text' if 'text' in self.ann[0] else 'sent'

    def __len__(self):
        return len(self.ann)


    def left_or_right_in(self, caption):
        def _func(s):
            if ('left' in s) or ('right' in s):
                return True
            else:
                return False

        if _func(caption):
            return True

        return False

    def __getitem__(self, index):

        ann = self.ann[index]
        caption = pre_caption(ann[self.caption_key], self.max_words)
        
        # coco2014
        # image_path = os.path.join(self.image_root, ann['image'])

        # coco2017
        coco_id = int(ann['image'].split('/')[-1].split('.')[0][-12:])
        image_path = os.path.join(self.image_root,'train2017/%012d.jpg' % coco_id)
        
        image = Image.open(image_path).convert('RGB')
        W, H = image.size

        x, y, w, h = self.refer.refToAnn[ann['ref_id']]['bbox']
        if self.mode == 'train':
            # random crop
            assert (x >= 0) and (y >= 0) and (x + w <= W) and (y + h <= H) and (w > 0) and (
                    h > 0), f"elem invalid, x: {x}, y: {y}, x+w: {x+w}, y+h: {y+h}, W: {W}, H: {H}"

            x0, y0 = random.randint(0, math.floor(x)), random.randint(0, math.floor(y))
            x1, y1 = random.randint(min(math.ceil(x + w), W), W), random.randint(min(math.ceil(y + h), H),
                                                                                 H)  # fix bug: max -> min
            w0, h0 = x1 - x0, y1 - y0
            assert (x0 >= 0) and (y0 >= 0) and (x0 + w0 <= W) and (y0 + h0 <= H) and (w0 > 0) and (
                    h0 > 0), "elem randomcrop, invalid"
            image = image.crop((x0, y0, x0 + w0, y0 + h0))

            W, H = image.size

            do_hflip = False
            if rand() < 0.5:
                if self.hflip_mode==0 or \
                    (self.hflip_mode==1 and not self.left_or_right_in(caption) ):
                    do_hflip = True
                
                if do_hflip:
                    image = hflip(image)

            image = resize(image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)
            image = self.transform(image)

            # axis transform: for crop
            x = x - x0
            y = y - y0
    
            if do_hflip:  # flipped applied
                x = max((W - x) - w, 0)  # W is w0
                # assert x>=0, f'x: {x}'

        else:
            image = self.transform(image)  # test_transform

        # resize applied
        x = self.image_res / W * x
        w = self.image_res / W * w
        y = self.image_res / H * y
        h = self.image_res / H * h

        center_x = x + 1 / 2 * w
        center_y = y + 1 / 2 * h

        target_bbox = torch.tensor([center_x / self.image_res, center_y / self.image_res,
                                    w / self.image_res, h / self.image_res], dtype=torch.float)

        image_atts = torch.tensor(self.get_image_attns(x, y, w, h))

        return image, caption, image_atts, target_bbox, ann['ref_id']

    def get_image_attns(self, x, y, w, h):
        x_min = min(math.floor(x / self.patch_size), self.num_patch - 1)
        x_max = max(x_min+1, min(math.ceil((x+w) / self.patch_size), self.num_patch))  # exclude

        y_min = min(math.floor(y / self.patch_size), self.num_patch - 1)
        y_max = max(y_min+1, min(math.ceil((y+h) / self.patch_size), self.num_patch))  # exclude

        image_atts = [0] * (1 + self.num_patch ** 2)
        image_atts[0] = 1  # always include [CLS]
        for j in range(x_min, x_max):
            for i in range(y_min, y_max):
                index = self.num_patch * i + j + 1
                assert (index > 0) and (index <= self.num_patch ** 2), f"patch index out of range, index: {index}"
                image_atts[index] = 1

        return image_atts
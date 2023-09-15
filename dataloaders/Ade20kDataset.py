import json
import os
import random

import jittor.transform as TR
import numpy as np
from jittor.dataset.dataset import Dataset
from PIL import Image


class Ade20kDataset(Dataset):
    def __init__(self, opt, load_type='train'):
        super(Ade20kDataset, self).__init__()
        self.load_type = load_type
        if opt.phase == "test":
            opt.load_size = 256
        else:
            opt.load_size = 286
        opt.crop_size = 256
        opt.label_nc = 29
        opt.contain_dontcare_label = True
        opt.semantic_nc = 30 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 1.0

        self.opt = opt
        self.images, self.labels, self.paths = self.list_images(load_type)

    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.paths[0], self.images[idx])).convert('RGB')
        label = Image.open(os.path.join(self.paths[1], self.labels[idx]))
        tmp_label = np.asarray(label)
        tmp_label = tmp_label.astype(np.uint8)
        label = Image.fromarray(tmp_label)
        image, label = self.transforms(image, label)
        # tmp_label1 = np.array(label)
        label = label * 255
        # tmp_label2 = np.array(label)
        return {"image": image, "label": label, "name": self.labels[idx]}

    def list_images(self, load_type):
        # mode = "validation" if self.opt.phase == "test" else "training"
        if load_type == "train" or load_type == "val":
            path_img = os.path.join(self.opt.dataset_path, "imgs", load_type)
            path_lab = os.path.join(self.opt.dataset_path, "labels", load_type)
            img_list = os.listdir(path_img)
            lab_list = os.listdir(path_lab)
            img_list = [filename for filename in img_list if ".png" in filename or ".jpg" in filename]
            lab_list = [filename for filename in lab_list if ".png" in filename or ".jpg" in filename]
            images = sorted(img_list)
            labels = sorted(lab_list)
            assert len(images)  == len(labels), "different len of images and labels %s - %s" % (len(images), len(labels))
            for i in range(len(images)):
                assert os.path.splitext(images[i])[0] == os.path.splitext(labels[i])[0], '%s and %s are not matching' % (images[i], labels[i])
        else: # load_type == "test"
            path_lab = self.opt.input_path
            label2img_path = os.path.join(os.path.dirname(path_lab), 'label_to_img.json')
            with open(label2img_path, 'r') as fp:
                label2img_dict = json.load(fp)
            path_img = self.opt.img_path
            lab_list = os.listdir(path_lab)
            labels = sorted(lab_list)
            images = [label2img_dict[label] for label in labels]

        return images, labels, (path_img, path_lab)
            


    def transforms(self, image, label):
        assert image.size == label.size
        # resize
        new_width, new_height = (self.opt.load_size, self.opt.load_size)
        image = TR.resize(image, (new_width, new_height), Image.BICUBIC)
        label = TR.resize(label, (new_width, new_height), Image.NEAREST)

        if not self.opt.phase == "test":
            # crop
            crop_x = random.randint(0, np.maximum(0, new_width -  self.opt.crop_size))
            crop_y = random.randint(0, np.maximum(0, new_height - self.opt.crop_size))
            image = image.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
            label = label.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
            # flip
            if not self.opt.no_flip:
                if random.random() < 0.5:
                    image = TR.hflip(image)
                    label = TR.hflip(label)
        # to tensor
        image = TR.ToTensor()(image)
        label = TR.ToTensor()(label)
        # label = TR.to_tensor(label)
        # normalize
        image = TR.ImageNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(image)
        return image, label

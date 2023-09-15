import os
import random
import shutil


def get_dataset_name(mode):
    if mode == "ade20k":
        return "Ade20kDataset"
    # if mode == "cityscapes":
    #     return "CityscapesDataset"
    else:
        ValueError("There is no such dataset regime as %s" % mode)


def get_dataloaders(opt, load_type='train'):
    dataset_name   = get_dataset_name(opt.dataset_mode)

    file = __import__("dataloaders."+dataset_name)

    dataset = file.__dict__[dataset_name].__dict__[dataset_name](opt, load_type)

    # print("Created %s, size train: %d, size val: %d" % (dataset_name, len(dataset_train), len(dataset_val)))

    return dataset


def dataset_seg(opt):
    if not os.path.exists(opt.dataset_path):
        os.makedirs(opt.dataset_path)

        img_dir_new = os.path.join(opt.dataset_path, 'imgs')
        anno_dir_new = os.path.join(opt.dataset_path, 'labels')
        os.makedirs(img_dir_new)
        os.makedirs(anno_dir_new)

        train_img_dir = os.path.join(img_dir_new, 'train')
        train_anno_dir = os.path.join(anno_dir_new, 'train')
        os.makedirs(train_img_dir)
        os.makedirs(train_anno_dir)

        val_img_dir = os.path.join(img_dir_new, 'val')
        val_anno_dir = os.path.join(anno_dir_new, 'val')
        os.makedirs(val_img_dir)
        os.makedirs(val_anno_dir)

        img_dir = os.path.join(opt.input_path, 'imgs')
        anno_dir = os.path.join(opt.input_path, 'labels')
        # img_list = sorted(os.listdir(img_dir))
        anno_list = sorted(os.listdir(anno_dir))
        num = 1000
        random_list = random.sample(anno_list, num) # for val
        for anno in anno_list:
            if anno not in random_list:
                shutil.copy(os.path.join(anno_dir, anno), train_anno_dir)
                tmp_path = anno.split('.')[0] + '.jpg'
                shutil.copy(os.path.join(img_dir, tmp_path), train_img_dir)
            else:
                shutil.copy(os.path.join(anno_dir, anno), val_anno_dir)
                tmp_path = anno.split('.')[0] + '.jpg'
                shutil.copy(os.path.join(img_dir, tmp_path), val_img_dir)


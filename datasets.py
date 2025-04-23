# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
from PIL import Image

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder

class IterationBasedDataset(Dataset):
    def __init__(self, root_dir, iterations, transform=None):
        self.root_dir = root_dir
        self.iterations = iterations
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.classes.sort()
        self.samples = []
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            file_list = [os.path.join(dp, f) for dp, dn, fn in os.walk(cls_path) for f in fn]
            file_list = [f for f in file_list if os.path.isfile(f) and f.endswith('png')]
            self.samples.extend([(os.path.join(cls_path, img), cls) for img in file_list])
        self.num_samples = len(self.samples)

    def __len__(self):
        return self.iterations

    def __getitem__(self, index):
        sample_idx = index % self.num_samples
        img_path, label = self.samples[sample_idx]
        image = Image.open(img_path).convert('RGB')
        label_idx = self.classes.index(label)
        if self.transform:
            image = self.transform(image)
        return image, label_idx


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    if is_train:
        dataset = IterationBasedDataset(root_dir=root, iterations=10000, transform=transform)
    else:
        dataset = datasets.ImageFolder(root, transform=transform)
    nb_classes = 3
    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

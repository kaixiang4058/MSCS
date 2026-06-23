import argparse
import os
import yaml
from collections import Counter

import numpy as np
from tqdm import tqdm

from dataprocess.HisPathDataset import HisPathDataset


def load_config(cfg_path):
    with open(cfg_path, 'r', encoding='utf-8') as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    return cfg


def build_dataset(stage: str, cfg: dict):
    rootset = cfg.get('rootset', {})
    settings = {
        'dataroot': rootset.get('dataroot'),
        'datalist': rootset.get('datalist'),
        'classes': cfg.get('classes', []),
        'patchsize': cfg.get('traindl', {}).get('patchsize', 512),
        'stridesize': cfg.get('traindl', {}).get('stridesize', 384),
        'tifpage': cfg.get('traindl', {}).get('tifpage', 0),
        'lr_ratio': cfg.get('lr_ratio', cfg.get('expset', {}).get('lrratio', 8)),
        'preprocess': None,
    }
    return HisPathDataset(stage=stage, transform=None, **settings)


def normalize_mask_array(mask):
    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 3 and mask_arr.shape[2] == 1:
        mask_arr = mask_arr[..., 0]
    elif mask_arr.ndim == 3 and mask_arr.shape[2] > 1:
        # if multi-channel mask, assume first channel contains label values
        mask_arr = mask_arr[..., 0]
    return mask_arr


def summarize_patch_labels(dataset):
    counter = Counter()
    for item in dataset.datalist:
        label = item[2]
        if isinstance(label, np.ndarray):
            label = label.tolist()
        counter[label] += 1
    return counter


def summarize_pixel_counts(dataset):
    pixel_counter = Counter()
    for item in tqdm(dataset.datalist, desc='Counting mask pixels'):
        img_path, mask_path, _, (x, y), datatype = item
        if mask_path is None:
            continue

        try:
            slide_mask = dataset.read_WSI(mask_path, 'tif', level=dataset.tifpage)
            mask = dataset._tiffcrop(slide_mask, x, y, dataset.patchsize, dataset.patchsize)
        except Exception as exc:
            print(f'Warning: failed to load mask for item {mask_path}: {exc}')
            continue

        mask_arr = normalize_mask_array(mask)
        unique, counts = np.unique(mask_arr, return_counts=True)
        for label, count in zip(unique.tolist(), counts.tolist()):
            pixel_counter[int(label)] += int(count)
    return pixel_counter


def display_counts(stage, patch_counts, pixel_counts=None, class_names=None):
    print(f'===== {stage} =====')
    total_patches = sum(patch_counts.values())
    print(f'Total patches: {total_patches}')
    print('Patch count by label/category:')
    for label, count in patch_counts.most_common():
        label_name = class_names[label] if class_names and isinstance(label, int) and label < len(class_names) else label
        print(f'  {label_name}: {count}')

    if pixel_counts is not None:
        print('Pixel count by label:')
        for label, count in sorted(pixel_counts.items()):
            label_name = class_names[label] if class_names and isinstance(label, int) and label < len(class_names) else label
            print(f'  {label_name}: {count}')
        total_pixels = sum(pixel_counts.values())
        print(f'Total pixels: {total_pixels}')
    print()


def validate_stage_name(stage):
    valid = {'train', 'train_label', 'train_unlabel', 'valid', 'test', 'all'}
    if stage not in valid:
        raise ValueError(f"Unknown stage '{stage}', must be one of: {', '.join(sorted(valid))}")
    return stage


def main():
    parser = argparse.ArgumentParser(description='Evaluate dataset patch and pixel counts')
    parser.add_argument('--data_cfg', type=str, default='./dataprocess/cfg/datacfg_MRCPS_labBreast.yaml',
                        help='Path to data config YAML file')
    parser.add_argument('--stage', type=validate_stage_name, default='train',
                        help='Dataset stage to summarize: train, train_label, train_unlabel, valid, test, all')
    parser.add_argument('--class_names', nargs='*', default=None,
                        help='Optional class names for pixel label mapping')
    args = parser.parse_args()

    cfg = load_config(args.data_cfg)
    class_names = args.class_names or cfg.get('classes', None)

    stages = []
    if args.stage == 'train':
        stages = ['train_label', 'train_unlabel']
    elif args.stage == 'all':
        stages = ['train_label', 'train_unlabel', 'valid', 'test']
    else:
        stages = [args.stage]

    for stage in stages:
        try:
            dataset = build_dataset(stage, cfg)
        except Exception as exc:
            print(f'Failed to build dataset for stage {stage}: {exc}')
            continue

        patch_counts = summarize_patch_labels(dataset)
        pixel_counts = None
        if stage != 'train_unlabel':
            pixel_counts = summarize_pixel_counts(dataset)
        display_counts(stage, patch_counts, pixel_counts, class_names)


if __name__ == '__main__':
    main()

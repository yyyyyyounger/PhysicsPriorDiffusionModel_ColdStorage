import os
import csv
import json
import torch
import torchvision
import random
import numpy as np
from torchvision import transforms, utils
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def _clean_record(record):
    return {
        str(key).strip(): value.strip() if isinstance(value, str) else value
        for key, value in record.items()
    }


def _metadata_source(metadata_csv=None, metadata_jsonl=None, finetune_root=None):
    if metadata_csv:
        return metadata_csv, 'csv'
    if metadata_jsonl:
        return metadata_jsonl, 'jsonl'
    if finetune_root:
        csv_path = os.path.join(finetune_root, 'metadata.csv')
        if os.path.isfile(csv_path):
            return csv_path, 'csv'
        jsonl_path = os.path.join(finetune_root, 'metadata.jsonl')
        if os.path.isfile(jsonl_path):
            return jsonl_path, 'jsonl'
    return None, None


def _resolve_metadata_path(path, base_dir):
    if path is None or path == '':
        return None
    path = str(path)
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def _split_matches(record_split, split):
    if record_split is None or record_split == '':
        return True
    if split is None:
        return True
    aliases = {
        'train': {'train', 'training'},
        'val': {'val', 'valid', 'validation'},
        'test': {'test', 'testing'},
    }
    record_split = str(record_split).strip().lower()
    split = str(split).strip().lower()
    return record_split in aliases.get(split, {split})


def _read_metadata_records(metadata_path, metadata_type):
    records = []
    if metadata_type == 'csv':
        with open(metadata_path, newline='') as f:
            reader = csv.DictReader(f)
            records = [_clean_record(row) for row in reader]
    elif metadata_type == 'jsonl':
        with open(metadata_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(_clean_record(json.loads(line)))
    else:
        raise ValueError('Unsupported metadata type: {}'.format(metadata_type))
    return records


def paired_paths_from_metadata(metadata_csv=None, metadata_jsonl=None, finetune_root=None, split=None):
    metadata_path, metadata_type = _metadata_source(
        metadata_csv=metadata_csv,
        metadata_jsonl=metadata_jsonl,
        finetune_root=finetune_root)
    if metadata_path is None:
        return None

    metadata_path = os.path.abspath(metadata_path)
    base_dir = os.path.abspath(finetune_root) if finetune_root else os.path.dirname(metadata_path)
    records = _read_metadata_records(metadata_path, metadata_type)
    paths = []
    for row_idx, record in enumerate(records):
        if not _split_matches(record.get('split'), split):
            continue
        hazy_path = record.get('hazy')
        gt_path = record.get('gt') or record.get('clear')
        if not hazy_path or not gt_path:
            raise ValueError(
                'metadata row {} must contain hazy and gt paths'.format(row_idx))
        beta = record.get('beta', 0.0)
        if beta in (None, ''):
            beta = 0.0
        paths.append({
            'lq_path': _resolve_metadata_path(hazy_path, base_dir),
            'gt_path': _resolve_metadata_path(gt_path, base_dir),
            'depth_path': _resolve_metadata_path(record.get('depth'), base_dir),
            'beta': float(beta),
        })

    if not paths:
        raise ValueError(
            'metadata {} has no samples for split {}'.format(metadata_path, split))
    return paths


def clean_depth(depth):
    depth = np.asarray(depth, dtype=np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth[depth < 0] = 0.0
    return depth


def depth2tensor(depth):
    depth = clean_depth(depth)
    if depth.ndim == 2:
        depth = depth[None, :, :]
    elif depth.ndim == 3:
        if depth.shape[0] == 1:
            pass
        elif depth.shape[-1] == 1:
            depth = np.transpose(depth, (2, 0, 1))
        else:
            depth = depth[:1, :, :]
    else:
        raise ValueError('depth array must be 2D or 3D, got shape {}'.format(depth.shape))
    return torch.from_numpy(np.ascontiguousarray(depth)).float()


def load_depth_npy(path):
    return depth2tensor(np.load(path))


def resize_depth(depth, img_size):
    if not torch.is_tensor(depth):
        depth = depth2tensor(depth)
    depth = depth.float().unsqueeze(0)
    depth = torch.nn.functional.interpolate(
        depth, size=img_size, mode='bilinear', align_corners=False)
    return depth.squeeze(0)


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img


# implementation by numpy and torch
# def transform_augment(img_list, split='val', min_max=(0, 1)):
#     imgs = [transform2numpy(img) for img in img_list]
#     imgs = augment(imgs, split=split)
#     ret_img = [transform2tensor(img, min_max) for img in imgs]
#     return ret_img




def paired_random_crop(img_gts, img_lqs, gt_patch_size):
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]
    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_gt, w_gt = img_gts[0].size()[-2:]

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_gt - gt_patch_size[0])
    left = random.randint(0, w_gt - gt_patch_size[1])

    # crop lq patch
    if input_type == 'Tensor':
        img_lqs = [v[:, top:top + gt_patch_size[0], left:left + gt_patch_size[1]] for v in img_lqs]

    # crop corresponding gt patch
    if input_type == 'Tensor':
        img_gts = [v[:, top:top + gt_patch_size[0], left:left + gt_patch_size[1]] for v in img_gts]


    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]

    return img_gts, img_lqs



#implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14 
totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()
resize = transforms.RandomResizedCrop(512,scale=(0.5,1.0))
def transform_augment(img_list, split='val', img_size=(512, 512), min_max=(0, 1), depth_list=None):
    Resize = transforms.Resize(img_size)
    img_list = [Resize(img) for img in img_list]
    imgs = [totensor(img) for img in img_list]
    depths = None
    if depth_list is not None:
        depths = [resize_depth(depth, img_size) for depth in depth_list]
    if split == 'train':
        do_hflip = random.random() < 0.5
        if do_hflip:
            imgs = [torch.flip(img, dims=[2]) for img in imgs]
            if depths is not None:
                depths = [torch.flip(depth, dims=[2]) for depth in depths]
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    if depth_list is not None:
        return ret_img, depths
    return ret_img

# totensor = torchvision.transforms.ToTensor()
# hflip = torchvision.transforms.RandomHorizontalFlip()
# resize = transforms.RandomResizedCrop(512,scale=(0.5,1.0))
# def transform_augment(img_list, split='val', img_size=(512, 512), min_max=(0, 1)):
#     imgs = [totensor(img) for img in img_list]
#     imgs[0], imgs[1] = paired_random_crop(imgs[0], imgs[1], img_size)
#     if split == 'train':
#         imgs = torch.stack(imgs, 0)
#         imgs = hflip(imgs)
#         imgs = torch.unbind(imgs, dim=0)
#     ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
#     return ret_img
    
    

    
    
    
    
'''
Author: tom
Date: 2024-10-28 15:27:29
LastEditors: Do not edit
LastEditTime: 2025-01-14 11:34:31
FilePath: /videio_vae/3d-knee-vae/experiments/custom.py
'''
import os
from torch.utils.data import Dataset
import numpy as np
from monai import transforms
from monai.data import Dataset as MonaiDataset
from monai.transforms import MapTransform
import nibabel as nib
from model.register_world_coordinate import create_nifti_dirs,create_nifti_series
from model.dicom2nifti import load_dicom_series,convert_to_nifti
import torch
import gc
from scipy.ndimage import zoom

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
import csv
import torch.nn.functional as F

def norm(x, lower_percentile=0., upper_percentile=99.75, b_min=0., b_max=1., clip=False):
    lower_value = np.percentile(x, lower_percentile)
    upper_value = np.percentile(x, upper_percentile)
    normalized_arr = (x - lower_value) / (upper_value - lower_value) * (b_max - b_min) + b_min   
    if clip:
        normalized_arr = np.clip(normalized_arr, b_min, b_max)
    return normalized_arr

def resize(img_data, affine,target_size=(384,384)):
    d,h,w = img_data.shape
    p = max(h,w)
    pad_h = p-h
    pad_w = p-w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padded_img = np.pad(img_data, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=0)
    zoom_factors = (1, target_size[0] / padded_img.shape[1], target_size[1] / padded_img.shape[2])
    resized_img = zoom(padded_img, zoom_factors, order=1)  # order参数可根据需求调整插值方法，这里使用线性插值

    # 计算水平和垂直方向的缩放比例
    scale_x = target_size[1] / (w + pad_w)
    scale_y = target_size[0] / (h + pad_h)
    # 计算填充后图像的中心坐标以及目标图像的中心坐标
    center_x_padded = (w + pad_w) / 2
    center_y_padded = (h + pad_h) / 2
    center_x_target = target_size[1] / 2
    center_y_target = target_size[0] / 2
    # 更新仿射变换矩阵中的缩放元素
    affine[0, 0] *= scale_x
    affine[1, 1] *= scale_y
    # 更新仿射变换矩阵中的平移元素，考虑缩放和中心位置变化
    affine[0, 3] = (affine[0, 3] - center_x_padded) * scale_x + center_x_target
    affine[1, 3] = (affine[1, 3] - center_y_padded) * scale_y + center_y_target

    return resized_img, affine

def clip_depth(img_data, affine, target_depth=32, start_d=None):
    if img_data.shape[0] < target_depth:
        padding_depth = target_depth - img_data.shape[0]
        padding_data = np.zeros((padding_depth, *img_data.shape[1:]), dtype=img_data.dtype)
        img_data = np.concatenate((img_data, padding_data), axis=0)        
    if start_d is None:
            start_d = np.random.randint(0, img_data.shape[0] - target_depth + 1)
    resized_img = img_data[start_d:start_d + target_depth, :, :]   
    affine = adjust_affine_for_random_depth_crop(affine,start_d)
    return resized_img,affine,start_d

def adjust_affine_for_random_depth_crop(affine, start_depth):
    original_start_depth = start_depth
    affine[2, 3] -= (original_start_depth - 0)  # 需要减去裁剪的起始位置偏移
    return affine


def get_knee_file_paths(data_path):
    images = []
    if os.path.isfile(data_path):
        images = [i for i in np.genfromtxt(data_path, dtype=np.str, encoding='utf-8')]
    else:
        assert os.path.isdir(data_path), '%s is not a valid directory' % data_path
        for root, _, fnames in sorted(os.walk(data_path)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if 'SAG' in path:
                    sag_path = path
                    sag_dir = sag_path.rsplit('/', 1)[0]
                    obj = sag_path.split('/')[-1]
                    cor_dir = sag_dir.replace('SAG','COR')
                    for cor_file in os.listdir(cor_dir):
                        cor_path = os.path.join(cor_dir, cor_file)
                        images.append({"subject_id": obj, "sag_path": sag_path, "cor_path": cor_path})
    return images


def get_oai_knee_file_paths(data_path):
    images = []
    if os.path.isfile(data_path):
        images = [i for i in np.genfromtxt(data_path, dtype=np.str, encoding='utf-8')]
    else:
        assert os.path.isdir(data_path), '%s is not a valid directory' % data_path
        for root, _, fnames in sorted(os.walk(data_path)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if '/dess/' in path:
                    dess_path = path
                    dess_dir = dess_path.rsplit('/', 1)[0]
                    obj = dess_path.split('/')[-1]
                    tse_sag_dir = dess_dir.replace('/dess/','/tse_sag_r/')
                    for tse_sag_file in os.listdir(tse_sag_dir):
                        tse_sag_path = os.path.join(tse_sag_dir, tse_sag_file)
                        images.append({"subject_id": obj, "dess_path": dess_path, "tse_sag_path": tse_sag_path})
    return images


class CustomBase(Dataset):
    def __init__(self, data_path,batch_size=1, device=None):
        super().__init__()
        
        self.data_path = data_path
        if 'OAI' in data_path:
            self.data_paths = get_oai_knee_file_paths(data_path)            #[:4000]
        else:
            self.data_paths = get_knee_file_paths(data_path)
        self.device = device


    def __len__(self):
        return len(self.data_paths)

    def load_dirs(self, path, start_d=None, device=None):
        datum = create_nifti_dirs(path, force_v1=True, device=device)
        if 'OAI' in self.data_path:
            datum['images'],datum['affine'] = resize(datum['images'],datum['affine'])
            datum['images'],datum['affine'],start_d = clip_depth(datum['images'],datum['affine'], target_depth=64, start_d = start_d) #resized_img,affine,start_d
        datum['images'] = datum['images'][None,...]
        return datum,start_d

    def __getitem__(self, idx):
        item_paths = self.data_paths[idx]
        if 'OAI' in self.data_path:
            with torch.no_grad():
                dess_dic,start_d = self.load_dirs(item_paths['dess_path'], device=self.device)
                tse_dic,_ = self.load_dirs(item_paths['tse_sag_path'],start_d=start_d,device=self.device)

            return {
                "subject_id": item_paths['subject_id'],
                "dess_dic": dess_dic,
                "tse_dic": tse_dic
            }

        else:
            with torch.no_grad():
                sag_dir = self.load_dirs(item_paths['sag_path'],device=self.device)
                cor_dir = self.load_dirs(item_paths['cor_path'],device=self.device)
            return {
                "subject_id": item_paths['subject_id'],
                "sag_dirs": sag_dir,
                "cor_dirs": cor_dir
            }
        
    def clear_cache(self):
        gc.collect()
        torch.cuda.empty_cache()

class CustomTrain(CustomBase):
    def __init__(self, data_path, device, **kwargs):
        super().__init__(data_path=data_path,device=device)

class CustomTest(CustomBase):
    def __init__(self, data_path,batch_size, device, **kwargs):
        super().__init__(data_path=data_path,batch_size=1,device=device)


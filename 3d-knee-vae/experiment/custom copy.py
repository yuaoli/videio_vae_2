import os
from torch.utils.data import Dataset
import numpy as np
from monai import transforms
from monai.data import Dataset as MonaiDataset
from monai.transforms import MapTransform
import nibabel as nib
from model.register_world_coordinate import create_nifti_series
import torch

# brats_transforms = transforms.Compose(
#     [
#         transforms.LoadImaged(keys=["t1", "t1ce", "t2", "flair"], allow_missing_keys=True),
#         transforms.EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair"], allow_missing_keys=True),
#         transforms.Lambdad(keys=["t1", "t1ce", "t2", "flair"], func=lambda x: x[0, :, :, :]),
#         transforms.AddChanneld(keys=["t1", "t1ce", "t2", "flair"]),
#         transforms.EnsureTyped(keys=["t1", "t1ce", "t2", "flair"]),
#         transforms.Orientationd(keys=["t1", "t1ce", "t2", "flair"], axcodes="RAI", allow_missing_keys=True),
#         transforms.CropForegroundd(keys=["t1", "t1ce", "t2", "flair"], source_key="t1", allow_missing_keys=True),
#         transforms.SpatialPadd(keys=["t1", "t1ce", "t2", "flair"], spatial_size=(160, 160, 128), allow_missing_keys=True),
#         transforms.RandSpatialCropd( keys=["t1", "t1ce", "t2", "flair"],
#             roi_size=(160, 160, 128),
#             random_center=True, 
#             random_size=False,
#         ),
#         transforms.ScaleIntensityRangePercentilesd(keys=["t1", "t1ce", "t2", "flair"], lower=0, upper=99.75, b_min=0, b_max=1),
#     ]
# )
def select_first_channel(x):
    return x[0, :, :, :]

knee_transforms = transforms.Compose(
    [
        # transforms.LoadImaged(keys=["SAG", "COR"]),
        transforms.EnsureChannelFirstd(keys=["SAG", "COR"]),
        transforms.Lambdad(keys=["SAG", "COR"], func=select_first_channel),
        transforms.AddChanneld(keys=["SAG", "COR"]),
        transforms.EnsureTyped(keys=["SAG", "COR"]),
        # transforms.Orientationd(keys=["SAG", "COR"], axcodes="RAI"),
        # transforms.CropForegroundd(keys=["SAG", "COR"], source_key="SAG"),
        # transforms.SpatialPadd(keys=["SAG", "COR"], spatial_size=(256, 32, 256)),
        # transforms.Resized(keys=["SAG", "COR"], spatial_size=(128, 128, 64)),
        # transforms.RandSpatialCropd( keys=["SAG", "COR"],
        #     roi_size=(256, 32, 256),
        #     random_center=True, 
        #     random_size=False,
        # ),
        # transforms.ScaleIntensityRangePercentilesd(keys=["SAG", "COR"], lower=0, upper=99.75, b_min=0, b_max=1), #[b c h d w][1 1 256 16 256]upper=99.75
        #transforms.RepeatChanneld(keys=["SAG", "COR"], repeats=3),
    ]
)

def get_brats_dataset(data_path):
    transform = brats_transforms 
    
    data = []
    for subject in os.listdir(data_path):
        sub_path = os.path.join(data_path, subject)
        if os.path.exists(sub_path) == False: continue
        t1 = os.path.join(sub_path, f"{subject}_t1.nii.gz") 
        t1ce = os.path.join(sub_path, f"{subject}_t1ce.nii.gz") 
        t2 = os.path.join(sub_path, f"{subject}_t2.nii.gz") 
        flair = os.path.join(sub_path, f"{subject}_flair.nii.gz") 
        seg = os.path.join(sub_path, f"{subject}_seg.nii.gz")

        data.append({"t1":t1, "t1ce":t1ce, "t2":t2, "flair":flair, "subject_id": subject})
                    
    print("num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)

def get_knee_dataset(data_path):

    transform = knee_transforms
    
    if os.path.isfile(data_path):
        images = [i for i in np.genfromtxt(data_path, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(data_path), '%s is not a valid directory' % data_path
        for root, _, fnames in sorted(os.walk(data_path)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if 'SAG' in path:
                    sag_path = path
                    sag_dir =  sag_path.rsplit('/', 1)[0]
                    obj = sag_path.split('/')[-1]
                    cor_dir = sag_dir.replace('SAG','COR')
                    for cor_file in os.listdir(cor_dir):
                        cor_path = os.path.join(cor_dir,cor_file)

                    sag_series = create_nifti_series(sag_path,force_v1=True)
                    cor_series = create_nifti_series(cor_path,force_v1=True)

                    sag_dic = {}
                    sag_dic['images'] = sag_series.images
                    sag_dic['T_pix2world'] = sag_series.T_pix2world
                    sag_dic['T_world2pix'] = sag_series.T_world2pix
                    sag_dic['T_nat2torch'] = sag_series.T_nat2torch
                    sag_dic['T_torch2nat'] = sag_series.T_torch2nat
                    sag_dic['spacing_xyz'] = sag_series.spacing_xyz
                    sag_dic['affine'] = sag_series.affine
                    sag_dic['affine_itk'] = sag_series.affine_itk
                    # sag_dic['shape'] = sag_series.shape
                    try:
                        for key in sag_dic:
                            sag_dic[key] = torch.as_tensor(sag_dic[key])
                    except:
                        pass


                    cor_dic = {}
                    cor_dic['images'] = cor_series.images
                    cor_dic['T_pix2world'] = cor_series.T_pix2world
                    cor_dic['T_world2pix'] = cor_series.T_world2pix
                    cor_dic['T_nat2torch'] = cor_series.T_nat2torch
                    cor_dic['T_torch2nat'] = cor_series.T_torch2nat
                    cor_dic['spacing_xyz'] = cor_series.spacing_xyz
                    cor_dic['affine'] = cor_series.affine
                    cor_dic['affine_itk'] = cor_series.affine_itk
                    # cor_dic['shape'] = cor_series.shape
                    try:
                        for key in cor_dic:
                            cor_dic[key] = torch.as_tensor(cor_dic[key])
                    except:
                        pass

                    # sag_img = nib.load(sag_path)
                    # sag_affine = sag_img.affine

                    # cor_img = nib.load(cor_path)
                    # cor_affine = cor_img.affine                    

                    # images.append({'SAG':sag_path,'COR':cor_path,"subject_id": obj,'sag_series':sag_series,'cor_series':cor_series})
                    images.append({"subject_id": obj,'sag_series':sag_dic,'cor_series':cor_dic})

    # return MonaiDataset(data=images, transform=transform)
    return images


class CustomBase(Dataset):
    def __init__(self,data_path):
        super().__init__()
        self.data = get_knee_dataset(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class CustomTrain(CustomBase):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path)


class CustomTest(CustomBase):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path)
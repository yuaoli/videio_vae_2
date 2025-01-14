import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import glob
import json
import os


def save_nifti(pixel_data, affine, fpath):
    pixel_data = np.transpose(pixel_data, (2, 1, 0))  # WHD

    nifti_img = nib.Nifti1Image(pixel_data, affine)
    nib.save(nifti_img, fpath)


def get_T_natural_to_torch(shape):
    T = np.eye(4)
    for i, dim in enumerate(shape[::-1]):
        if dim == 1:
            dim = np.nan
        T[i, i] = 2 / (dim - 1)
        T[i, -1] = -1
    return T


def get_T_torch_to_natural(shape):
    T = np.eye(4)
    for i, dim in enumerate(shape[::-1]):
        T[i, i] = (dim - 1) / 2
        T[i, -1] = (dim - 1) / 2
    return T


class DummySeries(object):
    def __init__(self, images, T_pix2world, T_world2pix, T_nat2torch, T_torch2nat):
        if len(images.shape)!=5:
            images = images[None,...]
        self.images = images
        self.shape = self.images.shape
        self.T_pix2world = T_pix2world
        self.T_world2pix = T_world2pix
        self.T_nat2torch = T_nat2torch
        self.T_torch2nat = T_torch2nat

class Dummydirs(object):
    def __init__(self, dic, device):
        for key,value in dic.items():
            dic[key] = torch.as_tensor(value, dtype=torch.float32, device=device)

def _parse_nifti_meta(nifti_image):
    header = nifti_image.header
    meta = {}
    for f in ["aux_file", "descrip"]:
        if not header[f]:
            continue
        meta_str = header[f].tolist().decode('utf-8')
        meta.update(dict([_.split(":") for _ in meta_str.split(";")]))
    # key mapping
    key_mapping = {"V": "version", "DO": "data_order", "S": "series_id", "R": "instance_range"}
    meta = {key_mapping[k]: v for k, v in meta.items()}
    if "instance_range" in meta:
        meta["instance_range"] = [int(_) for _ in meta["instance_range"].split("-")]
    return meta


def create_nifti_dirs(nii_file, force_v1=False,device='cuda'):
    if isinstance(nii_file, str):
        nifti_image = nib.load(nii_file)
    else:
        nifti_image = nii_file

    dir = dict()

    images = np.asanyarray(nifti_image.dataobj)
    meta = _parse_nifti_meta(nifti_image)
    v1 = False
    if meta or force_v1:
        if not force_v1:
            assert meta["version"] in {'knee1.0'}
        v1 = True
        if force_v1 or (meta["data_order"] == "whd"):  # to dhw
            images = np.transpose(images, (2, 1, 0))
        else:
            assert meta["data_order"] == "dhw", f'{meta["data_order"]} is not supported now'
    else:
        raise NotImplementedError(f"no version info is not supported currently")
    # in v1, direction of ** and ** has flipped for itk-snap support
    T_pix2world = nifti_image.affine.copy()
    # not needed
    if v1:
        T_pix2world[0] *= -1
        T_pix2world[1] *= -1
    T_world2pix = np.linalg.inv(T_pix2world)
    # to DHW
    # images = images.astype(np.int16)  # WHD->DHW
    T_nat2torch = get_T_natural_to_torch(images.shape)
    T_torch2nat = get_T_torch_to_natural(images.shape)
    # spacing_xyz
    spacing = np.linalg.norm(nifti_image.affine[:3, :3], axis=0)

    dir['images'] = images
    dir['T_pix2world'] = T_pix2world
    dir['T_world2pix'] = T_world2pix
    dir['T_nat2torch'] = T_nat2torch
    dir['T_torch2nat'] = T_torch2nat

    dir['spacing_xyz'] = spacing
    dir['affine'] = T_pix2world
    dir['affine_itk'] = nifti_image.affine

    # for key,value in dir.items():
    #     dir[key] = torch.as_tensor(value, dtype=torch.float32, device=device)

    return dir 

def create_nifti_series(nii_file, force_v1=False):
    if isinstance(nii_file, str):
        nifti_image = nib.load(nii_file)
    else:
        nifti_image = nii_file
    images = np.asanyarray(nifti_image.dataobj)
    meta = _parse_nifti_meta(nifti_image)
    v1 = False
    if meta or force_v1:
        if not force_v1:
            assert meta["version"] in {'knee1.0'}
        v1 = True
        if force_v1 or (meta["data_order"] == "whd"):  # to dhw
            images = np.transpose(images, (2, 1, 0))
        else:
            assert meta["data_order"] == "dhw", f'{meta["data_order"]} is not supported now'
    else:
        raise NotImplementedError(f"no version info is not supported currently")
    # in v1, direction of ** and ** has flipped for itk-snap support
    T_pix2world = nifti_image.affine.copy()
    # not needed
    if v1:
        T_pix2world[0] *= -1
        T_pix2world[1] *= -1
    T_world2pix = np.linalg.inv(T_pix2world)
    # to DHW
    # images = images.astype(np.int16)  # WHD->DHW
    T_nat2torch = get_T_natural_to_torch(images.shape)
    T_torch2nat = get_T_torch_to_natural(images.shape)
    # spacing_xyz
    spacing = np.linalg.norm(nifti_image.affine[:3, :3], axis=0)
    series = DummySeries(images, T_pix2world, T_world2pix, T_nat2torch, T_torch2nat)
    series.spacing_xyz = spacing
    # assert np.array_equal(nifti_image.affine.astype(np.float32), T_pix2world), "Affine matrices are not equal."
    series.affine = T_pix2world
    series.affine_itk = nifti_image.affine
    # assert np.array_equal(series.affine, series.affine_itk),"Affine matrices are not equal."
    series = DummySeries3(series)
    return series 

def create_nifti_series2(nii_file, force_v1=False):
    if isinstance(nii_file, str):
        nifti_image = nib.load(nii_file)
    else:
        nifti_image = nii_file
    images = np.asanyarray(nifti_image.dataobj)
    meta = _parse_nifti_meta(nifti_image)
    v1 = False
    if meta or force_v1:
        if not force_v1:
            assert meta["version"] in {'knee1.0'}
        v1 = True
        if force_v1 or (meta["data_order"] == "whd"):  # to dhw
            images = np.transpose(images, (2, 1, 0))
        else:
            assert meta["data_order"] == "dhw", f'{meta["data_order"]} is not supported now'
    else:
        raise NotImplementedError(f"no version info is not supported currently")
    # in v1, direction of ** and ** has flipped for itk-snap support
    T_pix2world = nifti_image.affine.copy()
    # not needed
    if v1:
        T_pix2world[0] *= -1
        T_pix2world[1] *= -1
    T_world2pix = np.linalg.inv(T_pix2world)
    # to DHW
    images = images.astype(np.int16)  # WHD->DHW
    T_nat2torch = get_T_natural_to_torch(images.shape)
    T_torch2nat = get_T_torch_to_natural(images.shape)
    # spacing_xyz
    spacing = np.linalg.norm(nifti_image.affine[:3, :3], axis=0)
    series = DummySeries(images, T_pix2world, T_world2pix, T_nat2torch, T_torch2nat)
    series.spacing_xyz = spacing
    series.affine_raw = nifti_image.affine
    series.affine_itk = nifti_image.affine
    if v1:
        series.view = ViewClassifier().classify_from_affine(T_pix2world)
    return series


class RegistrationByWorldCoordinate(object):
    def __init__(self):
        self.device, self.dtype = "cpu", torch.float

    @staticmethod
    def get_theta_1_to_2(series1, series2):
        T = series2.T_nat2torch @ series2.T_world2pix @ series1.T_pix2world @ series1.T_torch2nat
        return T

    def get_theta_2_to_1(self, series1, series2):
        return self.get_theta_1_to_2(series2, series1)

    def to_tensor(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        elif isinstance(x, torch.Tensor):
            x = x
        else:
            raise NotImplementedError(f"{type(x)}")
        return x.to(device=self.device, dtype=self.dtype)

    def create_normalized_series(self, series, target_spacing):
        shape = series.images.shape
        new_shape = [int(round(abs(dim * sp / target_spacing))) for dim, sp in zip(shape, series.spacing_xyz[::-1])]
        # series.
        # images = F.interpolate(self.to_tensor(series.images)[None, None],
        #                        size=new_shape, mode="trilinear", align_corners=False)[0, 0]
        images = self.resize(self.to_tensor(series.images), new_shape).cpu().numpy()
        scale_factor = np.array([ori / new for ori, new in zip(shape[::-1], new_shape[::-1])] + [1.])
        T_new2ori = np.diag(scale_factor)
        T_ori2new = np.diag(1 / scale_factor)
        # update pix world transform
        T_pix2world = series.T_pix2world @ T_new2ori
        T_world2pix = T_ori2new @ series.T_world2pix
        # update torch transform
        T_nat2torch = get_T_natural_to_torch(new_shape)
        T_torch2nat = get_T_torch_to_natural(new_shape)

        return DummySeries(images, T_pix2world, T_world2pix, T_nat2torch, T_torch2nat)

    def create_crop_and_resize_series(self, series, roi, dst_size=None):
        z1, y1, x1, z2, y2, x2 = roi
        images = series.images[z1:z2, y1:y2, x1:x2]
        crop_size = images.shape
        if dst_size is not None:
            images = self.resize(images, dst_size).type(torch.int16).cpu().numpy()
        else:
            dst_size = crop_size
        # todo: finish this
        sf_xyz = np.array([ds/ps for ds, ps in zip(dst_size, crop_size)][::-1])
        scale = np.diag(sf_xyz)  # xyz
        botleft = np.array([x1, y1, z1])
        offset = -scale @ botleft
        T_ori2new = np.hstack([scale,  offset[:, np.newaxis]])
        T_ori2new = np.vstack([T_ori2new, np.array([[0, 0, 0, 1]])])
        T_new2ori = np.linalg.inv(T_ori2new)

        # update pix world transform
        T_pix2world = series.T_pix2world @ T_new2ori
        T_world2pix = T_ori2new @ series.T_world2pix
        # update torch transform
        T_nat2torch = get_T_natural_to_torch(dst_size)
        T_torch2nat = get_T_torch_to_natural(dst_size)
        new_series = DummySeries(images, T_pix2world, T_world2pix, T_nat2torch, T_torch2nat)
        new_series.spacing_xyz = [a / b for a, b in zip(series.spacing_xyz, sf_xyz)]
        return new_series

    def resize(self, images, dst_size):
        images = self.to_tensor(images)
        return F.interpolate(images[None, None], dst_size, mode="trilinear", align_corners=False)[0, 0]

    def image1_to_2(self, series1, series2, src=None, mode="bilinear"):
        theta = self.get_theta_2_to_1(series1, series2)
        if src is None:
            src = series1.images
        src = self.to_tensor(src)
        theta = self.to_tensor(theta)[:3].unsqueeze(0)
        grid = F.affine_grid(theta, (1, 1) + tuple(series2.shape), align_corners=False)
        kwargs = {"align_corners": False} #if mode == "bilinear" else {}
        out = F.grid_sample(src[None, None], grid, mode=mode, **kwargs)[0, 0]
        return self.to_numpy_int16(out)

    def to_numpy_int16(self, t):
        if isinstance(t, torch.Tensor):
            t = t.round().type(torch.int16).cpu().numpy()
        elif isinstance(t, np.ndarray):
            t = t.astype(np.int16)
        else:
            raise NotImplementedError(f"{type(t)}")
        return t
    



def main():

    radio = 0.9

    file_paths = glob.glob("./extracted_data/cor_pd-sag_pd/R.json")

    base_root_1 = "/mnt/LungNoduleHDD/liufeng/knee/fast_mri/ser_nii"#/mnt/LungNoduleHDD/liufeng/knee/fast_mri/ser_nii/3836133/1.2.840.113654.2.70.1.94263270961723630827740372146420913071/1.2.840.113654.2.70.1.278329345154130539583669740637402214083.nii.gz
    base_root_2 = "/mnt/LungNoduleHDD/liufeng/knee/multicenter/nii" #/mnt/LungNoduleHDD/liufeng/knee/fast_mri/ser_nii/3827358/9999.202039550284634041753470433114927182625/9999.205506686136633679113938041957701726239.nii.gz
    base_root_3 = "/mnt/LungNoduleHDD/liufeng/knee/pla/nii2"
    base_root_4 = "/mnt/LungNoduleHDD/liufeng/knee/pla2/nii"

    save_path_pd_train = "/mnt/users/read_side_and_weighting_data/registrated_image/cor_pd-sag_pd/train/cor_to_sag/"  #pd代表被配准的
    save_path_pdfs_train = "/mnt/users/read_side_and_weighting_data/registrated_image/cor_pd-sag_pd/train/sag/"       #pdfs代表不变的
    save_path_before_pd_train = "/mnt/users/read_side_and_weighting_data/registrated_image/cor_pd-sag_pd/train/cor/"

    save_path_pd_val = "/mnt/users/read_side_and_weighting_data/registrated_image/cor_pd-sag_pd/val/cor_to_sag/"
    save_path_pdfs_val = "/mnt/users/read_side_and_weighting_data/registrated_image/cor_pd-sag_pd/val/sag/"
    save_path_before_pd_val = "/mnt/users/read_side_and_weighting_data/registrated_image/cor_pd-sag_pd/val/cor/"

    registrator = RegistrationByWorldCoordinate()


    def find_file(file_name, search_dirs):
        for base_dir in search_dirs:
            for root, dirs, files in os.walk(base_dir):
                if file_name.split('/')[0] in dirs:
                    dir = file_name.split('/')[0]
                    for file in os.listdir(os.path.join(root,dir)):
                        if file_name.rsplit('/',1)[0] == dir+'/' + file:
                            return os.path.join(root, file_name) , base_dir
        return None,None

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = json.load(file)
            cut = int(len(data)*radio)
            idx = 0            
            if isinstance(data, dict):
                for key, value in data.items():
                    if os.path.isdir(os.path.join(save_path_pd_train,key)) or os.path.isdir(os.path.join(save_path_pdfs_train,key)) or os.path.isdir(os.path.join(save_path_pd_val,key)) or os.path.isdir(os.path.join(save_path_pdfs_val,key)):
                        assert('该病人已存在！')
                    pd_fs = value['SAG_PD'] #'3835432/1.2.840.113654.2.70.1.187071049621839892209445787731477777075/1.2.840.113654.2.70.1.68773205733919536174017230383350566712'
                    pd = value['COR_PD']

                    base_roots = [base_root_1, base_root_2, base_root_3, base_root_4]

                    f1,base_root_pd_fs = find_file(pd_fs+'.nii.gz', base_roots)
                    f2,base_root_pd = find_file(pd+'.nii.gz', base_roots)
                    # print('base_root_pd_fs:',base_root_pd_fs,'      ','base_root_pd:',base_root_pd)

                    if f1 is not None and f2 is not None: #保证同一个study的pd和pd_fs来自同一个中心??
                        try:
                            series1,pd_fs_shape = create_nifti_series(f1, force_v1=True)
                            series2,pd_shape = create_nifti_series(f2, force_v1=True)   #pd代表被配准的/cor                        
                            # print('pd_fs_shape:',pd_fs_shape,'      ','pd_shape:',pd_shape)
                        except:
                            print('error:no files-',f1)
                            continue

                        if (np.abs(series1.affine_itk - series2.affine_itk) == 0).all():  # identical affine matrix 判断仿射矩阵是否相同，如果相同就不需要配准了
                            print(f"identical affine matrix, skip registration")
                            images_2to1 = series2.images
                            images_1 = series1.images
                            images_2 = series2.images
                        else:
                            images_2to1 = registrator.image1_to_2(series2, series1, mode="bilinear")  #配准 cor_to_sag
                            images_1 = series1.images
                            images_2 = series2.images

                        if idx < cut:                            
                            os.makedirs(save_path_pd_train + pd.rsplit('/', 1)[0],exist_ok=True)
                            os.makedirs(save_path_pdfs_train + pd_fs.rsplit('/', 1)[0],exist_ok=True)
                            os.makedirs(save_path_before_pd_train + pd.rsplit('/', 1)[0],exist_ok=True)
                            save_nifti(images_2to1, series1.affine_itk, save_path_pd_train + pd+'.nii.gz')
                            save_nifti(images_1, series1.affine_itk, save_path_pdfs_train + pd_fs+'.nii.gz')
                            save_nifti(images_2, series2.affine_itk, save_path_before_pd_train + pd+'.nii.gz')
                        else:
                            os.makedirs(save_path_pd_val + pd.rsplit('/', 1)[0],exist_ok=True)
                            os.makedirs(save_path_pdfs_val + pd_fs.rsplit('/', 1)[0],exist_ok=True)
                            os.makedirs(save_path_before_pd_val + pd.rsplit('/', 1)[0],exist_ok=True)
                            save_nifti(images_2to1, series1.affine_itk, save_path_pd_val + pd+'.nii.gz')
                            save_nifti(images_1, series2.affine_itk, save_path_pdfs_val + pd_fs+'.nii.gz')
                            save_nifti(images_2, series2.affine_itk, save_path_before_pd_val + pd+'.nii.gz')                       
                    else:
                        print('not find file!')

                    idx += 1 
                    print(idx)


if __name__ == '__main__':
    main()


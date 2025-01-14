import os
import sys
import math
import glob
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from skimage.measure import regionprops, label
from .register_world_coordinate import create_nifti_series

# sys.path.insert(0, "/mnt/LungNoduleHDD/liufeng/knee/scripts/knee_mri")
# from utils.lmdb_writer import PngLMDBWriter
# from utils.series import create_nifti_series
# from bone_contusion.extract_bone_contusion_patch import save_nifti
# from bone_contusion.extract_bone_contusion_patch import list_bone_contusion_annotations, get_image_files, \
#     align_segmentation



def _center_dim_to_xyxy(center, dims):
    center = np.array(center)
    dims = np.array(dims)
    start = center - dims // 2
    stop = start + dims
    return start.tolist() + stop.tolist()



def _box_a_inside_b(a, b):
    return all((a[i] >= b[i]) & (a[i + 3] <= b[i + 3]) for i in range(3))


def ceil_int(x):
    return int(math.ceil(x))


def resize_torch(x, new_shape, is_mask=False):
    dtype = x.dtype
    if is_mask:
        unk_lbls = x.unique()
        out = []
        for l in unk_lbls:
            out.append(
                F.interpolate((x[None, None] == l).float(), new_shape, mode="trilinear", align_corners=False)[0, 0]
            )
        out = torch.stack(out, dim=0).argmax(dim=0)
        out = unk_lbls[out]
        return out.type(dtype)
    else:
        out = F.interpolate(x[None, None,...], new_shape, mode="trilinear", align_corners=False)
        return out


def update_affine_np(affine, roi, dst_size):  # todo: improve naming
    if isinstance(roi, torch.Tensor):
        roi = roi.tolist()
    z1, y1, x1, z2, y2, x2 = roi
    crop_size = [z2 - z1, y2 - y1, x2 - x1]
    sf_xyz = np.array([ds / ps for ds, ps in zip(dst_size, crop_size)][::-1])
    scale = np.diag(sf_xyz)  # xyz
    botleft = np.array([x1, y1, z1])
    offset = -scale @ botleft
    T_ori2new = np.hstack([scale, offset[:, np.newaxis]])
    T_ori2new = np.vstack([T_ori2new, np.array([[0, 0, 0, 1]])])
    T_new2ori = np.linalg.inv(T_ori2new)

    # update pix world transform
    new_affine = affine @ T_new2ori
    new_affine = torch.from_numpy(new_affine)
    return new_affine

def update_affine(affine, roi, dst_size):  # todo: improve naming
    if not isinstance(roi, torch.Tensor):
        roi = torch.tensor(roi)
    z1, y1, x1, z2, y2, x2 = roi
    crop_size = [z2 - z1, y2 - y1, x2 - x1]
    sf_xyz = torch.tensor([ds / ps for ds, ps in zip(dst_size, crop_size)][::-1])
    scale = torch.diag(sf_xyz)  # xyz
    botleft = torch.tensor([x1, y1, z1]).float()
    offset = -scale @ botleft
    T_ori2new = torch.hstack([scale, offset[:, None]])
    T_ori2new = torch.vstack([T_ori2new, torch.tensor([[0., 0., 0., 1.]])])
    T_new2ori = torch.linalg.inv(T_ori2new)

    # update pix world transform
    new_affine = affine @ T_new2ori.to(affine.device)
    return new_affine

class OutOfImageException(Exception):
    pass


class DualSegExtractor(object):
    def __init__(self, patch_size=(16, 128, 128), device="cuda"):
        self.device = device
        self.patch_size = patch_size

    def _crop_roi(self, image, crop_box):
        z1, y1, x1, z2, y2, x2 = crop_box
        crop_w, crop_h, crop_d = x2 - x1, y2 - y1, z2 - z1
        assert crop_d == self.patch_size[0]
        out = image.new_zeros((crop_d, crop_h, crop_w))
        max_d, max_h, max_w = image.shape
        ct_x1, ct_y1, ct_z1 = max(x1, 0), max(y1, 0), max(z1, 0)
        pt_x1, pt_y1, pt_z1 = ct_x1 - x1, ct_y1 - y1, ct_z1 - z1
        _w, _h, _d = min(x2, max_w) - ct_x1, min(y2, max_h) - ct_y1, min(z2, max_d) - ct_z1
        out[pt_z1:pt_z1 + _d, pt_y1:pt_y1 + _h, pt_x1:pt_x1 + _w] = image[ct_z1:ct_z1 + _d,
                                                                    ct_y1:ct_y1 + _h,
                                                                    ct_x1:ct_x1 + _w]
        return out

    def _make_stat(self, image):
        # todo: change max to 99.8% percentile
        minv, maxv = image.min().item(), image.max().item()
        mu, std = image.mean().item(), image.std().item()
        return {"min": minv, "max": maxv, "mu": np.round(mu, 3), "std": np.round(std, 3)}

    @staticmethod
    def _to_numpy(x, dtype=torch.int16):
        return x.round().type(dtype).cpu().numpy()

    def _extract_single(self, image, crop_box):
        patch_im = resize_torch(self._crop_roi(image, crop_box), self.patch_size) #.clamp_(0, 32767)
        return patch_im
        # patch_ms = resize_torch(self._crop_roi(mask, crop_box), self.patch_size, True)
        # return self._to_numpy(patch_im)#, self._to_numpy(patch_ms)

    @staticmethod
    def _out_of_image(image, crop_box):
        crop_box_tensor = torch.tensor(crop_box)
        ub = torch.tensor(image.shape).to(crop_box_tensor)
        crop_box = crop_box_tensor.reshape(2, 3)
        inside = (crop_box.amax(dim=0) > 0).all() & (crop_box.amin(dim=0) < ub).all()
        return not inside.item()

    def wrap_image(self, image):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        return image.to(device=self.device, dtype=torch.float32)

    def _name_patch(self, crop_box):
        bbox = crop_box.float().tolist()
        return "z1{:.0f}y1{:.0f}x1{:.0f}z2{:.0f}y2{:.0f}x2{:.0f}".format(*bbox)

    def __call__(self, series1, series2, mask1 = None, mask2 = None, crops = None):
        patches, meta = [], []
        image1 = self.wrap_image(series1.images)
        image2 = self.wrap_image(series2.images)

        nfail = 0
        # key_crops, alt_crops = crops["key_crops"], crops["alt_crops"]
        # for crop_box1, crop_box2 in zip((crops), (crops)):
        crop_box1 = crops[0]
        crop_box2 = crops[1]

        if self._out_of_image(image1[0,0,...], crop_box1) or self._out_of_image(image2[0,0,...], crop_box2):
            # print(f"box out of image: v1 {image1.shape} {crop_box1}, v2 {image2.shape} {crop_box2}")
            # nfail += 1
            # continue
            assert True,'box out of image!'
        patch_im1 = self._extract_single(image1[0,0,...], crop_box1)
        affine1 = update_affine(series1.T_pix2world, crop_box1, self.patch_size).to(image1.device)
        patch_im2 = self._extract_single(image2[0,0,...], crop_box2)
        affine2 = update_affine(series2.T_pix2world, crop_box2, self.patch_size).to(image2.device)
        series1.images,series1.affine,series1.affine_itk = patch_im1,affine1,affine1
        series2.images,series2.affine,series2.affine_itk = patch_im2,affine2,affine2
        return series1,series2

        # todo: check image out of bounds
        # update affine matrix
        # from fast_mri.draw_anno import save_nifti
        # save_nifti(patch_im1, affine1, "bone_contusion/vis/im1.nii.gz")
        # save_nifti(patch_ms1, affine1, "bone_contusion/vis/ms1.nii.gz")
        # save_nifti(patch_im2, affine2, "bone_contusion/vis/im2.nii.gz")
        # save_nifti(patch_ms2, affine2, "bone_contusion/vis/ms2.nii.gz")
        # import pdb; pdb.set_trace()
    #     key_label = (patch_ms1 == 1).any()
    #     patches.append({"v1": (patch_im1, patch_ms1), "v2": (patch_im2, patch_ms2)})
    #     meta.append({"label": int(key_label), "name": self._name_patch(crop_box1),
    #                  "crop_box1": crop_box1,
    #                  "affine_v1": affine1, "affine_v2": affine2})
        # stat1 = self._make_stat(image1)
        # stat2 = self._make_stat(image2)
        # succeed = nfail / len(key_crops) < 0.3


        # return patches, meta, {"v1": stat1, "v2": stat2}, succeed


class Stage2DualSegExtractor(DualSegExtractor):
    def _apply_region_mask(self, seed_mask, crop_box, crop_label, key_mask_crop):
        patch_ms = resize_torch(self._crop_roi((seed_mask == crop_label).to(seed_mask), crop_box), self.patch_size, False)
        active_region = self._to_numpy(patch_ms) > 0.5
        key_label = (key_mask_crop[active_region] == 1).any()
        # 1. in active region, the same as GT
        # 2. outside of active region: 0 -> 254; 1->253, 255->255
        mapping = np.full((256,), 255, dtype=key_mask_crop.dtype)
        mapping[0] = 254
        mapping[1] = 253
        out = np.where(active_region, key_mask_crop, mapping[key_mask_crop])
        return out, key_label

    def __call__(self, series1, series2, mask1, mask2, crops, seed_mask):
        patches, meta = [], []
        image1, mask1, seed_mask = self.wrap_image(series1.images), self.wrap_image(mask1), self.wrap_image(seed_mask)
        image2, mask2 = self.wrap_image(series2.images), self.wrap_image(mask2)

        nfail = 0
        key_crops, alt_crops, crop_labels = crops["key_crops"], crops["alt_crops"], crops["key_labels"]
        for crop_box1, crop_box2, key_ridx in zip(key_crops, alt_crops, crop_labels):
            if self._out_of_image(image1, crop_box1) or self._out_of_image(image2, crop_box2):
                print(f"box out of image: v1 {image1.shape} {crop_box1}, v2 {image2.shape} {crop_box2}")
                nfail += 1
                continue
            patch_im1, patch_ms1 = self._extract_single(image1, mask1, crop_box1)
            patch_ms1, key_label = self._apply_region_mask(seed_mask, crop_box1, key_ridx, patch_ms1)
            if (patch_ms1 > 250).all():
                print(f"warning: all values in patch_ms1 > 250, {np.unique(patch_ms1)}")
                continue
            affine1 = update_affine(series1.T_pix2world, crop_box1, self.patch_size)
            # save_nifti(patch_im1, affine1, "bone_contusion/vis/im1.nii.gz")
            # save_nifti(patch_ms1, affine1, "bone_contusion/vis/ms1.nii.gz")
            patch_im2, patch_ms2 = self._extract_single(image2, mask2, crop_box2)
            affine2 = update_affine(series2.T_pix2world, crop_box2, self.patch_size)
            # key_label = (patch_ms1 == 1).any()
            patches.append({"v1": (patch_im1, patch_ms1), "v2": (patch_im2, patch_ms2)})
            meta.append({"label": int(key_label), "name": self._name_patch(crop_box1),
                         "crop_box1": crop_box1,
                         "affine_v1": affine1, "affine_v2": affine2})
        stat1 = self._make_stat(image1)
        stat2 = self._make_stat(image2)
        succeed = nfail / len(key_crops) < 0.3
        return patches, meta, {"v1": stat1, "v2": stat2}, succeed


class ScheduleCrops(object):
    def __init__(self, dst_dim_xy=128, dst_spacing_xy=0.3, depth=16, device="cuda:0"):
        self.dst_spacing_xy = dst_spacing_xy
        self.depth = depth
        self.dst_dim_xy = dst_dim_xy
        self.device = device
        self.stride_norm = (32, 32, 4)

    def _get_kernel_size(self, series_key):
        sx, sy, _ = series_key.spacing_xyz
        kw = int(round(self.dst_dim_xy * self.dst_spacing_xy / sx))
        kh = int(round(self.dst_dim_xy * self.dst_spacing_xy / sy))
        return [self.depth, kh, kw]

    def _get_stride(self, series_key):
        sx, sy, _ = series_key.spacing_xyz
        dx, dy, dz = self.stride_norm
        sw = int(round(dx * self.dst_spacing_xy / sx))
        sh = int(round(dy * self.dst_spacing_xy / sy))
        return [dz, sh, sw]

    def _compute_padding(self, kernel_size, stride, input_size):
        padding = []
        for k, s, i in zip(kernel_size, stride, input_size):
            o_sub1 = ceil_int((i - k) / s)
            p = ceil_int((o_sub1 * s + k - i) / 2)
            padding.append(p)
        return padding

    def _decode_conv_points(self, cond_map, kernel_size, stride, padding, frac=None):
        out_coords = cond_map.nonzero(as_tuple=False)
        if frac is not None:
            n = len(out_coords)
            inds = np.random.choice(n, size=(int(n*frac),), replace=False)
            out_coords = out_coords[torch.from_numpy(inds).to(device=self.device, dtype=torch.long)]
        start = out_coords * torch.tensor(stride).to(out_coords) - torch.tensor(padding).to(out_coords)
        stop = start + torch.tensor(kernel_size).to(start)
        return torch.cat([start, stop], dim=1)

    def _schedule_sliding_window(self, series_key, mask_key):
        kernel_size = self._get_kernel_size(series_key)
        stride = self._get_stride(series_key)
        padding = self._compute_padding(kernel_size, stride, mask_key.shape)
        mask = torch.from_numpy(mask_key == 1).to(dtype=torch.float32, device=self.device)
        kernel = torch.ones((1, 1, *kernel_size)).to(dtype=torch.float32, device=self.device)
        out = F.conv3d(mask[None, None], kernel, stride=stride, padding=padding)[0, 0]
        return torch.cat([
            self._decode_conv_points(out, kernel_size, stride, padding),
            self._decode_conv_points(out == 0, kernel_size, stride, padding, 0.1),
        ], dim=0)

    def add_center_window(self, series_key, mask_key):
        regions = regionprops(label((mask_key == 1).astype(np.uint8)))
        kernel_size = self._get_kernel_size(series_key)
        center = torch.tensor([r.centroid for r in regions]).reshape(-1, 3).to(device=self.device)
        return self._generate_crops_from_center_and_shape(center, kernel_size)

    @staticmethod
    def _map_crop_box_to_2nd(crop_box, T):
        z1, y1, x1, z2, y2, x2 = crop_box
        rng = [[x1, x2], [y1, y2], [z1, z2]]
        src_points = []
        for x in rng[0]:
            for y in rng[1]:
                for z in rng[2]:
                    src_points.append([x, y, z])
        src_points = np.array(src_points)
        # map
        src_points = np.concatenate([src_points, np.ones((len(src_points), 1))], axis=1)
        dst_points = T @ src_points.T
        dst_points = dst_points[:3].T
        # get envelope
        x1, y1, z1 = dst_points.min(axis=0)
        x2, y2, z2 = dst_points.max(axis=0)
        return np.array([z1, y1, x1, z2, y2, x2])

    @staticmethod
    def map_crops_to_another_view(key_crops, T):
        z1, y1, x1, z2, y2, x2 = key_crops.unbind(dim=1)
        ones = torch.ones_like(z1)
        rng = [[x1, x2], [y1, y2], [z1, z2]]
        src_points = []
        for x in rng[0]:
            for y in rng[1]:
                for z in rng[2]:
                    src_points.append(torch.stack([x, y, z, ones], dim=1))
        src_points = torch.stack(src_points, dim=0).float()
        dst_points = (T @ src_points.view(-1, 4).T)[:3].T
        dst_points = dst_points.reshape(8, -1, 3).flip(2)  # xyz->zyx
        return torch.cat([dst_points.amin(dim=0), dst_points.amax(dim=0)], dim=1)

    def compute_crops_in_alter_view(self, key_crops, series_key, series_alter):
        if len(key_crops) == 0:
            return key_crops
        T = torch.from_numpy(series_alter.T_world2pix @ series_key.T_pix2world).to(device=self.device, dtype=torch.float32)
        dst_crops = self.map_crops_to_another_view(key_crops, T)
        # compare with original implementation
        # out = []
        # for crop_box in key_crops.cpu().numpy():
        #     out.append(self._map_crop_box_to_2nd(crop_box, T.cpu().numpy()))
        # out = torch.from_numpy(np.stack(out, axis=0)).to(dst_crops)
        # torch.allclose(dst_crops, out)

        # to required shape
        kernel_size = self._get_kernel_size(series_alter)
        centers = 0.5 * (dst_crops[:, :3] + dst_crops[:, 3:])
        dims = dst_crops[:, 3:] - dst_crops[:, :3]
        illegal = ((dims - torch.tensor([kernel_size]).to(centers)) < 0).any(dim=1)
        # adjust kernel size if it is smaller than mapped size
        max_sizes = dims[:, 1:].amax(dim=1)
        # print(max_sizes.unique())
        # if illegal.any():
        #     print(dims[illegal])
        kernel_size = (kernel_size[0], max_sizes.amax().item(), max_sizes.amax().item())
        # print(kernel_size, self._get_kernel_size(series_alter))
        return self._generate_crops_from_center_and_shape(centers, kernel_size)

    @staticmethod
    def _generate_crops_from_center_and_shape(centers, kernel_size):
        start = (centers - torch.tensor([kernel_size]).to(centers) * 0.5).long()
        stop = start + torch.tensor(kernel_size).to(start)
        return torch.cat([start, stop], dim=1)

    def __call__(self, series_key, mask_key, series_alter):
        # key_crops = torch.cat([
        #     self._schedule_sliding_window(series_key, mask_key),
        #     self.add_center_window(series_key, mask_key)
        # ], dim=0)
        key_crops = self.add_center_window(series_key, mask_key)
        alt_crops = self.compute_crops_in_alter_view(key_crops, series_key, series_alter)
        return {"key_crops": key_crops, "alt_crops": alt_crops}


class ScheduleCropsAsStage2(ScheduleCrops):
    boarder = [2, 16, 16]

    def _get_boarder(self, series_key):
        sx, sy, _ = series_key.spacing_xyz
        return [
            self.boarder[0],
            int(math.ceil(self.boarder[1] * self.dst_spacing_xy / sy)),
            int(math.ceil(self.boarder[2] * self.dst_spacing_xy / sx))
        ]

    def _schedule_1d(self, lb, ub, ks, s=None):
        length = ub - lb
        if length <= ks:
            start = (lb + ub) // 2 - ks // 2
            # return [[start, start + ks]]
            return [start + ks // 2]
        else:  # needs schedule
            if s is None:
                nws = math.ceil(length / ks)
                s = int(math.ceil((length - ks) / (nws - 1)))
            else:
                nws = math.ceil((length - ks) / s + 1)
            p = (ks + s * (nws - 1) - length) // 2  # per side
            out = []
            for i in range(nws):
                start = -p + i * s + lb
                stop = start + ks
                # out.append([start, stop])
                out.append((start + stop) // 2)
        return out

    def _schedule_roi_crops(self, region, kernel_size, boarder):
        eff_size = [ks - 2 * bd for ks, bd in zip(kernel_size, boarder)]
        bbox = np.array(region.bbox)
        cz, cy, cx = region.centroid
        centroid_box = _center_dim_to_xyxy([cz, cy, cx], eff_size)
        if _box_a_inside_b(bbox, centroid_box):
            return torch.tensor([region.centroid]).to(device=self.device)
        else:  # sliding window
            zs = self._schedule_1d(bbox[0], bbox[3], eff_size[0])
            ys = self._schedule_1d(bbox[1], bbox[4], eff_size[1])
            xs = self._schedule_1d(bbox[2], bbox[5], eff_size[2])
            return torch.tensor([(z, y, x) for z in zs for y in ys for x in xs]).to(device=self.device)

    def add_center_window(self, series_key, seed_mask_key, gt_mask_key):
        # regions = regionprops(label((seed_mask_key == 1).astype(np.uint8)))
        regions = regionprops(seed_mask_key)  # input must be ccs
        # filter region only contains ignore
        valid_regions = []
        for r in regions:
            vals = gt_mask_key[r.slice][r.image]
            if vals.any():
                vals = np.unique(vals[vals > 0]).tolist()  # fg voxels
                if 1 not in vals:  # seed region only contains ignore
                    continue
            valid_regions.append(r)
        if len(valid_regions) != len(regions):
            print(f"removed: {len(regions) - len(valid_regions)} regions of all ignore")
        if len(valid_regions) == 0:
            print(f"warning: no regions")
            return [], []
        kernel_size = self._get_kernel_size(series_key)
        boarder = self._get_boarder(series_key)
        crops = [self._schedule_roi_crops(r, kernel_size, boarder) for r in valid_regions]
        labels = [torch.full((len(_crops),), r.label, device=_crops.device) for _crops, r in zip(crops, valid_regions)]
        center, labels = torch.cat(crops, dim=0), torch.cat(labels, dim=0)
        return self._generate_crops_from_center_and_shape(center, kernel_size), labels

    def __call__(self, series_key, seed_mask_key, series_alter, gt_mask_key):
        key_crops, key_labels = self.add_center_window(series_key, seed_mask_key, gt_mask_key)
        assert len(key_labels) == len(key_crops)
        alt_crops = self.compute_crops_in_alter_view(key_crops, series_key, series_alter)
        return {"key_crops": key_crops, "alt_crops": alt_crops, "key_labels": key_labels}


class PatchSaver(object):
    def __init__(self, save_root):
        os.makedirs(save_root, exist_ok=True)
        self.save_root = save_root
        self.writer = PngLMDBWriter(f"{save_root}/images", value_offset=0)
        self.views = ["v1", "v2"]
        self.stat_fields = ["min", "max", "mu", "std"]
        self.columns = ["key", "label"] + [f"{v}_{f}" for v in self.views for f in self.stat_fields]
        self.data = {k: [] for k in self.columns}
        self.affine_records = {}

    def add_item(self, patch_i, sub, meta_i, stat):
        vals = [f'{sub}/{meta_i["name"]}', meta_i["label"]]
        suffs = ["", "_mask"]
        for view in self.views:
            for patch, suffix in zip(patch_i[view], suffs):
                key = f'{sub}/{meta_i["name"]}_{view}{suffix}'
                self.affine_records[key] = meta_i[f"affine_{view}"]
                self.writer.save(key, patch)
                vals += [stat[view][k] for k in self.stat_fields]
        for k, v in zip(self.columns, vals):
            self.data[k].append(v)

    def add(self, patches, sub, meta, stat):
        assert len(patches) == len(meta)
        for patch_i, meta_i in zip(patches, meta):
            self.add_item(patch_i, sub, meta_i, stat)

    def save_meta(self, name="list"):
        import pandas as pd
        df = pd.DataFrame(self.data, columns=self.columns)
        df.to_csv(f"{self.save_root}/{name}.csv", header=True, index=False)
        #  save affine recordsd
        torch.save(self.affine_records, f"{self.save_root}/affine_records.pth")


def load_fused_stage1_segmentation(study):
    root = "/mnt/LungNoduleHDD/liufeng/knee/func/bone_contusion/pseudo/xval/merged/"
    study = study.replace("/", "_")
    files = glob.glob(os.path.join(root, f"{study}_*mask.nii.gz"))
    series = [create_nifti_series(f, True) for f in files]
    return {s.view: s.images for s in series}


def compose_pred_gt_mask(prop_mask, gt_mask):
    from skimage.morphology import binary_dilation, ball
    selem = np.ones((3, 9, 9), dtype=np.uint8)
    prop = (1 - binary_dilation(prop_mask > 0, selem=selem).astype(gt_mask.dtype)) * 254
    out = np.where(gt_mask > 0, gt_mask, prop)  # change background to 254
    return out


def main():
    def _get_patstu(x):
        return x.replace("/", "_")

    as_stage2 = True
    save_root = "/mnt/lungNoduleGroup/liufeng/knee/segment/bone_contusion/3d2nd_fix_reg"
    os.makedirs(save_root, exist_ok=True)
    saver = PatchSaver(save_root)

    im_root = "/mnt/LungNoduleHDD/liufeng/knee/func/meniscus/nii/"
    seg_files = list_bone_contusion_annotations()
    _, im_studies = get_image_files(im_root)
    im_studies = {k: v for k, v in im_studies.items() if k.replace("/", "_") in seg_files}
    studies = sorted(im_studies.keys())

    init_segs = {}
    crop_scheduler = ScheduleCropsAsStage2() if as_stage2 else ScheduleCrops()
    extractor = Stage2DualSegExtractor() if as_stage2 else DualSegExtractor()
    saved = False
    for study in tqdm(studies):
        pa_stu_key = _get_patstu(study)
        if len(im_studies[study]) != 2:
            print(f"{study}: {len(im_studies[study])} files found")
            continue
        cur_seg_files = seg_files[pa_stu_key]
        if len(cur_seg_files) < 2:
            print(f"skip miss seq")
            continue
        aligned_masks = align_segmentation(cur_seg_files, im_studies[study])
        im_series_dual = [create_nifti_series(imf, force_v1=True) for imf in im_studies[study]]
        im_series_dual = {s.view: s for s in im_series_dual}
        views = {"sagittal", "coronal"}
        if any(v not in im_series_dual for v in views):
            print(f"{study}: missing view, {im_series_dual.keys()}")
            continue
        if as_stage2:
            init_segs = load_fused_stage1_segmentation(study)
        for view in views:
            series1, mask1 = im_series_dual[view], aligned_masks[view]
            if (not as_stage2) and (mask1 != 1).all():
                print(f"{study}/{view} all negative, skip")
                continue
            alt_view = (views - {view}).pop()
            series2, mask2 = im_series_dual[alt_view], aligned_masks[alt_view]
            if as_stage2:  # use merged prediction to schedule crops
                prop_mask = init_segs.get(view, mask1)
                if not prop_mask.any():
                    print(f"{study}/{view} all negative, skip")
                    continue
                prop_ccs = label(prop_mask)
                crops = crop_scheduler(series1, prop_ccs, series2, mask1)
                if len(crops["key_crops"]) == 0:
                    continue
                if not saved:
                    save_nifti(mask1, series1.affine_itk, "bone_contusion/vis/gt.nii.gz")
                    save_nifti(prop_mask, series1.affine_itk, "bone_contusion/vis/prop.nii.gz")
                # mask1 = compose_pred_gt_mask(prop_mask, mask1)
                # todo: may keep dilation
                # if not saved:
                #     save_nifti(mask1, series1.affine_itk, "bone_contusion/vis/gt_post.nii.gz")
                #     saved = True
                patches, meta, stat, succeed = extractor(series1, series2, mask1, mask2, crops, prop_ccs)
            else:
                crops = crop_scheduler(series1, mask1, series2)
                if len(crops["key_crops"]) == 0:
                    continue
                patches, meta, stat, succeed = extractor(series1, series2, mask1, mask2, crops)
            if succeed:
                saver.add(patches, f"{pa_stu_key}_{series1.view}", meta, stat)
            else:
                print(f"warning: failed {study}, too many out of scope crops")
    saver.save_meta("list")


def make_train_test_list():
    import pandas as pd
    root = "/mnt/lungNoduleGroup/liufeng/knee/segment/bone_contusion/3d2nd_fix_reg"
    df = pd.read_csv(f"{root}/list.csv")
    # split
    split_df = pd.read_csv(
        "/mnt/LungNoduleHDD/shixiaofei/datasets/knee/train_patch/bone_contusion/new_model_patch_0410/test.csv")

    def _get_study(x):
        return "_".join(x.replace("/", "_").split("_")[:2])

    test_studies = set(split_df["name"].apply(_get_study).tolist())
    # move 3836152 into training set
    is_test = df["key"].apply(lambda x: _get_study(x) in test_studies)
    df_test: pd.DataFrame = df[is_test].reset_index(drop=True)
    df_test.to_csv(f"{root}/test.csv", header=True, index=False)
    df_train = df[~is_test].reset_index(drop=True)
    df_train.to_csv(f"{root}/train.csv", header=True, index=False)
    assert len(set(df_train["key"].tolist()) & set(df_test["key"].tolist())) == 0
    print(len(df_train))
    print(len(df_test))


if __name__ == '__main__':
    main()
    make_train_test_list()

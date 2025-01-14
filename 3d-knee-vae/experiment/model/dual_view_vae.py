import logging
from typing import Sequence, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .transforms_cuda_det import build_cuda_transform
# from .nnunet import UNet3D, UpsampleBlock, ConvBlock
# from timm.utils.metric_seg import compute_stat_from_pred_gt
from .vae import Register_VAE
from .UNet import UNet
from .register import image_register,get_T_natural_to_torch,get_T_torch_to_natural
from .extract_bone_contusion_dualview_3d_patches import DualSegExtractor
from .register_world_coordinate import DummySeries
import nibabel as nib
import numpy as np
import os

from .Dynunet import init_dynunet

try:
    from .registry import register_model
except:
    register_model = lambda x: x

logger = logging.getLogger('train')

def save_nifti(pixel_data, affine, fpath):
    pixel_data = np.transpose(pixel_data, (2, 1, 0))  # WHD
    nifti_img = nib.Nifti1Image(pixel_data, affine)
    nib.save(nifti_img, fpath)

def save_vis(img,aff,path,name):
    img_arry = img[0,0,...].detach().cpu().numpy()
    aff = aff[0,...].detach().cpu().numpy()
    # aff[0] *= -1
    # aff[1] *= -1  # to affine itk
    path = os.path.join(path,name)
    save_nifti(img_arry,aff,path)
    
def update_all(series):
    images = series.images.float() #float默认是32
    T_pix2world = series.affine.clone().float()
    T_world2pix = torch.linalg.inv(T_pix2world).float()
    # to DHW
    T_nat2torch = get_T_natural_to_torch(images.shape).float().to(images.device)
    T_torch2nat = get_T_torch_to_natural(images.shape).float().to(images.device)
    # spacing_xyz
    spacing = torch.linalg.norm(series.affine[:3, :3], axis=0)
    new_series = DummySeries(images, T_pix2world, T_world2pix, T_nat2torch, T_torch2nat)
    new_series.spacing_xyz = spacing.float()
    assert torch.equal(series.affine.float(), T_pix2world), "Affine matrices are not equal."
    new_series.affine = T_pix2world.float()
    new_series.affine_itk = series.affine.float()
    assert torch.equal(series.affine, series.affine_itk),"Affine matrices are not equal."
    return new_series 

def center_crop(ori_shape, target_shape): #z1, y1, x1, z2, y2, x2 = crop_box

    b1,c1,d1,h1,w1 = ori_shape #dhw
    b2,c2,d2,h2,w2 = target_shape

    # 计算中心点的坐标
    start_z = d1 // 2 - (d2 // 2)
    start_y = h1 // 2 - (h2 // 2)    
    start_x = w1 // 2 - (w2 // 2)

    # 确保裁剪区域在图像范围内
    start_z = max(0, start_z)
    start_y = max(0, start_y)
    start_x = max(0, start_x)

    end_z = min(d1, start_z + d2)    
    end_y = min(h1, start_y + h2)
    end_x = min(w1, start_x + w2)

    return [start_z,start_y,start_x,end_z,end_y,end_x]




class RandomCropResizeWithAffine(nn.Module):
    def __init__(self, crop_shape, resize_shape, rand_z=False, max_offset=None):
        super(RandomCropResizeWithAffine, self).__init__()
        self.crop_size = list(crop_shape)
        self.resize_shape = resize_shape
        self.max_offset = max_offset
        self.rand_z = rand_z

    def _get_crop_size(self, x):
        n = len(self.crop_size)
        out = [cdim if cdim != -1 else maxd for cdim, maxd in zip(self.crop_size, x.shape[-n:])]
        return tuple(out)

    def _get_resize_shape(self, x):
        n = len(self.resize_shape)
        out = [cdim if cdim != -1 else maxd for cdim, maxd in zip(self.resize_shape, x.shape[-n:])]
        return tuple(out)

    def _get_crop_params_test(self, x):
        device = x.device
        N, C, D, H, W = x.shape
        pd, ph, pw = self._get_crop_size(x)
        z1s = torch.ones(size=(N,), device=device, dtype=torch.long) * (D - pd) // 2
        y1s = torch.ones(size=(N,), device=device, dtype=torch.long) * (H - ph) // 2
        x1s = torch.ones(size=(N,), device=device, dtype=torch.long) * (W - pw) // 2
        return z1s, y1s, x1s

    def _get_crop_params(self, x):
        if not self.training:
            return self._get_crop_params_test(x)
        device = x.device
        N, C, D, H, W = x.shape
        pd, ph, pw = self._get_crop_size(x)
        if self.max_offset is None:
            z1s = torch.randint(low=0, high=D - pd + 1, size=(N,), device=device)
            y1s = torch.randint(low=0, high=H - ph + 1, size=(N,), device=device)
            x1s = torch.randint(low=0, high=W - pw + 1, size=(N,), device=device)
            if not self.rand_z:
                z1s[...] = (D - pd) // 2
        else:
            offsets = torch.randint(low=-self.max_offset, high=self.max_offset + 1, size=(3, N),
                                    device=device)
            z1s = torch.randint(low=0, high=D - pd + 1, size=(N,), device=device)
            y1s = (H - ph) // 2 + offsets[1]
            x1s = (W - pw) // 2 + offsets[2]
            if not self.rand_z:
                z1s[...] = (D - pd) // 2
        return z1s, y1s, x1s

    @staticmethod
    def apply_crop_to_affine(start, affine):
        x1, y1, z1 = start
        T = torch.eye(4)
        T[0, -1] = x1
        T[1, -1] = y1
        T[2, -1] = z1
        return affine @ T.to(affine)
    
    @staticmethod
    def apply_padding_to_affine(pad, affine):
        T = torch.eye(4)
        T[0, -1] = -pad[0]
        T[1, -1] = -pad[1]
        T[2, -1] = -pad[2]
        return affine @ T.to(affine.device)

    @staticmethod
    def apply_resize_to_affine(affine, crop_shape, resize_shape):
        scale_f = [cdim / rdim for cdim, rdim in zip(crop_shape, resize_shape)]
        T = torch.diag(torch.tensor(scale_f[::-1] + [1])).to(affine)
        return affine @ T

    def forward(self, x, m, affine):
        ori_dim = x.ndim
        if ori_dim == 4:
            print('ori_dim == 4')
            x = x.unsqueeze(1)
            m = m.unsqueeze(1)
        N, C, D, H, W = x.shape

        # 如果尺寸不够，则进行填充
        crop_D, crop_H, crop_W = self.crop_size
        pad_D = max(0, crop_D - D)
        pad_H = max(0, crop_H - H)
        pad_W = max(0, crop_W - W)

        if pad_D > 0 or pad_H > 0 or pad_W > 0:
            x = F.pad(x, (0, pad_W, 0, pad_H, 0, pad_D))
            m = F.pad(m, (0, pad_W, 0, pad_H, 0, pad_D))
            affine = self.apply_padding_to_affine([pad_W, pad_H, pad_D], affine)
            D, H, W = x.shape[-3:]

        if (D, H, W) == tuple(self.crop_size):
            #return x.squeeze(1), m.squeeze(1), affine
            return x, m, affine
        
        crop_shape = self._get_crop_size(x)
        pd, ph, pw = crop_shape
        im_out = x.new_zeros(size=[N, C, pd, ph, pw])
        ms_out = x.new_zeros(size=[N, C, pd, ph, pw])
        z1s, y1s, x1s = self._get_crop_params(x)
        for z1, y1, x1, z2, y2, x2, o, o2, i, i2, aff in zip(z1s, y1s, x1s, z1s + pd, y1s + ph, x1s + pw,
                                                             im_out, ms_out, x, m, affine):
            o[...] = i[:, z1:z2, y1:y2, x1:x2]
            o2[...] = i2[:, z1:z2, y1:y2, x1:x2]
            aff[...] = self.apply_crop_to_affine([x1, y1, z1], aff)
        # apply resize
        resize_shape = self._get_resize_shape(x)
        if resize_shape != crop_shape:
            im_out = F.interpolate(im_out, resize_shape, mode="trilinear", align_corners=False)
            ms_out = F.interpolate(ms_out, resize_shape, mode="nearest", align_corners=False)
            #  update affine
            affine = self.apply_resize_to_affine(affine, crop_shape, resize_shape)

        # if ori_dim == 4:
        #     im_out = im_out.squeeze(1)
        #     ms_out = ms_out.squeeze(1)
        return im_out, ms_out, affine


class RandomJointFlip(nn.Module):
    def __init__(self, p=0.5):
        super(RandomJointFlip, self).__init__()
        self.p = p
        w = [
            [-0.04579317872402952, -0.07575516143899245, -0.061679127638288714, 0.02846023147369374, 0.9765938788126243,
             1.310693443330605],
            [-1.7524712268232336, 1.6816010000146304, -0.10457518199789638, 0.7600968544610555, 0.3062302844162186,
             0.2499470378689073],
            [1.5898357864427095, -1.9611154042037333, 0.22473070108297732, -0.9235759306577322, -1.417853830070638,
             -1.459610511848043]
        ]
        b = [0.0009407975978420954, 0.3678460758384997, -1.1950944844551188]
        self.register_buffer("view_w", torch.tensor(w, dtype=torch.float32)) #torch.float32
        self.register_buffer("view_b", torch.tensor(b, dtype=torch.float32)) #torch.float32
        # ['axial', 'sagittal', 'coronal']

    def _infer_view(self, affine):
        """
        affine: N*4*4
        """
        row_vec = affine[:, :3, 0]
        col_vec = affine[:, :3, 1]
        orient = torch.hstack([row_vec / torch.linalg.norm(row_vec, dim=1, keepdim=True),
                               col_vec / torch.linalg.norm(col_vec, dim=1, keepdim=True)]) #torch.float64
        scores = orient.float() @ self.view_w.T + self.view_b
        view_ind = scores.argmax(dim=1)
        # print(view_ind)
        return view_ind #float()

    @staticmethod
    def _do_horizontal_flip(x, m, affine):
        """
        x: D*H*W
        """
        x = x.flip(2)
        m = m.flip(2)
        T = torch.eye(4)
        T[0, 0] = -1
        T[0, -1] = x.size(2) - 1
        return x, m, affine @ T.to(affine)

    @staticmethod
    def _do_depth_flip(x, m, affine):
        """
        x: D*H*W
        """
        x = x.flip(0)
        m = m.flip(0)
        T = torch.eye(4)
        T[2, 2] = -1
        T[2, -1] = x.size(0) - 1
        return x, m, affine @ T.to(affine)

    def forward(self, x1, m1, affine1, x2, m2, affine2):
        is_sagittal = self._infer_view(affine1) == 1  # do depth flip #判断是否为sag,如果是do depth flip
        # is_coronal = self._infer_view(affine2) == 2  # do horizontal flip
        flip_inds = (torch.rand(size=(x1.size(0),), device=x1.device) < self.p).nonzero(as_tuple=True)[0]
        for idx in flip_inds:
            if is_sagittal[idx]:  # do depth flip for x1, lr flip for x2
                x1[idx], m1[idx], affine1[idx] = self._do_depth_flip(x1[idx], m1[idx], affine1[idx])
                x2[idx], m2[idx], affine2[idx] = self._do_horizontal_flip(x2[idx], m2[idx], affine2[idx])
            else:  # do lr flip for x1, depth flip for x2
                x1[idx], m1[idx], affine1[idx] = self._do_horizontal_flip(x1[idx], m1[idx], affine1[idx])
                x2[idx], m2[idx], affine2[idx] = self._do_depth_flip(x2[idx], m2[idx], affine2[idx])
        return x1, m1, affine1, x2, m2, affine2


class FeatureRegistration(nn.Module):
    def __init__(self, spatial_dim1, spatial_dim2, T1, T2, feat_dim, use_out_embedding=True):
        super(FeatureRegistration, self).__init__()
        self.T1 = T1
        self.T2 = T2
        self.feat_dim = feat_dim
        self.spatial_dim1 = spatial_dim1
        self.spatial_dim2 = spatial_dim2
        self.use_out_embedding = use_out_embedding
        if use_out_embedding:
            self.out_fov_embedding = nn.Embedding(1, feat_dim)

    @staticmethod
    def _transform_affine_to_feature(feats, affine, spatial_dim):   #new_affine
        feat_shape = feats.shape[-3:]
        scale_f = [cdim / rdim for cdim, rdim in zip(spatial_dim, feat_shape)]
        # print(scale_f)
        T = torch.diag(torch.tensor(scale_f[::-1] + [1.])).to(affine)  # no scaling on z
        return affine @ T

    @staticmethod
    def _get_t_natural_to_torch(feats):
        shape = feats.shape[-3:]
        T = torch.eye(4)
        for i, dim in enumerate(shape[::-1]):
            if dim == 1:
                T[i, :] = 0
            else:
                T[i, i] = 2 / (dim - 1)
                T[i, -1] = -1
        return T.to(feats)

    @staticmethod
    def _get_t_torch_to_natural(feats):
        shape = feats.shape[-3:]
        T = torch.eye(4)
        for i, dim in enumerate(shape[::-1]):
            # warning: not appliable for dim == 1
            T[i, i] = (dim - 1) / 2
            T[i, -1] = (dim - 1) / 2
        return T.to(feats)

    def _regist_feats2_to_feats1(self, feats1, affine1, feats2, affine2, align_corners=False):
        affine1, affine2 = affine1.float(), affine2.float()
        inv_affine2 = torch.linalg.inv(affine2)
        theta = self._get_t_natural_to_torch(feats2) @ inv_affine2 @ affine1 @ self._get_t_torch_to_natural(feats1)

        theta = theta[:, :3]
        grid = F.affine_grid(theta, feats1.shape, align_corners=align_corners)
        feats = F.grid_sample(feats2, grid, mode="bilinear", align_corners=align_corners)
        if self.use_out_embedding:
            in_fov = F.grid_sample(torch.ones_like(feats2[:, :1]), grid, mode="bilinear", align_corners=align_corners)
            # in_fov 的值在视野内的区域接近 1，在视野外的区域接近 0
            # print((in_fov == 1).float().mean(dim=(4, 3))[:, 0])
            feats = self.out_fov_embedding.weight[:, :, None, None, None].to(feats1.device) * (1 - in_fov) + in_fov * feats  # todo: use expand dims         
        return feats

    def forward(self, feats1, affine1, feats2, affine2):
        affine1 = self._transform_affine_to_feature(feats1, affine1, (self.T1,) + self.spatial_dim1)
        affine2 = self._transform_affine_to_feature(feats2, affine2, (self.T2,) + self.spatial_dim2)
        feats2_to_1= self._regist_feats2_to_feats1(feats1, affine1, feats2, affine2, True)  # seems to align better
        if False:  # for visualization
            feats1_to_2 = self._regist_feats2_to_feats1(feats2, affine2, feats1, affine1, True)  # seems to align better
            im1 = feats2_to_1[:, 0].clamp(0, 1) * 255
            im2 = feats1_to_2[:, 0].clamp(0, 1) * 255
            im3 = feats1[:, 0].clamp(0, 1) * 255

            def save(idx):
                save_nifti(im1[idx], affine1[idx], "visr1.nii.gz")
                save_nifti(im2[idx], affine2[idx], "visr2.nii.gz")
                save_nifti(im3[idx], affine2[idx], "vis_1rs.nii.gz")
            save(0)
        return feats2_to_1,affine1,affine2




class DualViewSegNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=1,training=True):
        super(DualViewSegNet, self).__init__()

        # self.net = UNet(in_channel=1, out_channel=1,training=training)         #Register_VAE(n_channels, gf_dim,self.net_depth,self.feat_regist)
        self.net = init_dynunet(in_channels=1, out_channels=2)
        self.out = nn.Conv3d(2, out_channel, 1, 1)

    def norm(self,x):
        lower_percentile = torch.quantile(x, self.lower / 100.0)
        upper_percentile = torch.quantile(x, self.upper / 100.0)
        x = (x - lower_percentile) / (upper_percentile - lower_percentile) * (self.b_max - self.b_min) + self.b_min
        return x

    def _decode_view(self, x):
        """
        x: B*(D*H*W)
        """
        shape = (16, 128, 128)
        data_len = math.prod(shape)
        N = x.size(0)
        image = x[:, :data_len].reshape(N, *shape)
        mask = x[:, data_len:data_len*2].reshape(N, *shape)
        affine = x[:, -16:].reshape(N, 4, 4)
        return image, mask, affine

    @staticmethod
    def ignore_boundary(mask):
        boundary_fg = torch.zeros_like(mask, dtype=torch.bool)
        boundary_fg[ :, 2:-2, 16:-16, 16:-16] = 1
        boundary_fg = (~boundary_fg) & (mask == 1)
        # boundary_fg = ~boundary_fg
        mask[boundary_fg] = 255
        return mask

    def forward(self, x, device, save_resize_vis=None,save_path=None):
        tse_series = DummySeries2(x['tse_dic'],device)
        dess_series = DummySeries2(x['dess_dic'],device)

        x = tse_series.images #[1 1 256 256 32]
        mask_f = torch.zeros_like(x)
        aff1 = tse_series.affine
        gt = dess_series.images
        mask_gt = torch.ones_like(gt)
        mask_f [mask_gt==1] = 1 
        aff2 = dess_series.affine

        out = self.net(x)
        out = self.out(out)

        dic = { 'reconstruction':out * mask_f,
               'aff':aff1,
               'input':x * mask_f,
               'gt':gt * mask_f,
            }

        return dic 
    
class DummySeries2(object):
    def __init__(self, dic, device):
        for key in dic:
            setattr(self, key, dic[key].float().to(device))

    def clone(self):
        # 创建当前对象的副本
        return DummySeries2(self.__dict__,self.images.device)

@register_model
def mri_dual_view_seg_net(**kwargs):
    model = DualViewSegNet(num_classes=kwargs['num_classes'],
                           crop_size=(16, 128, 128), resize_dim=(16, 128, 128))
    return model


@register_model
def mri_dual_view_seg_net_noaux(**kwargs):
    model = DualViewSegNet(num_classes=kwargs['num_classes'],
                           crop_size=(16, 128, 128), resize_dim=(16, 128, 128),
                           aux_seg=False)
    return model


@register_model
def mri_dual_view_seg_net_noaux_ft(**kwargs):
    model = DualViewSegNet(num_classes=kwargs['num_classes'],
                           crop_size=(16, 128, 128), resize_dim=(16, 128, 128),
                           aux_seg=False)
    ckpt_path = "/mnt/users/workspace/classification/timm3d/output/train/mri_dual_view_seg_net_noaux/model_best.pth.tar"
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


@register_model
def mri_dual_view_seg_net_noaux_ft_cteval(**kwargs):
    model = DualViewSegNet(num_classes=kwargs['num_classes'],
                           crop_size=(16, 128, 128), resize_dim=(16, 128, 128),
                           aux_seg=False, ct_eval=True)
    ckpt_path = "/mnt/users/workspace/classification/timm3d/output/train/mri_dual_view_seg_net_noaux/model_best.pth.tar"
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


@register_model
def mri_single_view_seg_net(**kwargs):
    model = DualViewSegNet(num_classes=kwargs['num_classes'],
                           crop_size=(16, 128, 128), resize_dim=(16, 128, 128),
                           aux_seg=False, single_view=True)
    return model


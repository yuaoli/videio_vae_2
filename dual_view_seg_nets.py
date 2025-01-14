import logging
from typing import Sequence, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transforms_cuda_det import build_cuda_transform
from .nnunet import UNet3D, UpsampleBlock, ConvBlock
from timm.utils.metric_seg import compute_stat_from_pred_gt
# from .3d-knee-vae.experiments.main_experiment_256 import Encoder,Decoder

try:
    from .registry import register_model
except:
    register_model = lambda x: x

logger = logging.getLogger('train')


def save_nifti(pixel_data, affine, fpath):
    import numpy as np
    import nibabel as nib
    affine = affine.cpu().numpy()
    pixel_data = pixel_data.type(torch.int16).cpu().numpy()
    pixel_data = np.transpose(pixel_data, (2, 1, 0))  # WHD

    nifti_img = nib.Nifti1Image(pixel_data, affine)
    nib.save(nifti_img, fpath)


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
        if (D, H, W) == tuple(self.crop_size):
            #return x, m, affine
            return x.squeeze(1), m.squeeze(1), affine
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

        if ori_dim == 4:
            im_out = im_out.squeeze(1)
            ms_out = ms_out.squeeze(1)
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
        self.register_buffer("view_w", torch.tensor(w, dtype=torch.float32))
        self.register_buffer("view_b", torch.tensor(b, dtype=torch.float32))
        # ['axial', 'sagittal', 'coronal']

    def _infer_view(self, affine):
        """
        affine: N*4*4
        """
        row_vec = affine[:, :3, 0]
        col_vec = affine[:, :3, 1]
        orient = torch.hstack([row_vec / torch.linalg.norm(row_vec, dim=1, keepdim=True),
                               col_vec / torch.linalg.norm(col_vec, dim=1, keepdim=True)])
        scores = orient @ self.view_w.T + self.view_b
        view_ind = scores.argmax(dim=1)
        # print(view_ind)
        return view_ind

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
        is_sagittal = self._infer_view(affine1) == 1  # do depth flip
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
            # print((in_fov == 1).float().mean(dim=(4, 3))[:, 0])
            feats = self.out_fov_embedding.weight[:, :, None, None, None] * (1 - in_fov) + in_fov * feats  # todo: use expand dims
        return feats

    def forward(self, feats1, affine1, feats2, affine2):
        affine1 = self._transform_affine_to_feature(feats1, affine1, (self.T1,) + self.spatial_dim1)
        affine2 = self._transform_affine_to_feature(feats2, affine2, (self.T2,) + self.spatial_dim2)
        feats2_to_1 = self._regist_feats2_to_feats1(feats1, affine1, feats2, affine2, True)  # seems to align better
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
        return feats2_to_1


class DualViewUNet():#UNet3D
    def __init__(self, num_classes, aux_seg=True, single_view=False):
        super(DualViewUNet, self).__init__(
            kernels=[[1, 3, 3], [1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            strides=[[1, 1, 1], [1, 2, 2], [1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]],
            width_factor=0.25, n_class=num_classes)  # shared encoder
        self.single_view = single_view
        # build aux decoder
        self.aux_seg = aux_seg
        # if aux_seg:
        #     self.upsamples_aux = self.get_module_list(
        #         conv_block=UpsampleBlock,
        #         in_channels=self.filters[1:][::-1],
        #         out_channels=self.filters[:-1][::-1],
        #         kernels=self.kernels[1:][::-1],  # todo: change it to 3*3*3 conv??
        #         strides=self.strides[1:][::-1],
        #     )
            # self.output_block_aux = self.get_output_block(decoder_level=0)
        # build cross view feature fusion module
        if not self.single_view:
            n_layers = len(self.filters[:-1])
            self.feat_regist = nn.ModuleList(
                FeatureRegistration((128, 128), (128, 128), 16, 16, feat_dim) for feat_dim in self.filters[:-1]
            )
            # self.fusion_convs = self.get_module_list(
            #     conv_block=ConvBlock,
            #     in_channels=[_ * 2 for _ in self.filters[:-1]],
            #     out_channels=self.filters[:-1],
            #     kernels=[(3, 3, 3), ] * n_layers,
            #     strides=[(1, 1, 1), ] * n_layers,
            # )
        self.apply(self.initialize_weights)

    def _forward_fusion(self, enc_outs1, aff1, enc_outs2, aff2):
        fusion_outputs = []
        for align_layer, fusion_layer, enc1, enc2 in zip(self.feat_regist, self.fusion_convs,
                                                         enc_outs1, enc_outs2):
            enc2_align = align_layer(enc1, aff1, enc2, aff2)
            out = fusion_layer(torch.cat([enc1, enc2_align], dim=1))
            fusion_outputs.append(out)
        return fusion_outputs

    def _forward_decoders_aux(self, out, encoder_outputs):
        decoder_outputs = []
        for upsample, skip in zip(self.upsamples_aux, reversed(encoder_outputs)):
            out = upsample(out, skip)
            decoder_outputs.append(out)
        return out, decoder_outputs

    def forward(self, im1, aff1, im2, aff2):
        out1, enc_outs1 = self._forward_encoders(im1)
        if self.single_view:
            out1, _ = self._forward_decoders(out1, enc_outs1)
            out1 = self.output_block(out1)
            return out1, None
        out2, enc_outs2 = self._forward_encoders(im2)
        enc_outs1 = self._forward_fusion(enc_outs1, aff1, enc_outs2, aff2)
        out1, _ = self._forward_decoders(out1, enc_outs1)
        out1 = self.output_block(out1)
        if self.aux_seg:
            out2, _ = self._forward_decoders_aux(out2, enc_outs2)
            out2 = self.output_block_aux(out2)
        else:
            out2 = None
        return out1, out2


class DualViewSegNet(nn.Module):
    def __init__(self, num_classes, crop_size=(16, 128, 128), resize_dim=(16, 128, 128), aux_seg=True,
                 single_view=False, ct_eval=False, lower = 0.0, upper = 99.75, b_min = 0., b_max=1.,):
        super(DualViewSegNet, self).__init__()
        self.resize_dim = resize_dim
        self.aux_seg = aux_seg
        self.ct_eval = ct_eval
        self.num_classes = num_classes
        self.flip_fn = RandomJointFlip(p=0.5)
        self.crop_resize_fn1 = RandomCropResizeWithAffine(crop_shape=crop_size,
                                                          resize_shape=resize_dim,
                                                          rand_z=False)
        self.crop_resize_fn2 = RandomCropResizeWithAffine(crop_shape=crop_size,
                                                          resize_shape=resize_dim,
                                                          rand_z=False,
                                                          max_offset=0)
        self.inten_trans = build_cuda_transform()
        # self.feat_regist = FeatureRegistration((128, 128), (128, 128), 16, 16, 1)
        self.net = DualViewUNet(num_classes, aux_seg=aux_seg, single_view=single_view)

        self.lower = lower
        self.upper = upper
        self.b_min = b_min
        self.b_max = b_max

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
        boundary_fg[:, 2:-2, 16:-16, 16:-16] = 1
        boundary_fg = (~boundary_fg) & (mask == 1)
        # boundary_fg = ~boundary_fg
        mask[boundary_fg] = 255
        return mask

    @torch.no_grad()
    def _decode_input(self, x):
        im1, ms1, aff1 = x['SAG'],x['sag_mask'],x['sag_affine']
        im2, ms2, aff2 = x['COR'],x['cor_mask'],x['cor_affine']  
        if self.training:
            im1, ms1, aff1, im2, ms2, aff2 = self.flip_fn(im1, ms1, aff1, im2, ms2, aff2)
        # crop
        im1, ms1, aff1 = self.crop_resize_fn1(im1, ms1, aff1)
        im2, ms2, aff2 = self.crop_resize_fn2(im2, ms2, aff2)
        # print('im1_ori_max:',im1.max(),' im1_ori_min:',im1.min())
        # print('im2_ori_max:',im2.max(),' im2_ori_min:',im2.min())
        im1 = self.norm(im1)
        im2 = self.norm(im2)
        # print('im1_norm_max:',im1.max(),' im1_norm_min:',im1.min())
        # print('im2_norm_max:',im2.max(),' im2_norm_min:',im2.min())

        npos = (ms1 == 1).flatten(1).any(dim=1).sum()
        print(f"pos={npos}, neg={len(ms1) - npos}")

        if self.training:  # random ignore BG
            # todo: warning use with caution; hard coded
            import random
            if random.random() > 0.5:  # train on all voxels
                ms1[ms1 == 253] = 1
                ms1[ms1 == 254] = 0
                ms2[ms2 == 254] = 0
            else:  # only train on stage1 predicted region
                ms1[ms1 == 253] = 255
                ms1[ms1 == 254] = 255
                ms2[ms2 == 254] = 255
        else:  # evaluation mode
            ms1[ms1 == 253] = 255
            ms1[ms1 == 254] = 255
            ms2[ms2 == 254] = 255

        ms1 = self.ignore_boundary(ms1)
        ms2 = self.ignore_boundary(ms2)
        # todo: warning: padding of mask should be 254???

        if False:  # visualization block
            def save(idx):
                save_nifti(im1[idx].clamp(0, 1) * 255, aff1[idx], "vis1im.nii.gz")
                save_nifti(ms1[idx].clamp(0, 1) * 255, aff1[idx], "vis1ms.nii.gz")
                save_nifti(im2[idx].clamp(0, 1) * 255, aff2[idx], "vis2im.nii.gz")
                save_nifti(ms2[idx].clamp(0, 1) * 255, aff2[idx], "vis2ms.nii.gz")
            save(0)
        return im1.unsqueeze(1), ms1, aff1, im2.unsqueeze(1), ms2, aff2

    def forward(self, x):
        im1, ms1, aff1, im2, ms2, aff2 = self._decode_input(x)
        pred1, pred2 = self.net(im1, aff1, im2, aff2)
        if self.training and self.aux_seg:
            logits = torch.cat([pred1, pred2], dim=0)
            mask = torch.cat([ms1, ms2], dim=0)
        else:
            logits, mask = pred1, ms1
        if (not self.training) and self.ct_eval:
            pred = torch.softmax(logits, dim=1).argmax(dim=1)
            return compute_stat_from_pred_gt(pred, mask, self.num_classes, ignore_index=255, reduction="node")
        return logits, mask


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


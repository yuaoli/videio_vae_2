__author__ = 'aoao'

import os
import numpy as np
import io
import glob
from PIL import Image
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
# from torchvision import datasets, transforms
# from torchvision.utils import save_image
from torch.utils import tensorboard
import torchvision
import functools
import matplotlib
import collections
from custom import CustomTrain,CustomTest
import math
import time
from datetime import datetime

from model.dual_view_vae import DualViewSegNet

from model.register import image_register
import SimpleITK as sitk
import nibabel as nib
from tqdm import tqdm
import random

import sys
sys.path.append('/mnt/users/3d_resiger_vae/taming-transformers')
from taming.modules.losses.contperceptual import LPIPSWithDiscriminator
# from taming.modules.discriminator.model import NLayerDiscriminator3D
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

# 用于多GPU训练的必要导入
import torch.distributed as dist
import torch.multiprocessing as mp


matplotlib.use('Agg')


np.seterr(all='raise')
np.random.seed(2019)
torch.manual_seed(2019)

parser = argparse.ArgumentParser()

parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--save_epoch_interval', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--latent-dimension', type=int, default=256, metavar='N',
                    help=' ')
parser.add_argument('--n-channels', type=int, default=1, metavar='N', #n_channels
                    help=' ')
parser.add_argument('--img-size', type=int, default=256, metavar='N',
                    help=' ')
parser.add_argument('--learning-rate', type=float, default=1e-4, metavar='N',
                    help=' ')
parser.add_argument('--architecture', type=str, default='old', metavar='N',
                    help=' ')
parser.add_argument('--reconstruction-data-loss-weight', type=float, default=1,
                    help=' ')
parser.add_argument('--kl-latent-loss-weight', type=float, default=0.00001,
                    help=' ')
parser.add_argument("--train_data_dir", type = str, 
                    default ="your_train_data_path",
                    help = "the data directory")
parser.add_argument("--test_data_dir", type = str, 
                    default ="your_test_data_path",
                    help = "the data directory")

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def remove_module_prefix(state_dict):
    """
    去掉 state_dict 中的 'module.' 前缀
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 去掉 'module.' 前缀
        else:
            new_state_dict[k] = v
    return new_state_dict

def save_nifti(pixel_data, affine, fpath):
    pixel_data = np.transpose(pixel_data, (2, 1, 0))  # WHD
    nifti_img = nib.Nifti1Image(pixel_data, affine)
    nib.save(nifti_img, fpath)

def save_vis(img,aff=None,path=None):
    if aff is not None:
        img_arry = img[0,0,...].detach().cpu().numpy()
        aff = aff.detach().cpu().numpy()
        # aff[0] *= -1
        # aff[1] *= -1  # to affine itk
        path = path
        save_nifti(img_arry,aff,path)
    else:
        img_arry = img[0,0,...].detach().cpu().numpy()
        path = path
        img = sitk.GetImageFromArray(img_arry)
        sitk.WriteImage(img,path)


def init_process(rank, world_size, fn, args):

    os.environ['MASTER_ADDR'] = 'localhost'  # 或者你的主节点IP
    os.environ['MASTER_PORT'] = '12355'       # 选择一个未被占用的端口

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    fn(rank, world_size, *args)


def make_grid_3d(tensor, nrow=8, padding=2, normalize=False, value_range=None, scale_each=False, pad_value=0):
    """
    Make a grid of 3D images.
    
    Args:
        tensor (Tensor): 5D mini-batch Tensor of shape (B x C x H x W x D).
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default is 8.
        padding (int, optional): Amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by ``value_range``. Default is False.
        value_range (tuple, optional): Tuple (min, max) where min and max are numbers.
            These numbers are used to normalize the image. By default, min and max are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of images separately
            rather than (min, max) over all images. Default is False.
        pad_value (float, optional): Value for the padded pixels. Default is 0.
    
    Returns:
        grid (Tensor): The tensor containing grid of images.
    """
    if not torch.is_tensor(tensor):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    if tensor.dim() == 4:
        tensor = tensor.unsqueeze(0)

    if tensor.dim() != 5:
        raise ValueError('Expected 5D tensor, got {}D tensor'.format(tensor.dim()))

    if normalize:
        tensor = tensor.clone()  # Avoid modifying tensor in-place
        if value_range is not None and not isinstance(value_range, tuple):
            raise TypeError('value_range has to be a tuple (min, max) if specified. min and max are numbers')

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each:
            for t in tensor:  # Loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    # Make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))

    height, width, depth = tensor.size(2), tensor.size(3), tensor.size(4)
    grid = tensor.new_full((tensor.size(1), height * ymaps + padding, width * xmaps + padding, depth), pad_value)

    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * (height + padding), height).narrow(
                2, x * (width + padding), width
            ).copy_(tensor[k])
            k += 1
    return grid



class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


reconstruction_loss = nn.MSELoss(reduction='mean')#sum


def KLLoss(z_mean, z_log_sigma):
    # 计算 KL 散度
    loss = -0.5 * torch.mean(
        1 + 2 * z_log_sigma - torch.pow(z_mean, 2) - torch.exp(2 * z_log_sigma)
    )
    return loss  # 这是一个标量值




def add_grid(writer, images, name, step,
             batch_size=32, n_channels=1, img_size=128):
    _,_,h,w,d = images.shape
    grid = make_grid_3d(
        images.view(batch_size, n_channels, img_size, img_size, d),
        nrow=1,
        normalize=True,
        value_range=(-1, 1))
    
    d = grid.shape[-1]
    mid_d = d//2

    writer.add_image(name, grid[:, :, :, mid_d], step)


def add_detailed_summaries(writer, decoder, phase, data, reconstruction, latent,
                           step,
                           batch_size=32,
                           n_channels=1,
                           img_size=128):
    add_grid(writer, data, 'Data/{}'.format(phase), step,
             batch_size=batch_size, n_channels=n_channels, img_size=img_size)
    add_grid(writer, reconstruction, 'Reconstruction/{}'.format(phase), step,
             batch_size=batch_size, n_channels=n_channels, img_size=img_size)

    zs = torch.randn_like(latent)
    samples = decoder(zs)
    add_grid(writer, samples, 'Samples/{}'.format(phase), step,
             batch_size=batch_size, n_channels=n_channels, img_size=img_size)


def train(model, optimizer_vae,optimizer_disc, epoch, step, train_loader, Discriminator, writer,
          reconstruction_data_loss_weight=1.0,
          kl_latent_loss_weight=1.0,
          batch_size=32,
          log_interval=1000,
          n_channels=1,
          img_size=128,
          rank = 0,
          results_dir = None,
          ddp = False):
    
    rank = torch.device(f'cuda:{rank}')

    model.train()
    Discriminator.train()
    train_vae_loss_sum = 0
    train_disc_loss_sum = 0 


    def get_last_layer():
        if ddp:
            return model.module.net.decoder.out.weight
        else:
            try:
                return model.out.weight
            except:
                return model.net.out.weight

    for idx, batch in enumerate(tqdm(train_loader, desc="Training", unit="batch")):

        if idx % log_interval == 0:
            save_path = os.path.join(results_dir,'train',str(epoch)+'_epoch',str(idx)+'_step')
            os.makedirs(save_path,exist_ok=True)
            save_resize_vis = True
        else:
            save_resize_vis = False

        optimizer_vae.zero_grad()
        optimizer_disc.zero_grad()
        results = model(batch,rank,save_resize_vis,save_path) #[1 16 32 4 32] #x1_mu, x1_std,x1_reconstruction

        for dic in (results,):

            train_vae_loss = 0
            train_disc_loss = 0
            vae_losses_2d = []
            disc_losses = []

            # img = dic['input']
            reconstruction = dic['reconstruction']
            img = dic['gt']

            rec_loss = reconstruction_loss(reconstruction,img)
            vae_loss_3d = rec_loss * args.reconstruction_data_loss_weight

            
            B,C,D,H,W = img.shape

            # Selects one random 2D image from each 3D Image
            frame_idx_T = torch.randperm(D, device='cuda')[:8] #torch.randperm(H, device='cuda')[:8]
            frames_T = []
            frames_recon_T = []
            for i in frame_idx_T:
               img_slice = img[:, :, i, :, :] 
               reconstruction_slice = reconstruction[:, :, i, :, :]
               frames_T.append(img_slice)
               frames_recon_T.append(reconstruction_slice)

            frame_idx_H = torch.randperm(H, device='cuda')[:8]
            frames_H = []
            frames_recon_H = []
            for i in frame_idx_H:
               img_slice = img[:, :, :, i, :] 
               reconstruction_slice = reconstruction[:, :, :, i, :]
               frames_H.append(img_slice)
               frames_recon_H.append(reconstruction_slice)


            frame_idx_W = torch.randperm(W, device='cuda')[:8]
            frames_W = []
            frames_recon_W = []
            for i in frame_idx_W:
               img_slice = img[:, :, :, :, i] 
               reconstruction_slice = reconstruction[:, :, :, :, i]
               frames_W.append(img_slice)
               frames_recon_W.append(reconstruction_slice)


            for (img_2d_list,reconstruction_2d_list) in [(frames_T,frames_recon_T),(frames_H,frames_recon_H),(frames_W,frames_recon_W)]:
                
                for i in range(len(img_2d_list)):
                    img_2d = img_2d_list[i]
                    reconstruction_2d = reconstruction_2d_list[i]
                    try:
                        loss_vae_2d, log_vae_2d = Discriminator(img_2d,
                                                            reconstruction_2d,
                                                            posteriors = None, 
                                                            optimizer_idx=0, 
                                                            global_step = step,
                                                            last_layer=get_last_layer(),
                                                            )
                        loss_disc, log_disc = Discriminator(img_2d.detach(),
                                                            reconstruction_2d.detach(),
                                                            posteriors = None,
                                                            optimizer_idx=1,
                                                            global_step = step,
                                                            last_layer=get_last_layer(),)
                    
                        vae_losses_2d.append(loss_vae_2d)
                        disc_losses.append(loss_disc)

                    except RuntimeError as e:
                        print(f"Error in batch {i}, sub-batch {B}: {e}")
                        continue


            train_vae_loss = torch.mean(torch.stack(vae_losses_2d)) # 24 = 3*8
            train_vae_loss += vae_loss_3d
            train_disc_loss = torch.mean(torch.stack(disc_losses))

            train_vae_loss_sum += train_vae_loss
            train_disc_loss_sum += train_disc_loss

            train_vae_loss.backward()
            optimizer_vae.step()

            del vae_losses_2d, disc_losses
            # torch.cuda.empty_cache()

            train_disc_loss.backward()
            optimizer_disc.step() 

            del loss_vae_2d, loss_disc, img_2d, reconstruction_2d
            # torch.cuda.empty_cache()    

        step += 1

        writer.add_scalar('Loss/train_vae_sum', train_vae_loss.item(), step)

        writer.add_scalar('Loss/train_3d_vae', vae_loss_3d.item(), step)
        writer.add_scalar('Loss/train_3d_reconstruction_data_loss', rec_loss.item(), step)

        writer.add_scalar('Loss/train_2d_reconstruction_data_loss', log_vae_2d['train/rec_loss'].item(), step)
        writer.add_scalar('Loss/train_nll_loss', log_vae_2d['train/nll_loss'].item(), step)
        writer.add_scalar('Loss/train_g_loss', log_vae_2d['train/g_loss'].item(), step)

        writer.add_scalar('Loss/train_disc_sum', train_disc_loss.item(), step)
        writer.add_scalar('Loss/train_real', log_disc['train/logits_real'].item(), step)
        writer.add_scalar('Loss/train_fake', log_disc['train/logits_fake'].item(), step)

        del train_vae_loss,vae_loss_3d,rec_loss,log_vae_2d,train_disc_loss,log_disc

        if idx % log_interval == 0:
            
            sag_reconstruction = results['reconstruction']
            sag_img =results['input']
            dess_img =results['gt']      
            save_vis(sag_reconstruction , path = os.path.join(save_path,'sag_reconstruction_sitk'+'.nii'))
            save_vis(sag_img,  path = os.path.join(save_path,'ori_sag_sitk'+'.nii'))
            save_vis(dess_img , path = os.path.join(save_path,'dess_img_sitk'+'.nii'))
            del sag_img, sag_reconstruction,dess_img

    print('====> Epoch: {} Average vae loss: {:.4f}'.format(epoch, train_vae_loss_sum / len(train_loader.dataset)))
    print('====> Epoch: {} Average disc loss: {:.4f}'.format(epoch, train_disc_loss_sum / len(train_loader.dataset)))
    
    return model, Discriminator, optimizer_vae, optimizer_disc, step



def test(model, epoch, step, test_loader, Discriminator, writer,
         reconstruction_data_loss_weight=1.0,
         kl_latent_loss_weight=1.0,
         batch_size=32,
         n_channels=1,
         img_size=128,
         rank = '0',
         results_dir = None,
         ddp = False
         ):
    
    rank = torch.device(f'cuda:{rank}')
    n_test = 20


    def get_last_layer():
        if ddp:
            return model.module.net.decoder.out.weight
        else:
            try:
                return model.out.weight
            except:
                return model.net.out.weight
        

    model.eval()
    Discriminator.eval()

    test_vae_loss_sum = 0
    test_disc_loss_sum = 0         

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="Testing", unit="batch")): #tqdm(train_loader, desc="Training", unit="batch")

            save_path = os.path.join(results_dir,'test','epoch_' + str(epoch),'img_'+str(i))
            os.makedirs(save_path,exist_ok=True)

            save_resize_vis = True

            test_vae_loss = 0
            test_disc_loss= 0   

            results = model(batch,rank,save_resize_vis,save_path) #[1 16 32 4 32] #x1_mu, x1_std,x1_reconstruction

            for dic in (results,):
                img = dic['input']
                reconstruction = dic['reconstruction']
                img = dic['gt']

                rec_loss = reconstruction_loss(reconstruction,img)
                vae_loss_3d = rec_loss * args.reconstruction_data_loss_weight
                


                B,C,D,H,W = img.shape

                # Selects one random 2D image from each 3D Image
                frame_idx_T = torch.randperm(D, device='cuda')[:16]
                frames_T = []
                frames_recon_T = []
                for i in frame_idx_T:
                    img_slice = img[:, :, i, :, :] 
                    reconstruction_slice = reconstruction[:, :, i, :, :]
                    frames_T.append(img_slice)
                    frames_recon_T.append(reconstruction_slice)

                frame_idx_H = torch.randperm(H, device='cuda')[:16]
                frames_H = []
                frames_recon_H = []
                for i in frame_idx_H:
                    img_slice = img[:, :, :, i, :] 
                    reconstruction_slice = reconstruction[:, :, :, i, :]
                    frames_H.append(img_slice)
                    frames_recon_H.append(reconstruction_slice)


                frame_idx_W = torch.randperm(W, device='cuda')[:16]
                frames_W = []
                frames_recon_W = []
                for i in frame_idx_W:
                    img_slice = img[:, :, :, :, i] 
                    reconstruction_slice = reconstruction[:, :, :, :, i]
                    frames_W.append(img_slice)
                    frames_recon_W.append(reconstruction_slice)

                vae_losses_2d = []
                disc_losses = []
                for (img_2d_list,reconstruction_2d_list) in [(frames_T,frames_recon_T),(frames_H,frames_recon_H),(frames_W,frames_recon_W)]:
                    
                    for i in range(len(img_2d_list)):
                        img_2d = img_2d_list[i]
                        reconstruction_2d = reconstruction_2d_list[i]

                    try:
                        loss_vae_2d, log_vae_2d = Discriminator(img_2d,
                                                            reconstruction_2d,
                                                            posteriors = None, 
                                                            optimizer_idx=0, 
                                                            global_step = step,
                                                            last_layer=get_last_layer(),
                                                            split="test"
                                                            )
                        loss_disc, log_disc = Discriminator(img_2d.detach(),
                                                            reconstruction_2d.detach(),
                                                            posteriors = None,
                                                            optimizer_idx=1,
                                                            global_step = step,
                                                            last_layer=get_last_layer(),
                                                            split="test")
                    
                        vae_losses_2d.append(loss_vae_2d)
                        disc_losses.append(loss_disc)

                    except RuntimeError as e:
                        print(f"Error in batch {i}, sub-batch {B}: {e}")
                        continue

            test_vae_loss = torch.mean(torch.stack(vae_losses_2d))
            test_vae_loss += vae_loss_3d
            test_vae_loss_sum += test_vae_loss
            test_disc_loss = torch.mean(torch.stack(disc_losses))
            test_disc_loss_sum += test_disc_loss

            torch.cuda.empty_cache()    


            writer.add_scalar('Loss/test_vae_sum', test_vae_loss.item(), step)

            writer.add_scalar('Loss/test_3d_vae', vae_loss_3d.item(), step)
            writer.add_scalar('Loss/test_3d_reconstruction_data_loss', rec_loss.item(), step)

            writer.add_scalar('Loss/test_2d_reconstruction_data_loss', log_vae_2d['test/rec_loss'].item(), step)
            writer.add_scalar('Loss/test_nll_loss', log_vae_2d['test/nll_loss'].item(), step)
            writer.add_scalar('Loss/test_g_loss', log_vae_2d['test/g_loss'].item(), step)

            writer.add_scalar('Loss/test_disc_sum', test_disc_loss.item(), step)
            writer.add_scalar('Loss/test_real', log_disc['test/logits_real'].item(), step)
            writer.add_scalar('Loss/test_fake', log_disc['test/logits_fake'].item(), step)
            
            del test_vae_loss,vae_loss_3d,rec_loss,log_vae_2d,test_disc_loss,log_disc

            # sag_reconstruction = sag_dic['reconstruction']
            # sag_img =sag_dic['input']
            # cor_reconstruction = cor_dic['reconstruction']
            # cor_img = cor_dic['input']            
            # save_vis(sag_reconstruction , path = os.path.join(save_path,'sag_reconstruction_sitk'+'.nii'))
            # save_vis(sag_img,  path = os.path.join(save_path,'ori_sag_sitk'+'.nii'))
            # save_vis(cor_reconstruction , path = os.path.join(save_path,'cor_reconstruction_sitk'+'.nii'))
            # save_vis(cor_img,  path = os.path.join(save_path,'ori_cor_sitk'+'.nii'))
            # del sag_img, sag_reconstruction, cor_img, cor_reconstruction

    writer.flush()
    print('====> Test vae set loss: {:.4f}'.format(test_vae_loss_sum / n_test))
    print('====> Test disc set loss: {:.4f}'.format(test_disc_loss_sum / n_test))
    return model, step


def make_train_loader(train_data_dir,device):
    train_data = CustomTrain(train_data_dir,device)
    return train_data

def make_test_loader(test_data_dir,device):
    test_data = CustomTrain(test_data_dir,device)
    return test_data



def main(rank, world_size, args):

    torch.manual_seed(args.seed)

    model = DualViewSegNet(in_channel=1, out_channel=1,training=(args.phase == 'train'))
    Discriminator = LPIPSWithDiscriminator(disc_start = 2001,
                                            kl_weight = 1.0e-06,
                                            disc_in_channels=3,
                                            disc_weight = 0.5,
                                            perceptual_weight=1.0)

    # model = DualViewSegNet(args.n_channels, gf_dim=2, lat_ch1 = 4,lat_ch2 = 4)

    if args.distributed:
        Discriminator = Discriminator.to(rank)
        Discriminator = nn.parallel.DistributedDataParallel(Discriminator, device_ids=[rank], find_unused_parameters=True)
        model = model.to(rank)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    else:
        model = model.to(rank)
        Discriminator = Discriminator.to(rank)

    optimizer_vae = optim.Adam(list(model.parameters()), lr=args.learning_rate)
    optimizer_disc = optim.Adam(list(Discriminator.parameters()), lr=args.learning_rate)
    print('Cuda is {}available'.format('' if torch.cuda.is_available() else 'not '))
    print(model)
    print(Discriminator)


    if args.resume_path is None:
        timestamp = time.time()
        EXPERIMENT = '3d_vae_sag_pd' + datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d-%H-%M-%S')
        run_name = 'vol_256_lr_{}' \
                '_kl_{}_' \
                '_bsize_{}' \
                ''.format(
            args.learning_rate,
            args.kl_latent_loss_weight,
            args.batch_size)

        load_from_ckpt_dir = os.path.join('experiments', EXPERIMENT, 'gen', run_name, 'checkpoints')
        os.makedirs(load_from_ckpt_dir, exist_ok=True)

        print('testing complete')
        print("=> no checkpoint found")
        start_epoch = 1
        step = 0
    else:
        resume_path = args.resume_path #os.path.join(load_from_ckpt_dir, sorted(os.listdir(load_from_ckpt_dir))[-1])
        if not os.path.exists(resume_path):
            print("=> no checkpoint found")
            start_epoch = 1
            step = 0
        else:
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location=torch.device(f'cuda:{rank}'))
            start_epoch = checkpoint['epoch']
            try:
                model.load_state_dict(remove_module_prefix(checkpoint['model']))
                # Discriminator.load_state_dict(remove_module_prefix(checkpoint['Discriminator']))
            except:
                model.load_state_dict(checkpoint['model'])
                # Discriminator.load_state_dict(checkpoint['Discriminator'])
            optimizer_vae.load_state_dict(checkpoint['optimizer_vae'])
            optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])
            step = checkpoint['step']
            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
            
        EXPERIMENT = resume_path.split('/')[5]
        run_name = resume_path.split('/')[7]

    ckpt_dir = os.path.join('experiments', EXPERIMENT, 'gen', run_name, 'checkpoints')
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    tb_dir = os.path.join('experiments', EXPERIMENT, 'gen', 'tensorboard', run_name)
    if not os.path.isdir(tb_dir):
        os.makedirs(tb_dir, exist_ok=True)

    frame_dir = os.path.join('experiments', EXPERIMENT, 'gen', run_name, 'frames')
    if not os.path.isdir(frame_dir):
        os.makedirs(frame_dir, exist_ok=True)

    results_dir = os.path.join('experiments', EXPERIMENT, 'gen', run_name, 'results')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    writer = tensorboard.SummaryWriter(log_dir=tb_dir)
    
    if args.phase == 'train':
        train_dataset = make_train_loader(args.train_data_dir, device = rank)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                                    num_workers=args.num_workers,
                                                    # shuffle=True,
                                                    drop_last=True,
                                                    # multiprocessing_context='spawn',
                                                    )
        
        for epoch in range(start_epoch, args.epochs + 1):
            model, Discriminator, optimizer_vae, optimizer_disc, step = train(model, optimizer_vae,optimizer_disc, epoch, step, train_loader,
                                                                Discriminator,
                                                                writer,
                                                                reconstruction_data_loss_weight=args.reconstruction_data_loss_weight,
                                                                kl_latent_loss_weight=args.kl_latent_loss_weight,
                                                                batch_size=args.batch_size,
                                                                log_interval=args.log_interval,#args.log_interval
                                                                n_channels=args.n_channels,
                                                                img_size=args.img_size,
                                                                rank = rank,
                                                                results_dir = results_dir,
                                                                ddp = True if args.distributed else False)
            train_dataset.clear_cache()

            if epoch % args.save_epoch_interval == 0:
                save_path = os.path.join(ckpt_dir, 'model_{0:08d}.pth.tar'.format(epoch))
                torch.save({
                    'model': model.state_dict(),
                    'optimizer_vae': optimizer_vae.state_dict(),
                    'optimizer_disc': optimizer_disc.state_dict(),
                    'Discriminator': Discriminator.state_dict(),
                    'epoch': epoch,
                    'step': step
                },
                    save_path)
                print('Saved model')

                if epoch % (args.save_epoch_interval+4) == 0 and epoch!=0:

                    args.phase = 'test'        

                    test_dataset = make_test_loader(args.test_data_dir, device = rank)
                    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
                    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, sampler=test_sampler,
                                                                num_workers=args.num_workers,
                                                                shuffle=False,
                                                                drop_last=True,
                                                                )
                    model, step = test(model, epoch, step, test_loader, Discriminator, writer,
                                                    reconstruction_data_loss_weight=args.reconstruction_data_loss_weight,
                                                    kl_latent_loss_weight=args.kl_latent_loss_weight,
                                                    batch_size=1,
                                                    n_channels=args.n_channels,
                                                    img_size=args.img_size,
                                                    rank = rank,
                                                    results_dir = results_dir)
                    args.phase = 'train'
    else:  

        test_dataset = make_test_loader(args.test_data_dir, device = rank)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, sampler=test_sampler,
                                                    num_workers=args.num_workers,
                                                    shuffle=False,
                                                    drop_last=True,
                                                    )   
        model, step = test(model, 1, step, test_loader, Discriminator, writer,
                            reconstruction_data_loss_weight=args.reconstruction_data_loss_weight,
                            kl_latent_loss_weight=args.kl_latent_loss_weight,
                            batch_size=1,
                            n_channels=args.n_channels,
                            img_size=args.img_size,
                            rank = rank,
                            results_dir = results_dir)


if __name__ == "__main__":
    # mp.set_start_method('spawn')

    args = parser.parse_args()
    args.train_data_dir = '/mnt/users/OAI_dess/oai_data/train/dess'
    args.test_data_dir = '/mnt/users/OAI_dess/oai_data/test/dess'
    args.num_workers = 3
    args.gpu_ids = [0,1,2]
    args.distributed = False
    args.resume_path = None
    # args.resume_path = '/mnt/users/videio_vae/experiments/3d_vae_sag_pd2025-01-10-19-14-03/gen/vol_256_lr_0.0001_kl_1e-05__bsize_2/checkpoints/model_00000001.pth.tar'
    args.phase = 'train'

    ''' cuda devices '''
    gpu_str = ','.join(str(x) for x in args.gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    ''' use DistributedDataParallel(DDP) and multiprocessing for multi-gpu training'''
    # [Todo]: multi GPU on multi machine
    if args.distributed:
        ngpus_per_node = len(args.gpu_ids) # or torch.cuda.device_count()
        args.world_size = ngpus_per_node
        mp.spawn(init_process, args=(args.world_size, main, (args,)), nprocs=args.world_size, join=True)
    else:
        args.world_size = 1 
        main(0, 1, args)

'''
Author: tom
Date: 2025-01-09 10:31:26
LastEditors: Do not edit
LastEditTime: 2025-01-10 18:09:30
FilePath: /videio_vae/3d-knee-vae/experiments/model/UNet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet as monaiUNet
from monai.networks.nets import DynUNet


class UNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, training=True):
        super(UNet, self).__init__()
        self.training = training
        self.encoder1 = nn.Conv3d(in_channel, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2=   nn.Conv3d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3=   nn.Conv3d(64, 128, 3, stride=1, padding=1)
        self.encoder4=   nn.Conv3d(128, 256, 3, stride=1, padding=1)
        # self.encoder5=   nn.Conv3d(256, 512, 3, stride=1, padding=1)
        
        # self.decoder1 = nn.Conv3d(512, 256, 3, stride=1,padding=1)  # b, 16, 5, 5
        self.decoder2 =   nn.Conv3d(256, 128, 3, stride=1, padding=1)  # b, 8, 15, 1
        self.decoder3 =   nn.Conv3d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 =   nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv3d(32, 2, 3, stride=1, padding=1)
        
        self.map4 = nn.Sequential(
            nn.Conv3d(2, out_channel, 1, 1),
            nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            nn.Upsample(scale_factor=(1, 8, 8), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(1, 16, 16), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),
            nn.Upsample(scale_factor=(1, 32, 32), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        self.out = nn.Conv3d(2, out_channel, 1, 1)

        self.monaiunet = monaiUNet(spatial_dims=3,
                            in_channels=1,
                            out_channels=2,
                            channels=(4, 8, 16),
                            strides=(2, 2),
                            num_res_units=2)

    def forward(self, series1, series2):

        x = series1.images #[1 1 256 256 32]
        aff1 = series1.affine
        gt = series2.images
        aff2 = series2.affine

        #============monai=============
        
        out = self.monaiunet(x)
        out = self.out(out)
        dic = { 'reconstruction':out,
               'aff':aff1,
               'input':x,
               'gt':gt,
            }

        # out = F.relu(F.max_pool3d(self.encoder1(x),(1, 2, 2),(1, 2, 2))) #[1, 32, 32, 192, 192]
        # t1 = out
        # out = F.relu(F.max_pool3d(self.encoder2(out),(1, 2, 2),(1, 2, 2)))#[1, 64, 32, 96, 96]
        # t2 = out
        # out = F.relu(F.max_pool3d(self.encoder3(out),(1, 2, 2),(1, 2, 2))) #[1, 128, 32, 48, 48]
        # t3 = out
        # out = F.relu(F.max_pool3d(self.encoder4(out),(1, 2, 2),(1, 2, 2))) #[1, 256, 32, 24, 24]
        # # t4 = out
        # # out = F.relu(F.max_pool3d(self.encoder5(out),2,2))
        
        # # t2 = out
        # # out = F.relu(F.interpolate(self.decoder1(out),scale_factor=(2,2,2),mode ='trilinear'))
        # # print(out.shape,t4.shape)
        # # output1 = self.map1(out)
        # out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(1,2,2),mode ='trilinear'))  #[1, 128, 32, 48, 48]
        # out = torch.add(out,t3)
        # # output2 = self.map2(out)
        # out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(1,2,2),mode ='trilinear')) #[1, 64, 32, 96, 96]
        # out = torch.add(out,t2)
        # # output3 = self.map3(out)
        # out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(1,2,2),mode ='trilinear')) #[1, 32, 32, 192, 192]
        # out = torch.add(out,t1)
        
        # out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(1,2,2),mode ='trilinear')) #[1, 2, 32, 384, 384]
        # output4 = self.out(out) #[1, 1, 32, 384, 384]
        # print(out.shape)
        # print(output1.shape,output2.shape,output3.shape,output4.shape)
        # if self.training is True:
        #     return output1, output2, output3, output4
        # else:
        # dic={  'reconstruction':output4,
        #        'aff':aff1,
        #        'input':x,
        #        'gt':gt,
        #     }
        return dic
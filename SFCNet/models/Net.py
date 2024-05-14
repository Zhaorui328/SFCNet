# -*- coding: utf-8 -*-
"""
@author: caigentan@AnHui University
@software: PyCharm
@file: SwinTransformer.py
@time: 2021/5/6 7:23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model import  Conv, UpSampler, FAI, DWT, BasicBlockL, BasicBlockH, CIU
from models.smt import smt_t


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()


        channels = [64, 128, 256, 512]
        self.channels = channels
        self.rgb_swin = smt_t()


        # ------------------------------------------------------------------------------------------------------------------------------
        self.conv_rgb0 = nn.Sequential(nn.Conv2d(self.channels[0], self.channels[1], 1, 1, 0), nn.BatchNorm2d(self.channels[1]), nn.ReLU())
        self.conv_rgb1 = nn.Sequential(nn.Conv2d(self.channels[1], self.channels[1], 1, 1, 0), nn.BatchNorm2d(self.channels[1]), nn.ReLU())
        self.conv_rgb2 = nn.Sequential(nn.Conv2d(self.channels[2], self.channels[1], 1, 1, 0), nn.BatchNorm2d(self.channels[1]), nn.ReLU())
        self.conv_rgb3 = nn.Sequential(nn.Conv2d(self.channels[3], self.channels[1], 1, 1, 0), nn.BatchNorm2d(self.channels[1]), nn.ReLU())


        self.DWT = DWT()
        self.after_dwt0 = nn.BatchNorm2d(self.channels[1])
        self.relu = nn.LeakyReLU(0.2)
        self.conv2_3 = self._make_layer(BasicBlockL, self.channels[1], self.channels[1])
        self.conv2_4 = self._make_layer(BasicBlockH, self.channels[1]*3, self.channels[1]*3)

        self.up_1 = nn.Sequential(UpSampler(scale=2, n_feats=self.channels[1] * 3))
        self.up_2 = nn.Sequential(UpSampler(scale=4, n_feats=self.channels[1] * 3))
        self.up_3 = nn.Sequential(UpSampler(scale=2, n_feats=self.channels[1]))
        self.up_4 = nn.Sequential(UpSampler(scale=4, n_feats=self.channels[1]))

        self.fusion_h2 = nn.Conv2d(in_channels=3 * self.channels[1] * 3, out_channels=self.channels[1] * 3, kernel_size=1, stride=1, padding=0)
        self.fusion_h3 = nn.Conv2d(in_channels=self.channels[1] * 3, out_channels=self.channels[1], kernel_size=1,
                                   stride=1, padding=0)
        self.conv2 =  nn.Conv2d(in_channels=self.channels[1] * 3, out_channels=self.channels[1], kernel_size=1, stride=1, padding=0)


        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32 + 1),
            nn.Sigmoid(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(self.channels[1], self.channels[1], 3, 1, 1),
            nn.BatchNorm2d(self.channels[1]),
            nn.ReLU(),
            nn.Conv2d(self.channels[1], 1, 1, 1, 0)
        )

        self.fuse = FAI(self.channels[1], num_heads=8, level=1)

        self.A2SP5 = CIU(self.channels[1], self.channels[1])
        self.A2SP4 = CIU(self.channels[1], self.channels[1])
        self.A2SP3 = CIU(self.channels[1], self.channels[1])
        self.A2SP2 = CIU(self.channels[1], self.channels[1])


        # 训练时额外输出stage2-5的结果
        if self.training:
            self.Sal_Head_2 = Conv(self.channels[1])
            self.Sal_Head_3 = Conv(self.channels[1])
            self.Sal_Head_4 = Conv(self.channels[1])
            self.Sal_Head_5 = Conv(self.channels[1])

        # 否则只用最后一个stage输出
        self.Sal_Head_1 = Conv(self.channels[1])

    def _make_layer(self, block, inplanes, out_planes):
        layers = []
        layers.append(block(inplanes, out_planes))
        return nn.Sequential(*layers)


    def forward(self, RGB):
        image_size = RGB.size()[2:]
        baseline_net = self.rgb_swin(RGB)

        # ------------------------------------------Encoder--------------------------------------------------------------------
        # backbone提取特征
        Fr1 = baseline_net[0]
        Fr2 = baseline_net[1]
        Fr3 = baseline_net[2]
        Fr4 = baseline_net[3]

        # 通道对齐
        Fr1 = self.conv_rgb0(Fr1)
        Fr2 = self.conv_rgb1(Fr2)
        Fr3 = self.conv_rgb2(Fr3)
        Fr4 = self.conv_rgb3(Fr4)

        # ------------------------------------------DTF--------------------------------------------------------------------

        #----------SFT------
        LL_2, LH_2, HL_2, HH_2 = self.DWT(Fr2)
        LL_2 = self.after_dwt0(LL_2)
        LH_2 = self.after_dwt0(LH_2)
        HL_2 = self.after_dwt0(HL_2)
        HH_2 = self.after_dwt0(HH_2)
        H_2 = torch.cat((LH_2, HL_2, HH_2), 1)
        L_2 = self.conv2_3(LL_2)
        H_2 = self.conv2_4(H_2)


        LL_3, LH_3, HL_3, HH_3 = self.DWT(Fr3)
        LL_3 = self.after_dwt0(LL_3)
        LH_3 = self.after_dwt0(LH_3)
        HL_3 = self.after_dwt0(HL_3)
        HH_3 = self.after_dwt0(HH_3)
        H_3 = torch.cat((LH_3, HL_3, HH_3), 1)
        L_3 = self.conv2_3(LL_3)
        H_3 = self.conv2_4(H_3)

        LL_4, LH_4, HL_4, HH_4 = self.DWT(Fr4)
        LL_4 = self.after_dwt0(LL_4)
        LH_4 = self.after_dwt0(LH_4)
        HL_4 = self.after_dwt0(HL_4)
        HH_4 = self.after_dwt0(HH_4)
        H_4 = torch.cat((LH_4, HL_4, HH_4), 1)
        L_4 = self.conv2_3(LL_4)
        H_4 = self.conv2_4(H_4)

        H_3 = self.up_1(H_3)
        H_4 = self.up_2(H_4)
        high2 = self.fusion_h2(torch.cat([H_3, H_4, H_2], dim=1))
        high = self.conv2(high2)

        L_3 = self.up_3(L_3)
        L_4 = self.up_4(L_4)
        low = self.fusion_h3(torch.cat([L_3, L_4, L_2], dim=1))

        F5 = self.fuse(high, low)

        # -------------BA-------------
        bz = RGB.shape[0]
        rgb_gap = self.gap1(Fr1)
        rgb_gap = rgb_gap.view(bz, -1)
        feat = self.fc(rgb_gap)
        gate = feat[:, -1].view(bz, 1, 1, 1)
        edge = gate * high

        # -----------------------------------------解码---------------------------------------------------------------------
        F4 = self.A2SP5(Fr4, F5, edge)

        F3 = self.A2SP4(Fr3 + F.interpolate(F4, Fr3.shape[2:], mode='bilinear', align_corners=False), F4, edge)

        F2 = self.A2SP3(Fr2 + F.interpolate(F4, Fr2.shape[2:], mode='bilinear', align_corners=False) +
                            F.interpolate(F3, Fr2.shape[2:], mode='bilinear', align_corners=False), F3, edge)

        F1 = self.A2SP2(Fr1 + F.interpolate(F4, Fr1.shape[2:], mode='bilinear', align_corners=False) +
                            F.interpolate(F3, Fr1.shape[2:], mode='bilinear', align_corners=False) +
                            F.interpolate(F2, Fr1.shape[2:], mode='bilinear', align_corners=False), F2, edge)

        if self.training:
            # 最后做一次卷积，恢复到原来图片大小并输出结果
            F5_out = F.interpolate(self.Sal_Head_4(F5), image_size, mode='bilinear', align_corners=False)
            F4_out = F.interpolate(self.Sal_Head_4(F4), image_size, mode='bilinear', align_corners=False)
            F3_out = F.interpolate(self.Sal_Head_3(F3), image_size, mode='bilinear', align_corners=False)
            F2_out = F.interpolate(self.Sal_Head_2(F2), image_size, mode='bilinear', align_corners=False)
            F1_out = F.interpolate(self.Sal_Head_1(F1), image_size, mode='bilinear', align_corners=False)
            edge_out = F.interpolate(self.conv(edge), image_size, mode='bilinear', align_corners=False)

            return F1_out, F2_out, F3_out, F4_out, F5_out, edge_out
        else:

            F_out = F.interpolate(self.Sal_Head_1(F1), image_size, mode='bilinear', align_corners=False)
            return F_out


    def load_pre(self, pre_model):
        self.rgb_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"Net loading pre_model ${pre_model}")







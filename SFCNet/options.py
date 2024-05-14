# -*- coding: utf-8 -*-
"""
@author: caigentan@AnHui University
@software: PyCharm
@file: options.py
@time: 2021/5/16 14:52

"""
import argparse
# RGBD
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--nb_timestep', type=float, default=3, help='number of timestep')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default='./pretrain/smt_tiny.pth', help='train from checkpoints')
parser.add_argument('--load_pre', type=str, default='./SwinTransNet_RGBD_cpts/SwinTransNet_epoch_best.pth', help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
parser.add_argument('--rgb_root', type=str, default='./TrainDataset/Imgs/', help='the training rgb images root')
parser.add_argument('--gt_root', type=str, default='./TrainDataset/GT/', help='the training gt images root')
parser.add_argument('--edge_root', type=str, default='./TrainDataset/Edge/', help='the training edge images root')
parser.add_argument('--test_rgb_root', type=str, default='./TestDataset/CAMO/Imgs/', help='the test gt images root')
parser.add_argument('--test_gt_root', type=str, default='./TestDataset/CAMO/GT/', help='the test gt images root')
parser.add_argument('--test_edge_root', type=str, default='./TestDataset/CAMO/Edge/', help='the test edge images root')
parser.add_argument('--save_path', type=str, default='./cpts/', help='the path to save models and logs')
opt = parser.parse_args()
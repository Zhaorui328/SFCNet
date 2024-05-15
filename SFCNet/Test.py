
"""
@author: Zhaorui@Dalian Minzu University
@software: PyCharm
@file: Test.py
@time: 2024/5/14 09:34
"""

import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from models.Net import Net
from data import test_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='./TestDataset/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

#load the model
model = Net()
model.load_state_dict(torch.load('cpts/Net_epoch_best.pth'),strict=False)
model.cuda()
model.eval()


# test
test_datasets = ['CHAMELEON', 'COD10K','NC4K','CAMO']
for dataset in test_datasets:
    save_path = './test_maps/Net/' + dataset + '/'
    edge_save_path = './test_maps/Net/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(edge_save_path)
    image_root = dataset_path + dataset + '/Imgs/'
    gt_root = dataset_path + dataset + '/GT/'
    depth_root = dataset_path + dataset + '/depth/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)


    for i in range(test_loader.size):
        image, gt, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        F_out = model(image)
        F_out = F.upsample(F_out, size=gt.shape, mode='bilinear', align_corners=False)
        F_out = F_out.sigmoid().data.cpu().numpy().squeeze()
        F_out = (F_out - F_out.min()) / (F_out.max() - F_out.min() + 1e-8)
        print('save img to: ',save_path + name)
        cv2.imwrite(save_path + name, F_out*255)

    print('Test Done!')

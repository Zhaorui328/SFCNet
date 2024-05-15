
"""
@author: zhaorui@Dalian Minzu University
@software: PyCharm
@file: Text.py
@time: 2024/5/14 16:12
"""
import os
import torch
import torch.nn.functional as F
import sys
import torch.nn as nn

sys.path.append('./models')
import numpy as np
from datetime import datetime
from models.Net import Net
from torchvision.utils import make_grid
from data import get_loader, test_dataset
from utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import pytorch_losses
import torch.backends.cudnn as cudnn
from options import opt
import torch.nn as nn
import torch
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable

# set the device for training
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
cudnn.benchmark = True

image_root = opt.rgb_root
gt_root = opt.gt_root
edge_root = opt.edge_root


test_image_root = opt.test_rgb_root
test_gt_root = opt.test_gt_root

save_path = opt.save_path

logging.basicConfig(filename=save_path + 'Net.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("Net-Train_4_pairs")


model = Net()

num_parms = 0
if (opt.load is not None):
    model.load_pre(opt.load)
    print('load model from ', opt.load)

model.cuda()
for p in model.parameters():
    num_parms += p.numel()
logging.info("Total Parameters (For Reference): {}".format(num_parms))
print("Total Parameters (For Reference): {}".format(num_parms))

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

# set the path

if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')
train_loader = get_loader(image_root, gt_root , edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_image_root, test_gt_root, opt.trainsize)
total_step = len(train_loader)

logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))

# set loss function
CE = torch.nn.BCEWithLogitsLoss()
ECE = torch.nn.BCELoss()
step = 0
writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 0

#--------------------------------------------edge Loss-------------------------------------------
def cross_entropy2d_edge(input, target, reduction='mean'):

    assert (input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg / num_total
    beta = 1.1 * num_pos / num_total
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)


def Hybrid_Loss(pred, target, reduction='mean'):
    # 先对输出做归一化处理
    pred = torch.sigmoid(pred)

    # BCE LOSS
    bce_loss = nn.BCELoss()
    bce_out = bce_loss(pred, target)

    # IOU LOSS
    iou_loss = pytorch_losses.IOU(reduction=reduction)
    iou_out = iou_loss(pred, target)

    # SSIM LOSS
    ssim_loss = pytorch_losses.SSIM(window_size=11)
    ssim_out = ssim_loss(pred, target)


    hybrid_loss = [bce_out, iou_out, ssim_out]
    losses = bce_out + iou_out + ssim_out

    return hybrid_loss, losses

# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()

    epoch_step = 0

    try:
        for i, (images, gts, edge) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            edge = edge.cuda()


            ############################################################################################################
            losses_all, loss_1_all, loss_2_all, loss_3_all, loss_4_all = 0, 0, 0, 0, 0
            F1_out, F2_out, F3_out, F4_out, F5_out, edge_out = model(images)
            hybrid_loss_1, loss_1 = Hybrid_Loss(F1_out, gts)
            hybrid_loss_2, loss_2 = Hybrid_Loss(F2_out, gts)
            hybrid_loss_3, loss_3 = Hybrid_Loss(F3_out, gts)
            hybrid_loss_4, loss_4 = Hybrid_Loss(F4_out, gts)
            hybrid_loss_5, loss_5 = Hybrid_Loss(F5_out, gts)
            loss_2_edge = cross_entropy2d_edge(edge_out, edge)
            losses = loss_1 + loss_2 + loss_3 + loss_4 + loss_2_edge + loss_5

            # 反向传播
            losses.backward()
            clip_gradient(optimizer, opt.clip)  # pytorch中梯度修剪，防止梯度爆炸的方法
            optimizer.step()
            step += 1
            epoch_step += 1
            losses_all += losses.data

            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

            if i % 200 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f} ||losses:{:4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step,
                             optimizer.state_dict()['param_groups'][0]['lr'],losses))
                # 运行记录中输出损失函数
                logging.info(
                    '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}, ||loss:{:4f}||loss1:{:4f}||loss2:{:4f}||loss3:{:4f}||loss4:{:4f}, mem_use:{:.0f}MB'.
                        format(epoch, opt.epoch, i, total_step, optimizer.state_dict()['param_groups'][0]['lr'],
                               losses.data, loss_1.data, loss_2.data,
                               loss_3.data, loss_4.data, memory_used))
                writer.add_scalar('Loss', losses.data, global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)
                res = F1_out[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('res', torch.tensor(res), step, dataformats='HW')

        losses_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}],Losses_AVG: {:.4f}'.format(epoch, opt.epoch,losses_all))

        writer.add_scalar('Loss-epoch', losses_all,  global_step=epoch)


        if (epoch) % 2 == 0:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise


def test(test_loader, model, epoch, save_path, writer):

    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt,  name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            result = model(image)

            res = F.upsample(result, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
        logging.info(
            '[Val Info]:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")
    # 初次衰减循环增大10个epoch即110后才进行第一次衰减
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        test(test_loader, model, epoch, save_path,writer)

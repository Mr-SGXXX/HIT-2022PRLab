import logging
import os
import random

import numpy as np
import torch
import argparse

from torchvision.utils import save_image
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from dataset import Flickr2K
from model import IMDN
from train import train_one_epoch, eval_one_epoch
from metrics import compute_psnr, compute_ssim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data_path', default='./data', help="数据集存放位置")
    parser.add_argument('-e', '--epoch', type=int, default=50, help="迭代轮数")
    parser.add_argument('-se', '--start_epoch', type=int, default=0, help="初始迭代轮数")
    parser.add_argument('-lp', '--log_path', default="log/result.log", help="日志路径")
    parser.add_argument('-d', '--device', default="cuda:0" if torch.cuda.is_available() else "cpu", help="使用的计算设备")
    parser.add_argument('-lr', '--learn_rate', type=float, default=1e-4, help="学习率")
    parser.add_argument('-op', '--optimizer', default="Adam", help="优化器设置Adam/SGD")
    parser.add_argument('-wd', '--weight_decay', type=float, default=0, help="正则化项系数")
    return parser.parse_args()


def model_train(epoch, start_epoch, learn_rate, weight_decay, model, optimizer_type, data_path, logger, device):
    if os.path.exists(f"./weight/IMDN_{optimizer_type}_{learn_rate}_{start_epoch - 1}.pth"):
        model.load_state_dict(torch.load(f"./weight/IMDN_{optimizer_type}_{learn_rate}_{start_epoch - 1}.pth"))
        logger.info(f"Load Last Epoch({start_epoch - 1}) Success")
    else:
        logger.info(f"Train From Beginning")
        if not os.path.exists("./weight/"):
            os.mkdir("weight")

    train_dataset = Flickr2K(data_path, 0)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    eval_dataset = Flickr2K(data_path, 1)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    if optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    elif optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    elif optimizer_type == "Momentum":
        optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type == "RMSProp":
        optimizer = optim.RMSprop(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    else:
        logger.info("Optimizer Choice Error")
        raise RuntimeError("优化器选择错误")

    step_lr = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_func = nn.L1Loss()
    best_psnr = 0
    best_ssim = 0
    best_epoch = 0

    for cur_epoch in range(start_epoch, epoch):
        loss = train_one_epoch(model, train_dataloader, optimizer, loss_func, device)
        torch.save(model.state_dict(), f"./weight/IMDN_{optimizer_type}_{learn_rate}_{cur_epoch}.pth")
        if os.path.exists(f"./weight/IMDN_{optimizer_type}_{learn_rate}_{cur_epoch - 1}.pth"):
            os.remove(f"./weight/IMDN_{optimizer_type}_{learn_rate}_{cur_epoch - 1}.pth")
        logger.info(f"Epoch: {cur_epoch + 1}/{epoch}\tTrain Loss:{loss}")
        loss, psnr, ssim = eval_one_epoch(model, eval_dataloader, loss_func, device)
        step_lr.step()
        if psnr > best_psnr:
            best_psnr = psnr
            best_ssim = ssim
            best_epoch = cur_epoch
            torch.save(model.state_dict(), f"./weight/IMDN_{optimizer_type}_{learn_rate}_best.pth")
        logger.info(
            f"Eval Loss:{loss}\tPSNR/SSIM:{psnr}/{ssim}\tBest PSNR/SSIM/Epoch:{best_psnr}/{best_ssim}/{best_epoch + 1}")
    logger.info(f"Train Phase Over\tBest PSNR/SSIM/Epoch:{best_psnr}/{best_ssim}/{best_epoch + 1}")


def model_test(epoch, learn_rate, model, optimizer, data_path, logger, device):
    model.eval()
    test_dataset = Flickr2K(data_path, 2)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if not os.path.exists("./result"):
        os.mkdir("result")
    if not os.path.exists("./result/result_last"):
        os.mkdir("result/result_last")
    if not os.path.exists("./result/result_best"):
        os.mkdir("result/result_best")

    loss_func = nn.L1Loss()
    psnr_total = 0
    ssim_total = 0
    loss_total = 0.0
    model.load_state_dict(torch.load(f"./weight/IMDN_{optimizer}_{learn_rate}_{epoch - 1}.pth"))
    with torch.no_grad():
        for lr, hr, name in test_dataloader:
            lr = lr.to(device)
            hr = hr.to(device)
            output = model(lr)
            save_image(output[0].cpu().detach(), f"./result/result_last/{name[0]}_x4.png")
            loss = loss_func(output, hr)
            psnr = compute_psnr(output, hr)
            ssim = compute_ssim(output, hr)
            loss_total += loss.item()
            psnr_total += psnr
            ssim_total += ssim
    logger.info(f"Last Epoch Test Loss:{loss_total / len(test_dataloader)}\t"
                f"PSNR/SSIM:{psnr_total / len(test_dataloader)}/{ssim_total / len(test_dataloader)}")

    loss_total = 0.0
    psnr_total = 0
    ssim_total = 0
    model.load_state_dict(torch.load(f"./weight/IMDN_{optimizer}_{learn_rate}_best.pth"))
    with torch.no_grad():
        for lr, hr, name in test_dataloader:
            lr = lr.to(device)
            hr = hr.to(device)
            output = model(lr)
            save_image(output[0].cpu().detach(), f"./result/result_best/{name[0]}")
            loss = loss_func(output, hr)
            psnr = compute_psnr(output, hr)
            ssim = compute_ssim(output, hr)
            loss_total += loss.item()
            psnr_total += psnr
            ssim_total += ssim
    logger.info(f"Best Epoch Test Loss:{loss_total / len(test_dataloader)}\t"
                f"PSNR/SSIM:{psnr_total / len(test_dataloader)}/{ssim_total / len(test_dataloader)}")


def main():
    args = parse_args()
    if not os.path.exists(os.path.dirname(args.log_path)):
        os.mkdir(os.path.dirname(args.log_path))

    # 设置日志
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(args.log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s]-[%(levelname)s]: %(message)s')
    handler.setFormatter(formatter)
    # 让日志也输出在控制台上
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # 固定随机种子、设置设备
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    np.random.seed(10)
    random.seed(10)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Learn Rate:{args.learn_rate}\t"
                f"Optimizer:{args.optimizer}\t"
                f"Weight Decay:{args.weight_decay}\tDevice:{args.device}\t"
                f"Total Epoch:{args.epoch}\tStart Epoch:{args.start_epoch}")
    model = IMDN().to(args.device)
    # model_train(args.epoch, args.start_epoch, args.learn_rate, args.weight_decay, model,
    #             args.optimizer, args.data_path, logger, args.device)
    model_test(args.epoch, args.learn_rate, model, args.optimizer, args.data_path, logger, args.device)


if __name__ == "__main__":
    main()

import argparse
import os
import logging
import random

import csv
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import PlantSeeding
from model import *
from train import train_one_epoch, eval_one_epoch

Cat_list = ["Black-grass", "Charlock", "Cleavers", "Common Chickweed", "Common wheat", "Fat Hen", "Loose Silky-bent",
            "Maize", "Scentless Mayweed", "Shepherds Purse", "Small-flowered Cranesbill", "Sugar beet"]

Transforms_train = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        transforms.Resize([224, 224]),
        transforms.Normalize([0.32895455, 0.28938746, 0.20749362], [0.09349957, 0.09730683, 0.10652738])
    ]
)

Transforms_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize([224, 224]),
        transforms.Normalize([0.32895455, 0.28938746, 0.20749362], [0.09349957, 0.09730683, 0.10652738])
    ]
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='VGG', help="选择要使用的模型：VGG/ResNet/SENet")
    parser.add_argument('-bs', '--batch_size', type=int, default=16, help="设置Batch Size")
    parser.add_argument('-dp', '--data_path', default='./plant-seedlings-classification', help="数据集存放位置")
    parser.add_argument('-e', '--epoch', type=int, default=100, help="迭代轮数")
    parser.add_argument('-se', '--start_epoch', type=int, default=0, help="初始迭代轮数")
    parser.add_argument('-lp', '--log_path', default="log/VGG.log", help="日志路径")
    parser.add_argument('-d', '--device', default="cuda:0" if torch.cuda.is_available() else "cpu", help="使用的计算设备")
    parser.add_argument('-wn', '--weight_name', default="VGG", help="权重名称")
    parser.add_argument('-lr', '--learn_rate', type=float, default=1e-4, help="学习率")
    parser.add_argument('-op', '--optimizer', default="Adam", help="优化器设置Adam/SGD")
    parser.add_argument('-da', '--data_augmentation', type=bool, default=False, help="是否使用数据增强")
    parser.add_argument('-wd', '--weight_decay', type=float, default=0, help="正则化项系数")
    return parser.parse_args()


def get_mean_std(data_path):
    train_dataset = PlantSeeding(data_path, 0)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_dataset))
    std.div_(len(train_dataset))
    return list(mean.numpy()), list(std.numpy())


def model_train(epoch, start_epoch, learn_rate, weight_decay, batch_size, device, model, optimizer, transform_train,
                transfrom_eval, weight_name, data_path, logger):
    if os.path.exists(f"./weight/{weight_name}_{start_epoch - 1}.pth"):
        model.load_state_dict(torch.load(f"./weight/{weight_name}_{start_epoch - 1}.pth"))
        logger.info(f"Load Last Epoch({start_epoch - 1}) Success")
    else:
        logger.info(f"Train From Beginning")
        if not os.path.exists("./weight/"):
            os.mkdir("weight")

    train_dataset = PlantSeeding(data_path, 0, cat_list=Cat_list, transforms=transform_train)
    train_dataloader = DataLoader(train_dataset, batch_size, True)
    eval_dataset = PlantSeeding(data_path, 1, cat_list=Cat_list, transforms=transfrom_eval)
    eval_dataloader = DataLoader(eval_dataset, batch_size, False)

    if optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    elif optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    else:
        logger.info("Optimizer Choice Error")
        raise RuntimeError("优化器选择错误")
    step_lr = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_func = nn.CrossEntropyLoss()
    best_acc = 0
    best_epoch = 0
    for cur_epoch in range(start_epoch, epoch):
        loss = train_one_epoch(model, train_dataloader, optimizer, loss_func, device)
        torch.save(model.state_dict(), f"./weight/{weight_name}_{cur_epoch}.pth")
        if os.path.exists(f"./weight/{weight_name}_{cur_epoch - 1}.pth"):
            os.remove(f"./weight/{weight_name}_{cur_epoch - 1}.pth")
        logger.info(f"Epoch: {cur_epoch + 1}/{epoch}\tTrain Loss:{loss}")
        loss, acc = eval_one_epoch(model, eval_dataloader, loss_func, device)
        step_lr.step()
        if acc > best_acc:
            best_acc = acc
            best_epoch = cur_epoch
            torch.save(model.state_dict(), f"./weight/{weight_name}_best.pth")
        logger.info(f"Eval Loss:{loss}\tAccuracy:{acc}\tBest Accuracy/Epoch:{best_acc}/{best_epoch + 1}")
    logger.info(f"Train Phase Over\tBest Accuracy/Epoch:{best_acc}/{best_epoch + 1}")


def model_test(epoch, batch_size, device, model, transform, weight_name, data_path, logger):
    model.to(device)
    model.eval()
    test_dataset = PlantSeeding(data_path, 2, transforms=transform)
    test_dataloader = DataLoader(test_dataset, batch_size, False)

    if not os.path.exists("./result"):
        os.mkdir("result")
    labels = ['file', 'species']

    img_num = 0
    total_time = 0
    model.load_state_dict(torch.load(f"./weight/{weight_name}_best.pth"))
    with open(f'result/{weight_name}_best.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=labels)
        writer.writeheader()
        for imgs, paths in test_dataloader:
            img_num += imgs.size(0)
            imgs = imgs.to(device)
            start_time = time.time()
            outputs = model(imgs)
            end_time = time.time()
            total_time += end_time - start_time
            outputs = torch.argmax(outputs, dim=1)
            for output, path in zip(outputs, paths):
                writer.writerow({'file': path, 'species': Cat_list[output]})
        logger.info("Best Weight Result Generated")

    model.load_state_dict(torch.load(f"./weight/{weight_name}_{epoch - 1}.pth"))
    with open(f'result/{weight_name}_last.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=labels)
        writer.writeheader()
        for imgs, paths in test_dataloader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            outputs = torch.argmax(outputs, dim=1)
            for output, path in zip(outputs, paths):
                writer.writerow({'file': path, 'species': Cat_list[output]})
        logger.info("Last Weight Result Generated")
    logger.info(f"Averaged Single Image Process Time:{total_time * 1000 / img_num}ms")


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

    # 固定随机种子
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    np.random.seed(10)
    random.seed(10)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Model:{args.model}\tBatch Size:{args.batch_size}\t"
                f"Total Epoch:{args.epoch}\tStart Epoch:{args.start_epoch}")
    if args.model == 'VGG':
        model = VGG().to(args.device)
    elif args.model == 'ResNet':
        model = ResNet().to(args.device)
    elif args.model == 'SENet':
        model = SENet().to(args.device)
    else:
        logger.info("Model Choice Error")
        raise RuntimeError("模型选项错误")
    model_train(args.epoch, args.start_epoch, args.learn_rate, args.weight_decay, args.batch_size, args.device, model,
                args.optimizer,
                Transforms_train if args.data_augmentation else Transforms_test, Transforms_test, args.weight_name,
                args.data_path, logger)
    model_test(args.epoch, args.batch_size, args.device, model, Transforms_test, args.weight_name, args.data_path,
               logger)


if __name__ == "__main__":
    # print(get_mean_std('./plant-seedlings-classification'))
    main()

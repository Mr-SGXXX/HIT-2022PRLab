import os

import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
from dataset import Caltech101
from train import train_one_epoch, eval_model
from torch.utils.data import DataLoader
import numpy as np
import random
from model import AlexNet
import logging
from torch.utils.tensorboard import SummaryWriter

TrainTransformer = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        # transforms.RandomResizedCrop(224),
        transforms.Resize([224, 224]),
        transforms.Normalize([0.5], [0.5])
    ]
)

TestTransformer = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize([224, 224]),
        transforms.Normalize([0.5], [0.5])
    ]
)

Writer = SummaryWriter('./log')


def train(epoch_num, last_epoch, model, device, data_dir, logger=None):
    train_dataset = Caltech101(data_dir, 0, TrainTransformer)
    train_dataloader = DataLoader(train_dataset, 256, shuffle=True)
    eval_dataset = Caltech101(data_dir, 1, TestTransformer)
    eval_dataloader = DataLoader(eval_dataset, 256, shuffle=False)
    loss_func = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    optimizer = optim.Adam(model.parameters(), weight_decay=0.0005)
    step_lr = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    best_acc = 0.
    for epoch in range(last_epoch, epoch_num):
        loss = train_one_epoch(model, device, train_dataloader, optimizer, loss_func)
        Writer.add_scalar('loss/train loss', loss, epoch)
        if logger is not None:
            logger.info(f"Epoch: {epoch + 1}/{epoch_num}\tLoss:{loss}")
        if epoch % 10 == 9:
            loss, acc = eval_model(model, 'cpu', eval_dataloader, loss_func)
            Writer.add_scalar('loss/eval loss', loss, epoch)
            Writer.add_scalar('eval accuracy', acc)
            step_lr.step()
            # 保存模型参数
            if acc > best_acc:
                torch.save(model.state_dict(), "./best_model_weight.pth")
                best_acc = acc
            if logger is not None:
                logger.info(f"Eval Loss = {loss}\t Eval Accuracy = {acc}\t Current Best Accuracy = {best_acc}")


def model_test(model, device, data_dir, logger):
    model.to(device)
    test_dataset = Caltech101(data_dir, 2, TestTransformer)
    test_dataloader = DataLoader(test_dataset, 256, shuffle=False)
    model.eval()
    loss_func = nn.CrossEntropyLoss()
    loss_total = 0.0
    right_num = 0
    total_num = 0
    for img, label in test_dataloader:
        img = img.to(device)
        label = label.to(device)
        output = model(img)
        loss = loss_func(output, label)
        right_num += (torch.argmax(output, dim=1) == label).sum().item()
        total_num += label.size(0)
        loss_total += loss.item()
    if logger is not None:
        logger.info(f"Test Loss:{loss_total / len(test_dataloader)}\tTest Accuracy:{right_num / total_num}")

  
def main():
    model_weight_path = "best_model_weight.pth"
    # model_weight_path = None
    data_dir = "./data/Caltech101"
    epoch_num = 500
    last_epoch = 0
    torch.backends.cudnn.benchmark = True

    # 设置日志
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler("train_log.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s]-[%(levelname)s]: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # 固定随机数
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    np.random.seed(10)
    random.seed(10)

    model = AlexNet(102).to(device)
    if model_weight_path is not None:
        try:
            model.load_state_dict(torch.load(model_weight_path))
            logger.info("Load Model Weight Success")
        except:
            logger.info("Failed To Load Model Weight")
    train(epoch_num, 0, model, device, data_dir, logger)
    torch.save(model.state_dict(), 'last_epoch_weight.pth')
    logger.info("Last Epoch Test")
    model_test(model, device, data_dir, logger)
    logger.info("Best Eval Test")
    model.load_state_dict(torch.load(model_weight_path))
    model_test(model, device, data_dir, logger)


if __name__ == "__main__":
    main()

import logging
import os
import random

import numpy as np
import paddle
import argparse
from paddle.io import DataLoader
from paddle.vision import transforms
import paddle.optimizer as optim
import paddle.nn as nn

from dataset import CUB
from model import MobileNetV2
from train import train_one_epoch, eval_one_epoch

Transforms_train = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.Resize([224, 224])
    ]
)

Transforms_test = transforms.Compose(
    [
        transforms.Resize([224, 224])
    ]
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help="设置Batch Size")
    parser.add_argument('-dp', '--data_path', default='./data/CUB_200_2011', help="数据集存放位置")
    parser.add_argument('-e', '--epoch', type=int, default=100, help="迭代轮数")
    parser.add_argument('-se', '--start_epoch', type=int, default=0, help="初始迭代轮数")
    parser.add_argument('-lp', '--log_path', default="log/result.log", help="日志路径")
    parser.add_argument('-d', '--device', default=paddle.get_device(), help="使用的计算设备")
    parser.add_argument('-lr', '--learn_rate', type=float, default=5e-4, help="学习率")
    parser.add_argument('-op', '--optimizer', default="Adam", help="优化器设置Adam/SGD")
    parser.add_argument('-da', '--data_augmentation', type=bool, default=True, help="是否使用数据增强")
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-5, help="正则化项系数")
    return parser.parse_args()


def model_train(epoch, start_epoch, learn_rate, weight_decay, batch_size, model, optimizer, transform_train,
                transfrom_eval, data_path, logger):
    if os.path.exists(f"./weight/MobileNetV2_{optimizer}_{learn_rate}_{start_epoch - 1}.pdparams"):
        model.load_state_dict(paddle.load(f"./weight/MobileNetV2_{optimizer}_{learn_rate}_{start_epoch - 1}.pdparams"))
        logger.info(f"Load Last Epoch({start_epoch - 1}) Success")
    else:
        logger.info(f"Train From Beginning")
        if not os.path.exists("./weight/"):
            os.mkdir("weight")

    train_dataset = CUB(data_path, 0, transforms=transform_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataset = CUB(data_path, 1, transforms=transfrom_eval)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    learn_rate = optim.lr.StepDecay(learn_rate, step_size=20, gamma=0.5)

    if optimizer == "Adam":
        optimizer = optim.Adam(parameters=model.parameters(), learning_rate=learn_rate, weight_decay=weight_decay)
    elif optimizer == "SGD":
        optimizer = optim.SGD(parameters=model.parameters(), learning_rate=learn_rate, weight_decay=weight_decay)
    elif optimizer == "Momentum":
        optimizer = optim.Momentum(parameters=model.parameters(), learning_rate=learn_rate, weight_decay=weight_decay)
    elif optimizer == "RMSProp":
        optimizer = optim.RMSProp(parameters=model.parameters(), learning_rate=learn_rate, weight_decay=weight_decay)
    else:
        logger.info("Optimizer Choice Error")
        raise RuntimeError("优化器选择错误")
    loss_func = nn.CrossEntropyLoss()
    best_acc = 0
    best_epoch = 0

    for cur_epoch in range(start_epoch, epoch):
        loss = train_one_epoch(model, train_dataloader, optimizer, loss_func)
        paddle.save(model.state_dict(), f"./weight/MobileNetV2_{cur_epoch}.pdparams")
        if os.path.exists(f"./weight/MobileNetV2_{cur_epoch - 1}.pdparams"):
            os.remove(f"./weight/MobileNetV2_{cur_epoch - 1}.pdparams")
        logger.info(f"Epoch: {cur_epoch + 1}/{epoch}\tTrain Loss:{loss.item()}")
        loss, acc = eval_one_epoch(model, eval_dataloader, loss_func)
        learn_rate.step()
        if acc > best_acc:
            best_acc = acc
            best_epoch = cur_epoch
            paddle.save(model.state_dict(), f"./weight/MobileNetV2_best.pdparams")
        logger.info(f"Eval Loss:{loss}\tAccuracy:{acc}\tBest Accuracy/Epoch:{best_acc}/{best_epoch + 1}")
    logger.info(f"Train Phase Over\tBest Accuracy/Epoch:{best_acc}/{best_epoch + 1}")


def model_test(epoch,batch_size, model, transform, data_path, logger):
    model.eval()
    test_dataset = CUB(data_path, 2, transforms=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    loss_func = nn.CrossEntropyLoss()
    loss_total = 0.0
    right_num = 0
    total_num = 0
    model.load_dict(paddle.load(f"./weight/MobileNetV2_best.pdparams"))
    with paddle.no_grad():
        for img, label in test_dataloader:
            label = label.astype("int64")
            output = model(img)
            loss = loss_func(output, label)
            right_num += paddle.sum(paddle.argmax(output, axis=1) == label).item()
            total_num += output.shape[0]
            loss_total += loss.item()
    logger.info(f"Best Epoch Test Loss:{loss_total / len(test_dataloader)}\tAccuracy:{right_num / total_num}")

    loss_total = 0.0
    right_num = 0
    total_num = 0
    model.load_dict(paddle.load(f"./weight/MobileNetV2_{epoch - 1}.pdparams"))
    with paddle.no_grad():
        for img, label in test_dataloader:
            label = label.astype("int64")
            output = model(img)
            loss = loss_func(output, label)
            right_num += paddle.sum(paddle.argmax(output, axis=1) == label).item()
            total_num += output.shape[0]
            loss_total += loss.item()
    logger.info(f"Best Epoch Test Loss:{loss_total / len(test_dataloader)}\tAccuracy:{right_num / total_num}")


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
    paddle.seed(10)
    np.random.seed(10)
    random.seed(10)
    paddle.disable_static()
    paddle.set_device(args.device)

    logger.info(f"Batch Size:{args.batch_size}\tLearn Rate:{args.learn_rate}\t"
                f"Optimizer:{args.optimizer}\tData Augmentation:{args.data_augmentation}\t"
                f"Weight Decay:{args.weight_decay}\t"
                f"Total Epoch:{args.epoch}\tStart Epoch:{args.start_epoch}")
    model = MobileNetV2(200)
    # model_train(args.epoch, args.start_epoch, args.learn_rate, args.weight_decay, args.batch_size, model,
    #             args.optimizer, Transforms_train if args.data_augmentation else Transforms_test, Transforms_test,
    #             args.data_path, logger)
    model_test(args.epoch, args.batch_size, model, Transforms_test, args.data_path,logger)


if __name__ == "__main__":
    main()

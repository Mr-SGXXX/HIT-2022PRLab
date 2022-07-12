import argparse
import logging
import os
import math
import random

import imageio
import numpy as np
import torch
from torch import autograd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from model import *


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='GAN', help="选择要使用的模型：GAN/WGAN/WGAN-GP")
    parser.add_argument('-bs', '--batch_size', type=int, default=128, help="设置Batch Size")
    parser.add_argument('-dp', '--data_path', default='data/points.mat', help="数据集存放位置")
    parser.add_argument('-e', '--epoch', type=int, default=100, help="迭代轮数")
    parser.add_argument('-se', '--start_epoch', type=int, default=0, help="初始迭代轮数")
    parser.add_argument('-lp', '--log_path', default="log/GAN_RMSProp.log", help="日志路径")
    parser.add_argument('-d', '--device', default="cuda:0" if torch.cuda.is_available() else "cpu", help="使用的计算设备")
    parser.add_argument('-wn', '--weight_name', default="GAN_RMSProp", help="权重名称")
    parser.add_argument('-lr', '--learn_rate', type=float, default=5e-5, help="学习率")
    parser.add_argument('-k', '--D_iter_num', type=int, default=5, help="内层判别器优化次数")
    parser.add_argument('-ns', '--noise_size', type=int, default=6, help="噪声尺寸")
    parser.add_argument('-op', '--optimizer', default="RMSProp", help="优化器类型")
    parser.add_argument('-c', '--clip', type=float, default=0.01, help="WGAN中的裁剪参数")
    parser.add_argument('-wp', '--weight_penalty', type=float, default=10, help="WGAN-GP中的裁剪参数")
    return parser.parse_args()


def model_train(epoch, start_epoch, batch_size, noise_size, D_iter_num, model_type, model_D, model_G, device, data,
                clip, weight_penalty, learn_rate, optimizer_type, weight_name, logger):
    if os.path.exists(f"./weight/{weight_name}_G_{start_epoch - 1}.pth") and \
            os.path.exists(f"./weight/{weight_name}_D_{start_epoch - 1}.pth"):
        model_G.load_state_dict(torch.load(f"./weight/{weight_name}_G_{start_epoch - 1}.pth"))
        model_D.load_state_dict(torch.load(f"./weight/{weight_name}_D_{start_epoch - 1}.pth"))
        logger.info(f"Load Last Epoch({start_epoch - 1}) Success")
    else:
        logger.info(f"Train From Beginning")
        if not os.path.exists("./weight/"):
            os.mkdir("weight")
    model_D.train().to(device)
    model_G.train().to(device)
    if optimizer_type == "Adam":
        # optimizer_G = torch.optim.Adam(model_G.parameters(), betas=(0, 0.9), lr=learn_rate)
        optimizer_G = torch.optim.Adam(model_G.parameters(), lr=learn_rate)
        optimizer_D = torch.optim.Adam(model_D.parameters(), lr=learn_rate)
    elif optimizer_type == "RMSProp":
        optimizer_G = torch.optim.RMSprop(model_G.parameters(), lr=learn_rate)
        optimizer_D = torch.optim.RMSprop(model_D.parameters(), lr=learn_rate)
    elif optimizer_type == "SGD":
        optimizer_G = torch.optim.SGD(model_G.parameters(), lr=learn_rate)
        optimizer_D = torch.optim.SGD(model_D.parameters(), lr=learn_rate)
    elif optimizer_type == "SGD-M":
        optimizer_G = torch.optim.SGD(model_G.parameters(), momentum=0.9, lr=learn_rate)
        optimizer_D = torch.optim.SGD(model_D.parameters(), momentum=0.9, lr=learn_rate)
    else:
        logger.info("Wrong Optimizer Type")
        raise RuntimeError("优化器类型错误")
    for cur_epoch in range(start_epoch, epoch):
        D_loss = 0
        G_loss = 0
        for i in range(math.ceil(data.shape[0] / batch_size)):
            true_data = torch.from_numpy(
                data[i * batch_size:
                     (i + 1) * batch_size if (i + 1) *
                                             batch_size < data.shape[0] else data.shape[0]]).to(device).to(
                torch.float32)
            fake_data = model_G(torch.randn(batch_size, noise_size).to(device))
            # 优化判别器
            for j in range(D_iter_num):
                p_t = model_D(true_data)
                p_f = model_D(fake_data)
                if model_type == "GAN":
                    D_loss = -torch.mean(torch.log(p_t) + torch.log(1 - p_f))
                elif model_type == "WGAN":
                    D_loss = torch.mean(p_f) - torch.mean(p_t)
                    for param in model_D.parameters():
                        torch.clip(param, min=-clip, max=clip)
                elif model_type == "WGAN-GP":
                    eps = torch.rand(true_data.size()).to(device)
                    x = eps * true_data + (1 - eps) * fake_data
                    D_loss = torch.mean(p_f) - torch.mean(p_t) + weight_penalty * cal_penalty(x, model_D, device)
                else:
                    logger.info("Wrong Model Type")
                    raise RuntimeError("模型类型错误")
                optimizer_D.zero_grad()
                D_loss.backward(retain_graph=True)
                optimizer_D.step()
            p_f = model_D(model_G(torch.randn(batch_size, noise_size).to(device)))
            if model_type == "GAN":
                G_loss = -torch.mean(torch.log(p_f))
            elif model_type == "WGAN":
                G_loss = -torch.mean(p_f)
            elif model_type == "WGAN-GP":
                G_loss = -torch.mean(p_f)
            else:
                logger.info("Wrong Model Type")
                raise RuntimeError("模型类型错误")
            optimizer_G.zero_grad()
            G_loss.backward()
            optimizer_G.step()
        torch.save(model_D.state_dict(), f"./weight/{weight_name}_D_{cur_epoch}.pth")
        torch.save(model_G.state_dict(), f"./weight/{weight_name}_G_{cur_epoch}.pth")
        if os.path.exists(f"./weight/{weight_name}_D_{cur_epoch - 1}.pth") \
                and os.path.exists(f"./weight/{weight_name}_D_{cur_epoch - 1}.pth"):
            os.remove(f"./weight/{weight_name}_G_{cur_epoch - 1}.pth")
            os.remove(f"./weight/{weight_name}_D_{cur_epoch - 1}.pth")
        logger.info(f"Epoch {cur_epoch + 1}/{epoch}\tLoss_G: {G_loss}\tLoss_D:{D_loss}")
        model_test(cur_epoch, noise_size, weight_name, model_G, device, data)
    logger.info(f"Result Image Saved in ./image")


def cal_penalty(x, model_D, device):
    p_x = model_D(x)
    place_holder = autograd.Variable(torch.FloatTensor(x.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
    grads = autograd.grad(p_x, x, grad_outputs=place_holder)[0]
    grads = grads.view(x.size(0), -1)
    return torch.mean((torch.norm(grads, dim=1) - 1) ** 2)


def model_test(cur_epoch, noise_size, weight_name, model_G, device, data):
    if not os.path.exists(f"./image/{weight_name}"):
        os.makedirs(f"./image/{weight_name}")
    model_G.eval().to(device)
    gen_data = model_G(torch.randn(data.shape[0], noise_size).to(device))
    gen_data = np.array(gen_data.cpu().data)
    plt.clf()
    plt.plot(gen_data[:, 0], gen_data[:, 1], 'r.', markersize='1', label="Generated Data")
    plt.plot(data[:, 0], data[:, 1], 'b.', markersize='1', label="True Data")
    plt.legend()
    plt.title(f"Epoch:{cur_epoch + 1}")
    plt.savefig(f'image/{weight_name}/{cur_epoch + 1}.jpg')


def main():
    args = parse_arg()

    # 设置日志
    if not os.path.exists(os.path.dirname(args.log_path)):
        os.mkdir(os.path.dirname(args.log_path))
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(args.log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s]-[%(levelname)s]: %(message)s')
    handler.setFormatter(formatter)
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

    mat = loadmat(args.data_path)
    data = mat['xx']
    np.random.shuffle(data)

    if args.model == "GAN":
        model_D = GAN_D(data.shape[1])
        model_G = GAN_G(args.noise_size, data.shape[1])
    elif args.model == "WGAN":
        model_D = WGAN_D(data.shape[1])
        model_G = WGAN_G(args.noise_size, data.shape[1])
    elif args.model == "WGAN-GP":
        model_D = WGAN_D(data.shape[1])
        model_G = WGAN_G(args.noise_size, data.shape[1])
    else:
        logger.info("Wrong Model Type")
        raise RuntimeError("模型类型错误")

    logger.info(f"Model:{args.model}\tBatch Size:{args.batch_size}\t"
                f"Total Epoch:{args.epoch}\tStart Epoch:{args.start_epoch}")
    model_train(args.epoch, args.start_epoch, args.batch_size, args.noise_size, args.D_iter_num, args.model, model_D,
                model_G, args.device, data, args.clip, args.weight_penalty, args.learn_rate, args.optimizer,
                args.weight_name, logger)
    with imageio.get_writer(uri=f'./image/result_{args.weight_name}.gif', mode='I', fps=1) as writer:
        for cur_dir, dirs, files in os.walk(f"./image/{args.weight_name}"):
            files.sort(key=lambda info: (int(info[0:-4]), info[-4:]))
            for file in files:
                writer.append_data(imageio.imread(os.path.join(cur_dir, file)))


if __name__ == "__main__":
    main()

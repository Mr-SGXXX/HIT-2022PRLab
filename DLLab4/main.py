import logging
import argparse
import os
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import *
from embedding import load_embedding
from train import *
from dataset import Shopping, Climate
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, median_absolute_error, recall_score, f1_score

cat_list = ["书籍", "平板", "手机", "水果", "洗发水", "热水器", "蒙牛", "衣服", "计算机", "酒店"]

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', default='shopping', help="选择要处理的任务：shopping/climate")
parser.add_argument('-m', '--model', default='BiLSTM', help="选择要使用的模型：RNN/GRU/LSTM/BiLSTM")
parser.add_argument('-bs', '--batch_size', type=int, default=1024, help="设置Batch Size")
parser.add_argument('-dp', '--data_path', default='data/online_shopping_10_cats.csv', help="数据集存放位置")
# parser.add_argument('-dp', '--data_path', default='data/jena_climate_2009_2016.csv', help="数据集存放位置")
parser.add_argument('-e', '--epoch', type=int, default=100, help="迭代轮数")
parser.add_argument('-se', '--start_epoch', type=int, default=0, help="初始迭代轮数")
parser.add_argument('-lp', '--log_path', default="log/Shopping_BiLSTM.log", help="日志路径")
parser.add_argument('-d', '--device', default="cuda:0" if torch.cuda.is_available() else "cpu", help="使用的计算设备")
parser.add_argument('-wn', '--weight_name', default="Shopping_BiLSTM", help="权重名称")
parser.add_argument('-lr', '--learn_rate', type=float, default=1e-3, help="学习率")


def shopping_train(epoch, start_epoch, learn_rate, batch_size, device, model, weight_name, data_path, logger, embedding,
                   word2idx):
    if os.path.exists(f"./weight/{weight_name}_{start_epoch - 1}.pth"):
        model.load_state_dict(torch.load(f"./weight/{weight_name}_{start_epoch - 1}.pth"))
        logger.info(f"Load Last Epoch({start_epoch - 1}) Success")
    else:
        logger.info(f"Train From Beginning")
        if not os.path.exists("./weight/"):
            os.mkdir("weight")
    train_dataset = Shopping(data_path, 0, word2idx, cat_list)
    train_dataloader = DataLoader(train_dataset, batch_size, True)
    eval_dataset = Shopping(data_path, 1, word2idx, cat_list)
    eval_dataloader = DataLoader(eval_dataset, batch_size, False)

    emb = nn.Embedding(embedding.weight.size(0), embedding.weight.size(1), padding_idx=0)
    emb.weight = embedding.weight
    emb.weight.requires_grad = False
    emb.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    loss_func = nn.CrossEntropyLoss()
    best_acc = 0
    best_epoch = 0
    for cur_epoch in range(start_epoch, epoch):
        loss = train_one_epoch_shopping(model, train_dataloader, optimizer, loss_func, device, emb)
        torch.save(model.state_dict(), f"./weight/{weight_name}_{cur_epoch}.pth")
        if os.path.exists(f"./weight/{weight_name}_{cur_epoch - 1}.pth"):
            os.remove(f"./weight/{weight_name}_{cur_epoch - 1}.pth")
        logger.info(f"Epoch: {cur_epoch + 1}/{epoch}\tTrain Loss:{loss}")
        loss, acc = eval_one_epoch_shopping(model, eval_dataloader, loss_func, device, emb)
        if acc > best_acc:
            best_acc = acc
            best_epoch = cur_epoch
            torch.save(model.state_dict(), f"./weight/{weight_name}_best.pth")
        logger.info(f"Eval Loss:{loss}\tAccuracy:{acc}\tBest Accuracy/Epoch:{best_acc}/{best_epoch + 1}")
    logger.info(f"Train Phase Over\tBest Accuracy/Epoch:{best_acc}/{best_epoch + 1}")


def shopping_test(epoch, batch_size, device, model, weight_name, data_path, logger, embedding, word2idx):
    model.to(device)
    model.eval()
    test_dataset = Shopping(data_path, 2, word2idx, cat_list)
    test_dataloader = DataLoader(test_dataset, batch_size, False)
    loss_func = nn.CrossEntropyLoss()

    emb = nn.Embedding(embedding.weight.size(0), embedding.weight.size(1), padding_idx=0)
    emb.weight = embedding.weight
    emb.weight.requires_grad = False
    emb.to(device)

    model.load_state_dict(torch.load(f"./weight/{weight_name}_best.pth"))
    loss_total = 0.0
    right_num = 0
    total_num = 0
    dst_list = []
    output_best_list = []
    with torch.no_grad():
        for word_list, label in test_dataloader:
            dst_list.append(label)
            word_list = word_list.to(device)
            word_list = emb(word_list)
            label = label.to(device)
            output = model(word_list)
            output_best_list.append(torch.argmax(output, dim=1).to('cpu'))
            loss = loss_func(output, label)
            right_num += (torch.argmax(output, dim=1) == label).sum().item()
            total_num += output.size(0)
            loss_total += loss
        r = recall_score(torch.cat(dst_list, dim=0), torch.cat(output_best_list, dim=0), average='macro')
        f1 = f1_score(torch.cat(dst_list, dim=0), torch.cat(output_best_list, dim=0), average='macro')
        logger.info(f"Best Epoch Test Loss:{loss_total / len(test_dataloader)}\tAccuracy:{right_num / total_num}\tRecall:{r}\tF1:{f1}")

    model.load_state_dict(torch.load(f"./weight/{weight_name}_{epoch - 1}.pth"))
    loss_total = 0.0
    right_num = 0
    total_num = 0
    output_best_list = []
    with torch.no_grad():
        for word_list, label in test_dataloader:
            word_list = word_list.to(device)
            word_list = emb(word_list)
            label = label.to(device)
            output = model(word_list)
            output_best_list.append(torch.argmax(output, dim=1).to('cpu'))
            loss = loss_func(output, label)
            right_num += (torch.argmax(output, dim=1) == label).sum().item()
            total_num += output.size(0)
            loss_total += loss
        r = recall_score(torch.cat(dst_list, dim=0), torch.cat(output_best_list, dim=0), average='macro')
        f1 = f1_score(torch.cat(dst_list, dim=0), torch.cat(output_best_list, dim=0), average='macro')
        logger.info(f"Last Epoch Test Loss:{loss_total / len(test_dataloader)}\tAccuracy:{right_num / total_num}\tRecall:{r}\tF1:{f1}")


def climate_train(epoch, start_epoch, learn_rate, batch_size, device, model, weight_name, data_path, logger):
    if os.path.exists(f"./weight/{weight_name}_{start_epoch - 1}.pth"):
        model.load_state_dict(torch.load(f"./weight/{weight_name}_{start_epoch - 1}.pth"))
        logger.info(f"Load Last Epoch({start_epoch - 1}) Success")
    else:
        logger.info(f"Train From Beginning")
        if not os.path.exists("./weight/"):
            os.mkdir("weight")

    train_dataset = Climate(data_path, 0)
    train_dataloader = DataLoader(train_dataset, batch_size, True)

    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    loss_func = nn.L1Loss()
    best_loss = 10000
    best_epoch = 0
    for cur_epoch in range(start_epoch, epoch):
        loss = train_one_epoch_climate(model, train_dataloader, optimizer, loss_func, device)
        torch.save(model.state_dict(), f"./weight/{weight_name}_{cur_epoch}.pth")
        if os.path.exists(f"./weight/{weight_name}_{cur_epoch - 1}.pth"):
            os.remove(f"./weight/{weight_name}_{cur_epoch - 1}.pth")
        if loss < best_loss:
            best_loss = loss
            best_epoch = cur_epoch
            torch.save(model.state_dict(), f"./weight/{weight_name}_best.pth")
        logger.info(f"Epoch: {cur_epoch + 1}/{epoch}\tTrain Loss:{loss}\tBest Loss/Epoch:{best_loss}/{best_epoch + 1}")
    logger.info(f"Train Phase Over\tBest Loss/Epoch:{best_loss}/{best_epoch + 1}")


def climate_test(epoch, batch_size, device, model, weight_name, data_path, logger):
    model.to(device)
    model.eval()
    test_dataset = Climate(data_path, 1)
    test_dataloader = DataLoader(test_dataset, batch_size, False)
    loss_func = nn.L1Loss()
    example_dst = None
    example_output_best = None
    example_output_last = None

    if not os.path.exists("./image"):
        os.mkdir("image")

    model.load_state_dict(torch.load(f"./weight/{weight_name}_best.pth"))
    loss_total = 0.0
    dst_list = []
    output_best_list = []
    with torch.no_grad():
        for src_data, dst_data in test_dataloader:
            dst_list.append(dst_data)
            src_data = src_data.to(device)
            dst_data = dst_data.to(device)
            output = model(src_data)
            output_best_list.append(output.to('cpu'))
            loss = loss_func(output, dst_data)
            loss_total += loss
            example_dst = dst_data
            example_output_best = output
        mean_error = mean_absolute_error(torch.cat(dst_list, dim=0), torch.cat(output_best_list, dim=0))
        median_error = median_absolute_error(torch.cat(dst_list, dim=0), torch.cat(output_best_list, dim=0))
        logger.info(
            f"Best Epoch Test Loss:{loss_total / len(test_dataloader)}\tMean Error:{mean_error}\tMedian Error:{median_error}")

    model.load_state_dict(torch.load(f"./weight/{weight_name}_{epoch - 1}.pth"))
    loss_total = 0.0
    output_last_list = []
    with torch.no_grad():
        for src_data, dst_data in test_dataloader:
            src_data = src_data.to(device)
            dst_data = dst_data.to(device)
            output = model(src_data)
            output_last_list.append(output.to('cpu'))
            loss = loss_func(output, dst_data)
            loss_total += loss
            example_output_last = output
        mean_error = mean_absolute_error(torch.cat(dst_list, dim=0), torch.cat(output_last_list, dim=0))
        median_error = median_absolute_error(torch.cat(dst_list, dim=0), torch.cat(output_last_list, dim=0))
        logger.info(
            f"Last Epoch Test Loss:{loss_total / len(test_dataloader)}\tMean Error:{mean_error}\tMedian Error:{median_error}")
    for i in range(example_dst.size(0)):
        plt.clf()
        plt.plot(example_dst.to('cpu')[i], 'k.-', label="Ground Truth")
        plt.plot(example_output_best.to('cpu')[i], 'r.-', label="Best Epoch Result")
        plt.plot(example_output_last.to('cpu')[i], 'b.-', label="Last Epoch Result")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Predicted Temperature")
        plt.title("Temperature Predict Result")
        plt.savefig(f'image/predict_result_{i}.jpg')


def main():
    args = parser.parse_args()
    if not os.path.exists(os.path.dirname(args.log_path)):
        os.mkdir(os.path.dirname(args.log_path))
    # 设置日志
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

    logger.info(f"Task:{args.task}\tModel:{args.model}\tBatch Size:{args.batch_size}\t"
                f"Total Epoch:{args.epoch}\tStart Epoch:{args.start_epoch}")
    if args.task == "shopping" and args.model == 'RNN':
        model = RNN(100, 256).to(args.device)
    elif args.task == "shopping" and args.model == 'GRU':
        model = GRU(100, 256).to(args.device)
    elif args.task == "shopping" and args.model == 'LSTM':
        model = LSTM(100, 256).to(args.device)
    elif args.task == "shopping" and args.model == 'BiLSTM':
        model = BiLSTM(100, 256).to(args.device)
    elif args.task == "climate" and args.model == 'GRU':
        model = RegGRU(2, 256).to(args.device)
    else:
        logger.info("Model Choice Error")
        raise RuntimeError("模型选项错误")

    if args.task == 'shopping':
        embedding, word2idx = load_embedding(args.data_path, logger=logger)
        # shopping_train(args.epoch, args.start_epoch, args.learn_rate, args.batch_size, args.device,
        #                model, args.weight_name, args.data_path, logger, embedding, word2idx)
        shopping_test(args.epoch, args.batch_size, args.device, model, args.weight_name, args.data_path, logger,
                      embedding, word2idx)
    elif args.task == 'climate':
        climate_train(args.epoch, args.start_epoch, args.learn_rate, args.batch_size, args.device,
                      model, args.weight_name, args.data_path, logger)
        climate_test(args.epoch, args.batch_size, args.device, model, args.weight_name, args.data_path, logger)
    else:
        logger.info("Task Choice Error")
        raise RuntimeError("任务选项错误")


if __name__ == "__main__":
    main()

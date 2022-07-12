# 用于评论分类及气温预测的数据集
import csv
import jieba
import numpy as np
import torch
from torch.utils.data import Dataset
import time


class Shopping(Dataset):
    def __init__(self, path, type, word2idx=None, cat_list=None, max_len=250):
        """
        根据评论将物品分为书籍、平板等十个类
        :param path:评论数据集路径
        :param type:0表示训练集 1表示验证集 2表示测试集
        :param word2idx:评论词-序号映射表，等于None时返回原词
        :param cat_list:类别列表用于映射成索引值，等于None时返回原词
        :param max_len:评论最大词数
        """
        super(Shopping, self).__init__()
        self.datas = []
        self.labels = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                words = jieba.lcut(row['review'])[0:max_len]
                while len(words) < max_len:
                    words.append("unk")
                if word2idx is not None:
                    words = [word2idx[word] if word in word2idx else word2idx["unk"] for word in words]
                label = row['cat'] if cat_list is None else cat_list.index(row['cat'])
                if type == 0 and 1 <= i % 5 <= 3:
                    self.datas.append(words)
                    self.labels.append(label)
                elif i % 5 == 4 and type == 1:
                    self.datas.append(words)
                    self.labels.append(label)
                elif i % 5 == 0 and type == 2:
                    self.datas.append(words)
                    self.labels.append(label)

    def __getitem__(self, item):
        return torch.tensor(self.datas[item]), torch.tensor(self.labels[item])

    def __len__(self):
        return len(self.datas)


class Climate(Dataset):
    def __init__(self, path, type):
        """
        根据前五天气候数据预测后两天温度数据
        :param path:温度数据集路径
        :param type:0表示训练集 1表示测试集
        """
        assert type == 1 or type == 0, "数据集类型有误"
        super(Climate, self).__init__()
        self.src_data_list = []
        self.dst_data_list = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            src_data = []
            dst_data = []
            for i, row in enumerate(reader):
                if (i // 144) % 7 != 5 and (i // 144) % 7 != 6:
                    # 特征全用
                    # temp_list = [float(row['p (mbar)']), float(row['T (degC)']), float(row['Tpot (K)']),
                    #              float(row['Tdew (degC)']), float(row['rh (%)']), float(row['VPmax (mbar)']),
                    #              float(row['VPact (mbar)']), float(row['VPdef (mbar)']), float(row['sh (g/kg)']),
                    #              float(row['H2OC (mmol/mol)']), float(row['rho (g/m**3)']), float(row['wv (m/s)']),
                    #              float(row['max. wv (m/s)']), float(row['wd (deg)'])]
                    # 用各种温度
                    temp_list = [float(float(row['T (degC)'])), float(row['Tdew (degC)'])]
                    # 只用没特殊说明的温度
                    # temp_list = [float(row['T (degC)'])]
                    # 这个特征的选取参考 https://blog.csdn.net/caqjeryy/article/details/120100844
                    # month = float(row['Date Time'][3:5])
                    # hour = float(row['Date Time'][11:13])
                    # sinh = torch.sin(hour * (2 * torch.tensor(np.pi) / 24))
                    # cosh = torch.cos(hour * (2 * torch.tensor(np.pi) / 24))
                    # temp_list = [month, sinh, cosh, float(row['T (degC)']), float(row['p (mbar)']),
                    #              float(row['max. wv (m/s)'])]
                    src_data.append(temp_list)
                else:
                    dst_data.append(float(row['T (degC)']))

                if i % 1008 == 1007:
                    if i <= 368291 and type == 0:
                        self.src_data_list.append(torch.tensor(src_data))
                        self.dst_data_list.append(torch.tensor(dst_data))
                    elif i > 368291 and type == 1:
                        self.src_data_list.append(torch.tensor(src_data))
                        self.dst_data_list.append(torch.tensor(dst_data))
                    src_data = []
                    dst_data = []

    def __getitem__(self, item):
        return self.src_data_list[item], self.dst_data_list[item]

    def __len__(self):
        return len(self.dst_data_list)

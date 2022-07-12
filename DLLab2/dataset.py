from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import os
from PIL import Image


class Caltech101(Dataset):
    def __init__(self, data_dir, data_type, transformer=None):
        """

        :param data_dir: 数据集路径，结构为
            |- Caltech101
                |- 000
                    |- 1.jpg
                    ...
                    |- n.jpg
                ...
                |- 101
        :param data_type: 0表示训练集 1表示验证集 2表示测试集
        """
        super(Caltech101, self).__init__()
        self.labels = []
        self.image_paths = []
        self.transformer = transformer
        for cur_dir, dirs, files in os.walk(data_dir):
            label = os.path.basename(cur_dir)
            img_num = len(files)
            for i, file in enumerate(files):
                if data_type == 0 and i < img_num * 0.8:
                    self.image_paths.append(os.path.join(cur_dir, file))
                    self.labels.append(int(label))
                elif data_type == 1 and img_num * 0.8 <= i < img_num * 0.9:
                    self.image_paths.append(os.path.join(cur_dir, file))
                    self.labels.append(int(label))
                elif data_type == 2 and img_num * 0.9 <= i < img_num:
                    self.image_paths.append(os.path.join(cur_dir, file))
                    self.labels.append(int(label))

    def __getitem__(self, item):
        img = Image.open(self.image_paths[item]).convert("RGB")
        if self.transformer is not None:
            img = self.transformer(img)
        return img, self.labels[item]

    def __len__(self):
        return len(self.labels)

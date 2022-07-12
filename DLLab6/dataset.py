import os

import torchvision
from PIL import Image
from torch.utils.data import Dataset


class Flickr2K(Dataset):
    def __init__(self, path, type):
        """
        训练集：验证集：测试集 = 8：1：1
        :param path:数据集路径，结构为
            |- Flickr2K_HR
                |- ...
            |- Flickr2K_LR_bicubic
                |- ...
        :param type: 0表示训练集 1表示验证集 2表示测试集，对于训练集、验证集返回LR HR;对于测试集需返回LR HR 名称
        """
        super(Flickr2K, self).__init__()
        self.image_HR_paths = []
        self.image_LR_paths = []
        self.type = type
        self.name = []
        for cur_dir, dirs, files in os.walk(os.path.join(path, "Flickr2K_HR")):
            img_num = len(files)
            for i, file in enumerate(files):
                if i <= img_num * 0.8 and type == 0:
                    self.image_HR_paths.append(os.path.join(cur_dir, file))
                elif img_num * 0.8 < i <= img_num * 0.9 and type == 1:
                    self.image_HR_paths.append(os.path.join(cur_dir, file))
                elif img_num * 0.9 < i <= img_num and type == 2:
                    self.image_HR_paths.append(os.path.join(cur_dir, file))

        for cur_dir, dirs, files in os.walk(os.path.join(path, "Flickr2K_LR_bicubic/X4")):
            img_num = len(files)
            for i, file in enumerate(files):
                if i <= img_num * 0.8 and type == 0:
                    self.image_LR_paths.append(os.path.join(cur_dir, file))
                elif img_num * 0.8 < i <= img_num * 0.9 and type == 1:
                    self.image_LR_paths.append(os.path.join(cur_dir, file))
                elif img_num * 0.9 < i <= img_num and type == 2:
                    self.image_LR_paths.append(os.path.join(cur_dir, file))
                    self.name.append(file)

    def __getitem__(self, item):
        img_HR = Image.open(self.image_HR_paths[item]).convert("RGB")
        img_LR = Image.open(self.image_LR_paths[item]).convert("RGB")
        img_HR = torchvision.transforms.ToTensor()(img_HR)
        img_LR = torchvision.transforms.ToTensor()(img_LR)
        if self.type == 0 or self.type == 1:
            return img_LR, img_HR
        elif self.type == 2:
            return img_LR, img_HR, self.name[item]

    def __len__(self):
        return len(self.image_HR_paths)

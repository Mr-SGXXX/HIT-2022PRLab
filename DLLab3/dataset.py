import os

import torchvision.transforms
from PIL import Image
from torch.utils.data import Dataset


class PlantSeeding(Dataset):
    def __init__(self, path, type, cat_list=None, transforms=None):
        """
        验证集合层次采样取约10%
        :param path:数据集路径，结构为
            |- plant-seedlings-classification
                |- Black-grass
                    |- 0ace21089.png
                    ...
                    |- fef14b865.png
                ...
                |- Sugar beet
        :param type: 0表示训练集 1表示验证集 2表示测试集
        :param cat_list: 类别列表用于映射成索引值，等于None时返回原词
        :param transforms: 用于预处理的transforms
        """
        super(PlantSeeding, self).__init__()
        self.type = type
        self.labels = []
        self.image_paths = []
        self.transforms = transforms
        if type == 0 or type == 1:
            for cur_dir, dirs, files in os.walk(os.path.join(path, "train")):
                label = os.path.basename(cur_dir)
                for i, file in enumerate(files):
                    if type == 0 and i % 10 != 9:
                        self.image_paths.append(os.path.join(cur_dir, file))
                        self.labels.append(label if cat_list is None else cat_list.index(label))
                    elif type == 1 and i % 10 == 9:
                        self.image_paths.append(os.path.join(cur_dir, file))
                        self.labels.append(label if cat_list is None else cat_list.index(label))
        elif type == 2:
            for cur_dir, dirs, files in os.walk(os.path.join(path, "test")):
                for file in files:
                    self.image_paths.append(os.path.join(cur_dir, file))
        else:
            raise RuntimeError("数据集类型错误")

    def __getitem__(self, item):
        img = Image.open(self.image_paths[item]).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = torchvision.transforms.ToTensor()(img)
        if self.type == 0 or self.type == 1:
            return img, self.labels[item]
        elif self.type == 2:
            return img, os.path.basename(self.image_paths[item])

    def __len__(self):
        return len(self.image_paths)

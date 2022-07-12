import os.path

import numpy as np
from paddle.io import Dataset
from PIL import Image


class CUB(Dataset):
    def __init__(self, path, type, transforms=None):
        """
        训练集：验证集：测试集 = 8：1：1
        :param path:数据集路径，结构为
            |- CUB_200_2011
                |- Images
                    |- 001.Black_footed_Albatross
                        |- Black_Footed_Albatross_0001_796111.jpg
                        ...
                        |- Black_Footed_Albatross_0090_796077.jpg
                    ...
                    |- 200.Common_Yellowthroat
        :param type: 0表示训练集 1表示验证集 2表示测试集
        """
        super(CUB, self).__init__()
        self.image_paths = []
        self.type = type
        self.labels = []
        self.transforms = transforms
        if type == 0:
            for cur_dir, dirs, files in os.walk(os.path.join(path, "images")):
                img_num = len(files)
                try:
                    label = int(os.path.basename(cur_dir)[0:3]) - 1
                except:
                    pass
                for i, file in enumerate(files):
                    if i <= img_num * 0.8:
                        self.image_paths.append(os.path.join(cur_dir, file))
                        self.labels.append(label)

        elif type == 1:
            for cur_dir, dirs, files in os.walk(os.path.join(path, "images")):
                img_num = len(files)
                try:
                    label = int(os.path.basename(cur_dir)[0:3]) - 1
                except:
                    pass
                for i, file in enumerate(files):
                    if img_num * 0.8 < i <= img_num * 0.9:
                        self.image_paths.append(os.path.join(cur_dir, file))
                        self.labels.append(label)

        elif type == 2:
            for cur_dir, dirs, files in os.walk(os.path.join(path, "images")):
                img_num = len(files)
                try:
                    label = int(os.path.basename(cur_dir)[0:3]) - 1
                except:
                    pass
                for i, file in enumerate(files):
                    if img_num * 0.9 < i <= img_num:
                        self.image_paths.append(os.path.join(cur_dir, file))
                        self.labels.append(label)

    def __getitem__(self, item):
        img = Image.open(self.image_paths[item]).convert("RGB")
        img = np.asarray(img, dtype='float32')
        if self.transforms is not None:
            img = self.transforms(img)
        img = np.transpose(img, (2, 0, 1))
        return img, self.labels[item]

    def __len__(self):
        return len(self.image_paths)

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Optional, Callable
from PIL import Image
import torch
import pickle


class Dataset_SNN(Dataset):
    """自定义数据集类"""

    def __init__(self,normal):

        self.normal = normal
        self.samples = self._load_samples()

    def _exponential_decay(self,x, a = 3.247, b = 0.065, c = 0.019):
        return a * np.exp(-b * x) + c

    def _load_samples(self):
        """加载数据样本列表"""
        with open(r"C:\Users\LH\Desktop\SNN_MY\data_snn_my\data_all.pkl","rb") as f:
            data_all = pickle.load(f)
            # print(data_all[-1])
        return data_all

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        label,img = self.samples[idx]
        if(self.normal):
            img = self._exponential_decay(img)
        else:
            pass

        return torch.tensor(img,dtype=torch.float32),torch.tensor(label)


class Dataset_SNN_test(Dataset):
    """自定义数据集类"""

    def __init__(self,normal):

        self.normal = normal
        self.samples = self._load_samples()

    def _exponential_decay(self,x, a = 3.247, b = 0.065, c = 0.019):
        return a * np.exp(-b * x) + c

    def _load_samples(self):
        """加载数据样本列表"""
        test_each = 0.3
        test1 = int(149 * test_each)
        test2 = int(200 * test_each)
        test3 = int(295 * test_each)
        test4 = int(200 * test_each)
        test5 = int(313 * test_each)
        test6 = int(162 * test_each)
        test7 = int(295 * test_each)
        with open(r"C:\Users\LH\Desktop\SNN_MY\data_snn_my\data_all.pkl","rb") as f:
            data_all = pickle.load(f)
        data_test = (data_all[0:test1] + data_all[149: 149 + test2] + data_all[349: 349+test3] + data_all[644:644+test4]
                     + data_all[844:844+test5] + data_all[1157:1157+test6] + data_all[1319:1319+test7])
        return data_test

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        label,img = self.samples[idx]
        if(self.normal):
            img = self._exponential_decay(img)
        else:
            pass

        return torch.tensor(img,dtype=torch.float32),torch.tensor(label)



class Dataset_SNN_train(Dataset):
    """自定义数据集类"""

    def __init__(self,normal):

        self.normal = normal
        self.samples = self._load_samples()

    def _exponential_decay(self,x, a = 3.247, b = 0.065, c = 0.019):
        return a * np.exp(-b * x) + c

    def _load_samples(self):
        """加载数据样本列表"""
        test_each = 0.3
        test1 = int(149 * test_each)
        test2 = int(200 * test_each)
        test3 = int(295 * test_each)
        test4 = int(200 * test_each)
        test5 = int(313 * test_each)
        test6 = int(162 * test_each)
        test7 = int(295 * test_each)
        with open(r"C:\Users\LH\Desktop\SNN_MY\data_snn_my\data_all.pkl","rb") as f:
            data_all = pickle.load(f)
        data_test = (data_all[test1:149] + data_all[149 +test2: 349] + data_all[349 + test3: 644] + data_all[644 +test4 :844]
                     + data_all[844 + test5 :1157] + data_all[1157 + test6:1319] + data_all[1319 + test7:])
        return data_test

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        label,img = self.samples[idx]
        if(self.normal):
            img = self._exponential_decay(img)
        else:
            pass

        return torch.tensor(img,dtype=torch.float32),torch.tensor(label)






if __name__ == '__main__':
    dataset_train = Dataset_SNN_train(normal=True)
    dataset_test = Dataset_SNN_test(normal=True)

    print(len(dataset_train))
    print(len(dataset_test))
import scipy
import scipy.io
import os
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score, accuracy_score
from torch.utils.data import Dataset
import h5py
from collections import defaultdict
import numpy as np


def file_reading(filename):
    try:
        with h5py.File(filename, 'r') as f:
            if 'RGB_1d' in f.keys():
                data = f['RGB_1d'][()]
                return data
            # if 'emis_29_1d' in f.keys() and 'emis_31_1d' in f.keys() and 'RGB_1d' in f.keys():
            #     data1 = f['emis_29_1d'][()]
            #     data2 = f['emis_31_1d'][()]
            #     data0 = f['RGB_1d'][()]
            #     data = np.zeros((5, 128, 128))
            #     data[:3, :, :] = data0
            #     data[3, :, :] = data1
            #     data[4, :, :] = data2
            #     return data
    except OSError as e:
        print(f"{filename}文件无法打开。")
    return None

def file_reading_spicy(filename):
    try:
        data = scipy.io.loadmat(filename)
        data = data['RGB_1d']
        return data

    except OSError as e:
        print(f"{filename}文件无法打开。")
    return None

def file_reading_new(filename):
    try:
        with h5py.File(filename, 'r') as f:
            if 'refl_01_CB' in f.keys():
                data1 = f['refl_01_CB'][()]
            if 'refl_03_CB' in f.keys():
                data3 = f['refl_03_CB'][()]
            if 'refl_04_CB' in f.keys():
                data2 = f['refl_04_CB'][()]

            if len(np.shape(data1)) == 3:
                scenes = data1.shape[0]
            elif len(np.shape(data1)) == 2:
                scenes = 1
            else:
                print("Invalid shape!")

            data = np.zeros((scenes, 3, 128, 128))
            if scenes == 1:
                data[0, 0, :, :] = data1[:, :]
                data[0, 1, :, :] = data2[:, :]
                data[0, 2, :, :] = data3[:, :]
            else:
                for i in range(scenes):
                    data[i, 0, :, :] = data1[i, :, :]
                    data[i, 1, :, :] = data2[i, :, :]
                    data[i, 2, :, :] = data3[i, :, :]
            return data ,scenes
    except OSError:
        print(f"{filename}文件无法打开。")
    return None


# 测试准确率
def accuracy(output, target):
    y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    target = target.cpu()

    return accuracy_score(target, y_pred)


# 计算f1
def calculate_f1_macro(output, target):
    y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    target = target.cpu()

    return f1_score(target, y_pred, average='macro')


# 计算recall
def calculate_recall_macro(output, target):
    y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    target = target.cpu()
    # tp fn fp
    return recall_score(target, y_pred, average="macro", zero_division=0)


# 读取云数据
class CloudDataset(Dataset):
    def __init__(self, data_path, transform=None):
        label_map = {'cat0_Closed_MCC': 0, 'cat1_Clustered_Cu': 1, 'cat2_Disorganized MCC': 2, 'cat3_Open_MCC': 3,
                     'cat4_Solid_stratus': 4, 'cat5_Suppressed_Cu': 5}
        self.transform = transform
        self.data = []
        for folder in os.listdir(data_path):
            folder_path = os.path.join(data_path, folder)
            if not os.path.isdir(folder_path):
                continue
            for file_name in os.listdir(folder_path):
                if not file_name.endswith('.mat'):
                    continue
                file_path = os.path.join(folder_path, file_name)
                mat_data = file_reading(file_path)
                if mat_data is not None:
                    data = torch.from_numpy(mat_data)
                    label = label_map[folder]
                    label = torch.tensor(label)
                    self.data.append((data, label))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, label = self.data[idx]
        if self.transform:
            data = self.transform(data)
        return data, label


# 读取待判断的数据
class JudgeDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform
        self.data = []
        for file_name in os.listdir(data_path):
            if not file_name.endswith('.mat'):
                print("some files cant be judged")
                continue
            file_path = os.path.join(data_path, file_name)
            # 根据文件的不同修改不同的读取函数，这个带new的是读多场景的
            mat_data = file_reading(file_path)
            if mat_data is not None:
                data = torch.from_numpy(mat_data)
                # data = np.transpose(data, (2, 0, 1))
                self.data.append((data, file_name))
            else:
                print("some files cant be judged")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, file_name = self.data[idx]
        if self.transform:
            data = self.transform(data)
        return data, file_name




# 读取待判断的数据
class JudgeDataset_new(Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform
        self.data = []
        for file_name in os.listdir(data_path):
            if not file_name.endswith('.mat'):
                print("some files cant be judged")
                continue
            file_path = os.path.join(data_path, file_name)
            mat_data,scenes = file_reading_new(file_path)
            if mat_data is not None:
                data = torch.from_numpy(mat_data)
                self.data.append((data, file_name,scenes))
            else:
                print("some files cant be judged")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, file_name,scenes = self.data[idx]
        if self.transform:
            data = self.transform(data)
        return data, file_name,scenes




# 定义一系列云分类器
class simple_CloudClassifier(nn.Module):
    def __init__(self):
        super(simple_CloudClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = x.view(-1, 64 * 32 * 32)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class VGG(nn.Module):
    def __init__(self, num_of_classes=1000):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_of_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MetricManager:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"],
                    float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )

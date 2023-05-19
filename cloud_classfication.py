import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import scipy
import h5py
import matplotlib
import mytorchutils


if __name__ == '__main__':

    # 初始化设备、数据、模型、损失函数和优化器
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # data_path = 'E:/workspace/code/test_data'
    data_path = '../../../../work10/zhangyu/MCC_classification_test_data/data/2014/0103'
    # data_path = './test_data'
    dataset = mytorchutils.CloudDataset(data_path)
    total_len = mytorchutils.CloudDataset.__len__(dataset)
    print(total_len)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = mytorchutils.CloudClassifier().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # 模型结构
    print(model)

    # 训练模型
    mean_loss_list = []
    all_single_loss_list = []
    for epoch in range(30):
        running_loss = 0.0
        single_loss_list = []
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.to(torch.float32)
            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            single_loss_list.append(loss.item())
            all_single_loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
        mytorchutils.draw(single_loss_list,"./img/single_loss"+str(epoch)+".png",'iteration','Training Loss',
                          'Training Loss Curve')
        mean_loss_list.append(running_loss/total_len)



    print("training is over")
    # 绘制loss曲线
    mytorchutils.draw(mean_loss_list, "./img/mean_loss.png", 'epoch', 'Training Loss',
                      'Training Loss Curve')

    mytorchutils.draw(all_single_loss_list, "./img/all_single_loss.png", 'iteration', 'Training Loss',
                      'Training Loss Curve')

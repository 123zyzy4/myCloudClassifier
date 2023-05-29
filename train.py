import timm

import torchvision

import os
from torch import nn, optim
import mytorchutils
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from datetime import datetime

# 超参数设置
params = {
    'train_path': "../../work10/zhangyu/MCC_classification_test_data/data_split3/train",
    'val_path': "../../work10/zhangyu/MCC_classification_test_data/data_split3/val",
    # 'train_path': "./test_data_split/train",
    # 'val_path': "./test_data_split/val",
    'lr': 1e-4,  # 学习率
    'batch_size': 256,  # 批次大小
    'epochs': 45,  # 轮数
    'device_num': '0'
}


# 定义训练流程
def train(train_loader, model, criterion, optimizer,device):
    metric_manager = mytorchutils.MetricManager()

    model.train()
    for i, (inputs, labels) in enumerate(train_loader, start=1):
        inputs = inputs.to(torch.float32)
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        criterion = criterion.to(device)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # 计算f1分数
        f1_macro = mytorchutils.calculate_f1_macro(outputs, labels)
        # 计算recall分数
        recall_macro = mytorchutils.calculate_recall_macro(outputs, labels)
        # 计算准确率分数
        acc = mytorchutils.accuracy(outputs, labels)
        # 更新参数
        metric_manager.update('Loss', loss.item())
        metric_manager.update('F1', f1_macro)
        metric_manager.update('Recall', recall_macro)
        metric_manager.update('Accuracy', acc)

    return metric_manager.metrics['Accuracy']["avg"], metric_manager.metrics['Loss']["avg"]


# 定义验证流程
def validate(val_loader, model, criterion,device):
    metric_manager = mytorchutils.MetricManager()
    model.eval()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader, start=1):
            inputs = inputs.to(torch.float32)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            criterion = criterion.to(device)
            loss = criterion(outputs, labels)
            # 计算f1分数
            f1_macro = mytorchutils.calculate_f1_macro(outputs, labels)
            # 计算recall分数
            recall_macro = mytorchutils.calculate_recall_macro(outputs, labels)
            # 计算准确率分数
            acc = mytorchutils.accuracy(outputs, labels)
            # 更新参数
            metric_manager.update('Loss', loss.item())
            metric_manager.update('F1', f1_macro)
            metric_manager.update('Recall', recall_macro)
            metric_manager.update('Accuracy', acc)

    return metric_manager.metrics['Accuracy']["avg"], metric_manager.metrics['Loss']["avg"]


# 展示训练过程的曲线
def draw(acc, loss, val_acc, val_loss, save_path):
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')

    plt.savefig(save_path, dpi=100)


if __name__ == '__main__':
    # 初始化设备、数据、模型、损失函数、优化器和训练轮次
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    print("process id is {}".format(os.getpid()))
    device = torch.device('cuda:'+params['device_num'] if torch.cuda.is_available() else 'cpu')
    print(device)
    train_dataset = mytorchutils.CloudDataset(params['train_path'])
    val_dataset = mytorchutils.CloudDataset(params['val_path'])
    train_len = mytorchutils.CloudDataset.__len__(train_dataset)
    val_len = mytorchutils.CloudDataset.__len__(val_dataset)
    print(train_len, val_len)

    train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=True)
    # model = mytorchutils.ResNet50()
    # 使用vgg16预训练模型
    # model = mytorchutils.VGG()
    #
    # weights = torch.load('./models/vgg16-397923af.pth')
    # model.load_state_dict(weights)
    # model = model.to(device)
    # # 加载torch原本的vgg16模型，设置pretrained=True，即使用预训练模型
    # num_fc = model.classifier[6].in_features  # 获取最后一层的输入维度
    # model.classifier[6] = torch.nn.Linear(num_fc, 6)  # 修改最后一层的输出维度，即分类数
    # # 对于模型的每个权重，使其不进行反向传播，即固定参数
    # for param in model.parameters():
    #     param.requires_grad = False
    # # 将分类器的最后层输出维度换成了num_cls，这一层需要重新学习
    # for param in model.classifier[6].parameters():
    #     param.requires_grad = True
    # model = timm.create_model('resnet50d', pretrained=False)
    model = timm.create_model('resnet101d', pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(2048, 6)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    # 模型结构
    print(model)

    # 开始训练
    accs = []
    losss = []
    val_accs = []
    val_losss = []
    best_acc = 0.0
    for epoch in range(params['epochs']):
        print(datetime.now())
        acc, loss = train(train_dataloader, model, criterion, optimizer,device)
        val_acc, val_loss = validate(val_dataloader, model, criterion,device)
        print("epoch={} train loss={},train acc={},val loss={},val acc={}".format(epoch,loss, acc, val_loss, val_acc))
        accs.append(acc)
        losss.append(loss)
        val_accs.append(val_acc)
        val_losss.append(val_loss)
        if val_acc >= best_acc:
            best_acc = val_acc
            print("nicer model,acc = {}".format(best_acc))
            save_path = f"./models3/{epoch}epochs_accuracy{val_acc:.5f}.pth"
            torch.save(model.state_dict(), save_path)
    print("training is over")
    # 绘制loss曲线
    draw(accs, losss, val_accs, val_losss, "./img/loss_and_acc.png")

import timm

from torch import nn, optim

import mytorchutils
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import os
import numpy as np
from matplotlib import pyplot as plt

params = {
    'test_path':  "../../../../work10/zhangyu/MCC_classification_test_data/data_split3/test",
    # 'test_path': "./test_data_split/test",
    'model_path': "./models/37epochs_accuracy0.86698.pth",
    'batch_size': 256,
    'device_num':'0'
}


# 定义训练过程
def test(val_loader, model, class_names,device):
    metric_manager = mytorchutils.MetricManager()  # 验证流程

    model.eval()  # 模型设置为验证格式

    test_real_labels = []
    test_pre_labels = []
    with torch.no_grad():  # 开始推理
        for i, (inputs, labels) in enumerate(val_loader, start=1):
            inputs = inputs.to(torch.float32)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            labels_np = labels.cpu().numpy()
            y_pred = torch.softmax(outputs, dim=1)
            y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
            test_real_labels.extend(labels_np)
            test_pre_labels.extend(y_pred)

            # 计算f1分数
            f1_macro = mytorchutils.calculate_f1_macro(outputs, labels)
            # 计算recall分数
            recall_macro = mytorchutils.calculate_recall_macro(outputs, labels)
            # 计算准确率分数
            acc = mytorchutils.accuracy(outputs, labels)
            # 更新参数
            metric_manager.update('F1', f1_macro)
            metric_manager.update('Recall', recall_macro)
            metric_manager.update('Accuracy', acc)

    class_names_length = len(class_names)
    confusion_matrix = np.zeros((class_names_length, class_names_length))
    for test_real_label, test_pre_label in zip(test_real_labels, test_pre_labels):
        confusion_matrix[test_real_label][test_pre_label] = confusion_matrix[test_real_label][test_pre_label] + 1

    confusion_matrix_sum = np.sum(confusion_matrix, axis=1).reshape(-1, 1)
    confusion_matrix_acc = confusion_matrix / confusion_matrix_sum

    draw_confusion_matrix(title="confusion_matrix", x_labels=class_names, y_labels=class_names, acc=confusion_matrix_acc, save_path="./img/confusion_matrix.png")
    # 加上模型名称

    return metric_manager.metrics['Accuracy']["avg"], metric_manager.metrics['F1']["avg"], metric_manager.metrics['Recall']["avg"]


def draw_confusion_matrix(title, x_labels, y_labels, acc, save_path):

    fig, ax = plt.subplots()
    im = ax.imshow(acc, cmap="OrRd")

    ax.set_xticks(np.arange(len(y_labels)))
    ax.set_yticks(np.arange(len(x_labels)))
    ax.set_xticklabels(y_labels)
    ax.set_yticklabels(x_labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            ax.text(j, i, round(acc[i, j], 2),ha="center", va="center", color="black")
    ax.set_xlabel("Predict label")
    ax.set_ylabel("Actual label")
    ax.set_title(title)
    fig.tight_layout()
    plt.colorbar(im)
    plt.savefig(save_path, dpi=100)



if __name__ == '__main__':
    # 初始化类名、设备、数据集、模型
    print("process id is {}".format(os.getpid()))
    class_names = ['cat0_Closed_MCC', 'cat1_Clustered_Cu', 'cat2_Disorganized MCC', 'cat3_Open_MCC','cat4_Solid_stratus', 'cat5_Suppressed_Cu']
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = torch.device('cuda:'+params['device_num'] if torch.cuda.is_available() else 'cpu')
    print(device)
    test_dataset = mytorchutils.CloudDataset(params['test_path'])
    test_len = mytorchutils.CloudDataset.__len__(test_dataset)
    print(test_len)
    test_dataloader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=True)
    # model = mytorchutils.VGG()
    # num_fc = model.classifier[6].in_features
    # model.classifier[6] = torch.nn.Linear(num_fc, 6)
    model = timm.create_model('resnet50d', pretrained=False)
    model.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(2048, 6)
    model = model.to(device)
    weights = torch.load(params['model_path'])
    model.load_state_dict(weights)
    model = model.to(device)
    # 开始测试
    acc, f1, recall = test(test_dataloader, model, class_names,device)
    print("testing is over, the result is")
    print(f"acc: {acc}, F1: {f1}, recall: {recall}")


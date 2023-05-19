import timm
from torch import nn
import mytorchutils
import torch
from torch.utils.data import DataLoader
import os
from matplotlib import pyplot as plt

# 在使用模型前需要保证进行判断的mat文件中含有RGB_1d的变量，且变量格式为128*128*3double，读取其他类型的文件需要对mytorchutils文件进行修改

params = {
    # 将需要进行分类的文件放置在该文件夹下，文件夹名称可修改
    'judge_path': "./to_be_judged",

    'model_path': "./models/56epochs_accuracy0.93908.pth",
    'batch_size': 1,
    'device_num': '0'
}


def judge(judge_loader, model, class_names, device):
    model.eval()

    with torch.no_grad():
        for i, (input, name) in enumerate(judge_loader, start=1):
            name = str(name)[2:-7]
            plt.imshow(input[0].cpu().numpy().transpose(1, 2, 0))
            input = input.to(torch.float32)
            input = input.to(device)
            output = model(input)
            y_pred = torch.softmax(output, dim=1).cpu()
            cert = torch.max(y_pred).cpu() / torch.sum(y_pred).cpu()
            y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
            y_pred = int(y_pred)
            print(f"the category of {name} is {class_names[y_pred]},cert is {cert}")
            plt.title(f'cat:{class_names[y_pred][5:]}  cert: {cert:.2f}')
            plt.savefig('./result/{}.png'.format(name))


if __name__ == '__main__':
    # 初始化类名、设备、数据集、模型
    # print("process id is {}".format(os.getpid()))
    class_names = ['cat0_Closed_MCC', 'cat1_Clustered_Cu', 'cat2_Disorganized MCC', 'cat3_Open_MCC',
                   'cat4_Solid_stratus', 'cat5_Suppressed_Cu']
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = torch.device('cuda:' + params['device_num'] if torch.cuda.is_available() else 'cpu')
    # print(device)
    test_dataset = mytorchutils.JudgeDataset(params['judge_path'])
    test_len = mytorchutils.CloudDataset.__len__(test_dataset)
    # print(test_len)
    test_dataloader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
    # model = mytorchutils.VGG()
    # num_fc = model.classifier[6].in_features
    # model.classifier[6] = torch.nn.Linear(num_fc, 6)
    model = timm.create_model('resnet50d', pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(2048, 6)
    model = model.to(device)
    weights = torch.load(params['model_path'])
    model.load_state_dict(weights)
    model = model.to(device)
    # 开始判断
    judge(test_dataloader, model, class_names, device)

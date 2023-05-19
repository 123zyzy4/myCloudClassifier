import shutil
import timm
from torch import nn
import mytorchutils
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from matplotlib import pyplot as plt


params = {
    'judge_path': "../../work10/jhliu/Application_CNNModel_Global_Morphology_MODIS/output/2015",
    # 'judge_path': "../../work7/jhliu/MCC_classification_test_data/output/app_data/2016",
    # 'test_path': "./test_data_split/test",
    # 'result_path':"../../work10/zhangyu/app_of_cnn_model/output/2016",

    'result_path':"../../work10/zhangyu/global_app_of_model/2015",
    'model_path': "./models/56epochs_accuracy0.93908.pth",
    'batch_size': 1,
    'device_num': '0'
}


# def judge(judge_loader, model, class_names, device,path):
#     model.eval()
#
#
#
#
#     with torch.no_grad():
#         for i, (input, name) in enumerate(judge_loader, start=1):
#             name = str(name)[2:-7]
#             input = input.to(torch.float32)
#             input = input.to(device)
#             output = model(input)
#             y_pred = torch.softmax(output, dim=1).cpu()
#
#
#             single_cert = torch.max(y_pred).cpu() / torch.sum(y_pred).cpu()
#             cert = single_cert
#             y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
#             save_path = path+"/"+name
#             np.savez(save_path, cat=y_pred, cert=cert)
#             print(name+" is judged")

def judge(judge_loader, model, class_names, device,path):
    model.eval()

    with torch.no_grad():
        for i, (input, name,scenes) in enumerate(judge_loader, start=1):
            name = str(name)[2:-7]
            input = input.to(torch.float32)
            input = input.to(device)
            output = model(input[0])
            y_pred = torch.softmax(output, dim=1).cpu()
            cert = np.zeros(int(scenes))
            for j in range(scenes):
                single_cert = torch.max(y_pred[j]).cpu() / torch.sum(y_pred[j]).cpu()
                cert[j] = single_cert
            y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
            save_path = path+"/"+name
            np.savez(save_path, cat=y_pred, cert=cert)
            print(name+" is judged")







if __name__ == '__main__':
    # 初始化类名、设备、数据集、模型
    # print("process id is {}".format(os.getpid()))
    class_names = ['cat0_Closed_MCC', 'cat1_Clustered_Cu', 'cat2_Disorganized MCC', 'cat3_Open_MCC',
                   'cat4_Solid_stratus', 'cat5_Suppressed_Cu']
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = torch.device('cuda:' + params['device_num'] if torch.cuda.is_available() else 'cpu')

    # 获取所有子文件夹的名称列表
    subdirs = [f for f in os.listdir(params['judge_path']) if os.path.isdir(os.path.join(params['judge_path'], f))]

    # 遍历每一个子文件夹
    for subdir in subdirs:
        print('Processing subdir:', subdir)
        # if subdir != '001':
        #     continue

        # 创建输出文件夹
        input_dir = os.path.join(params['judge_path'], subdir)
        output_dir = os.path.join(params['result_path'], subdir)
        os.makedirs(output_dir, exist_ok=True)

        # 执行判断流程
        # test_dataset = mytorchutils.JudgeDataset_new(input_dir)
        test_dataset = mytorchutils.JudgeDataset_new(input_dir)
        test_dataloader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
        model = timm.create_model('resnet50d', pretrained=False)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(2048, 6)
        model = model.to(device)
        weights = torch.load(params['model_path'])
        model.load_state_dict(weights)
        model = model.to(device)
        # 开始判断
        judge(test_dataloader, model, class_names, device ,output_dir)















# train的main部分：
# if __name__ == '__main__':
#     # 初始化设备、数据、模型、损失函数、优化器和训练轮次
#     os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     train_dataset = mytorchutils.CloudDataset(params['train_path'])
#     val_dataset = mytorchutils.CloudDataset(params['val_path'])
#     total_len = mytorchutils.CloudDataset.__len__(train_dataset)
#     print(total_len)
#     dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
#     model = mytorchutils.CloudClassifier().cuda()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.00001)
#
#     # 模型结构
#     print(model)
#
#     # 开始训练
#     accs = []
#     losss = []
#     best_acc = 0.0
#     for epoch in range(params['epochs']):
#         acc, loss = train(dataloader, model, criterion, optimizer)
#         accs.append(acc)
#         losss.append(loss)
#
#     print("training is over")
# #     # 绘制loss曲线
# #     draw(accs, losss, "./img/loss_and_acc.png")
# import timm
#
# import requests
#
# import urllib
#
# from torch import optim, nn
#
# import mytorchutils
# import os
# import torch
# import torch.nn as nn
# import torchvision.models as models
# import ssl

#
# model = torch.load("./models/resnet50d_ra2-464e36ba.pth")
# # print(model)
# model = timm.create_model('resnet101d', pretrained=False)
# print(model)
# url = 'https://image.baidu.com/'
# res = requests.get(url)
# # This restores the same behavior as before.
# context = ssl._create_unverified_context()
# response = urllib.request.urlopen("https://no-valid-cert", context=context)
#
# # Load pre-trained ResNet50 model
# resnet50 = models.resnet50(pretrained=True)
#
# # Modify the first layer to accept 128x128x3 input
# resnet50.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#
# # Modify the last layer to output 6 classes
# resnet50.fc = nn.Linear(2048, 6)

# Print the modified ResNet50 model
# print(resnet50)
# params = {
#     'test_path':  "../../../../work10/zhangyu/MCC_classification_test_data/data_split/test",
#     # 'test_path': "./test_data_split/test",
#     'model_path': "./models/30epoch_acc.pth",
#     'batch_size': 256,
#     'device_num':'3'
# }
# print(os.getpid())
# device = torch.device('cuda:'+params['device_num'] if torch.cuda.is_available() else 'cpu')
# print(device)

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
#
# model = model.to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.00001)
#
# # 模型结构
# print(model)
# model = mytorchutils.resnet_50()



# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
#
# # from mpl_toolkits.basemap import Basemap
# filename = './analyse_test_data/MYD021KM.A2016038.2005.061.2018055145424.mat'
# # 读取mat文件
# # data = h5py.File('./analyse_test_data/MYD021KM.A2016038.2005.061.2018055145424.mat')
# try:
#     with h5py.File(filename, 'r') as f:
#         if 'lon_center' in f.keys():
#             lon = f['lon_center'][()]
#         if 'lat_center' in f.keys():
#             lat = f['lat_center'][()]
#         if 'CF' in f.keys():
#             CF = f['CF'][()]
#
# except OSError as e:
#     print(f"{filename}文件无法打开。")
#
# # 绘制全球地图
# fig = plt.figure(figsize=(12, 6))
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
# ax.set_xticks(np.arange(-180., 180.0, 10.0), crs=ccrs.PlateCarree())
# ax.set_yticks(np.arange(-90., 90.0, 10.0), crs=ccrs.PlateCarree())
# ax.xaxis.set_ticklabels([])
# ax.yaxis.set_ticklabels([])
#
# # print(lon[0,1])
#
# # 绘制云顶温度数据
# # im = ax.pcolormesh(lon, lat, CF, cmap='coolwarm', vmin=0, vmax=1, transform=ccrs.PlateCarree())
#
# # im = ax.pcolormesh(lon[0,i], lat[0,i], CF[0,i], cmap='coolwarm', vmin=-90, vmax=-10, transform=ccrs.PlateCarree())
# # im = ax.pcolormesh(np.ravel(lon), np.ravel(lat), np.ravel(CF), cmap='coolwarm', vmin=-90, vmax=-10, transform=ccrs.PlateCarree())
#
# ax.coastlines()
# # 添加颜色条
# # cbar = plt.colorbar(im, ax=ax, shrink=0.8)
# # cbar.ax.tick_params(labelsize=10)
#
# # 显示图形
# plt.title('Global Cloud Top Temperature Distribution')
# save_path = './analyse_result/1.png'
# plt.savefig(save_path, dpi=100)
# import matplotlib.pyplot as plt
# import numpy as np

# def f(x,y): return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)

# n = 3
# x = np.linspace(-3,3,n)
# y = np.linspace(-3,3,n)
# X,Y = np.meshgrid(x,y)
# Z = [5,7,9]
# Z = np.array(Z)
# Z = Z.reshape((len(x), len(y)))
# plt.contourf(X, Y, Z)
#
# plt.show()

# import numpy as np
# # 从npz文件中加载数组
# loaded_data = np.load("./analyse_result/2016_frequency.npz")
#
# # 获取数组
# cloud_frequency = loaded_data['cloud_frequency']
#
#
# # 打印数组
# print(cloud_frequency)

# import numpy as np
#
#
# # 生成一个360*180*6的随机矩阵
# data = np.zeros((3, 3, 6))
# print(data)
#
# # 对第三个维度求和，得到一个360*180的矩阵
# sum_matrix = np.sum(data, axis=2)
# zero_indices = np.where(sum_matrix == 0)
# sum_matrix[zero_indices] = 1
# # 将原矩阵除以新的矩阵，得到一个360*180*6的概率分布矩阵
# prob_matrix = data / sum_matrix[:, :, np.newaxis]
# print(prob_matrix)


# import numpy as np
#
# # 创建一个空的数组
# data_array = np.empty((6, 0))
#
# # 逐个添加数据
# data1 = 1  # 示例数据1
# data2 = 2  # 示例数据2
# data3 = 3 # 示例数据3
#
# # 添加到数组中
# data_array = np.column_stack((data_array, data1))
# data_array = np.column_stack((data_array, data2))
# data_array = np.column_stack((data_array, data3))
#
# # 打印数组
# print(data_array)
# print(data_array.shape)

# data_list = [[] for _ in range(6)]
#
# data1 = 1  # 示例数据1
# data2 = 2  # 示例数据2
# data3 = [8, 9]  # 示例数据3
#
# data_list[0].append(data1)
# data_list[1].append(data2)
# data_list[2].append(data3)
#
# # 打印列表
# print(data_list)
# import os
#
# dirpath = '/path/to/some/folder'
#
# # 获取倒数第二层文件夹名
# parent_folder_name = os.path.basename(os.path.dirname(dirpath))
#
# print(parent_folder_name)  # 输出倒数第二层文件夹名
# import h5py
# import os
#
#
# def read_alldata(filename):
#     try:
#         with h5py.File(filename, 'r') as f:
#             if 'lat_center' in f.keys():
#                 lat = f['lat_center'][()]
#             if 'lon_center' in f.keys():
#                 lon = f['lon_center'][()]
#             if 'CF' in f.keys():
#                 CF = f['CF'][()]
#             if 'LWP_ave' in f.keys():
#                 LWP = f['LWP_ave'][()]
#             if 'Re_ave' in f.keys():
#                 Re = f['Re_ave'][()]
#             if 'COT_ave' in f.keys():
#                 COT = f['COT_ave'][()]
#
#             return lat[0], lon[0], CF[0],LWP[0],Re[0],COT[0]
#     except OSError:
#         print(f"{filename}文件无法打开。")
#     return None
#
# filename = './to_be_judged/matlab.mat'
# lat, lon,single_CF,single_LWP,single_Re,single_COT = read_alldata(filename)
# print(lat, lon,single_CF,single_LWP,single_Re,single_COT)

# import numpy as np
# data= np.load("./analyse_result/global_frequency.npz")
# cloud_count =data['cloud_count']
# count = 0
# for i in range(360):
#     for j in range(180):
#         count = count + cloud_count[i][j]
#
# alllast = 0
# for i in range(6):
#     alllast = alllast +count[i]
# print(alllast)


from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm

class_names = ['cat0_Closed MCC', 'cat1_Clustered Cumulus', 'cat2_Disorganized MCC', 'cat3_Open MCC(x2)','cat4_Solid Stratus', 'cat5_Suppressed Cumulus']
# filename = './analyse_result/global_frequency.npz'
# data = np.load(filename)
cloud_frequency = np.zeros((360,180,6))


for i in range(360):
    for j in range(180):
        cloud_frequency[i,j,3] = 1


# 获取有效数据的范围
valid_indices = np.where(np.sum(cloud_frequency, axis=2) > 0)
min_lon = valid_indices[0].min()-180
max_lon = valid_indices[0].max()-180
min_lat = valid_indices[1].min()-90
max_lat = valid_indices[1].max()-90
# min_lon = -180
# max_lon = 180
# min_lat = -90
# max_lat = 90

# 获取经纬度网格
lons, lats = np.meshgrid(np.linspace(min_lon, max_lon, max_lon-min_lon+1), np.linspace(min_lat, max_lat, max_lat-min_lat+1))

land = cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='white')

fig = plt.figure(figsize=(12, 8))
ax0 = plt.subplot(231, projection=ccrs.PlateCarree())
ax1 = plt.subplot(234, projection=ccrs.PlateCarree())
ax2 = plt.subplot(232, projection=ccrs.PlateCarree())
ax3 = plt.subplot(233, projection=ccrs.PlateCarree())
ax4 = plt.subplot(235, projection=ccrs.PlateCarree())
ax5 = plt.subplot(236, projection=ccrs.PlateCarree())

# 绘制云分布频率图
axes = [ax1, ax4, ax2, ax3, ax0, ax5]
for i, ax in enumerate(axes):
    levels = np.linspace(cloud_frequency.min(), cloud_frequency.max(), 500)

    # 绘制颜色填充图
    cf = ax.contourf(lons, lats, cloud_frequency[min_lon + 180:max_lon + 181, min_lat + 90:max_lat + 91, i].T,
                     levels=levels, cmap=cm.coolwarm)

    # 绘制经纬度网格
    ax.set_xticks(np.arange(min_lon, max_lon+1, 60))
    ax.set_yticks(np.arange(min_lat, max_lat+1, 30))
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(axis='both', labelsize=10)

    # 绘制海岸线
    ax.coastlines()
    ax.add_feature(land)

    # 添加标题
    ax.set_title(class_names[i][5:], fontsize=12)


# 调整子图之间的间距
plt.subplots_adjust(hspace=0.05)

# 调整下面三张子图的位置
pos1 = ax1.get_position()
pos4 = ax4.get_position()
pos2 = ax2.get_position()
ax2.set_position([pos2.x0, pos1.y0, pos2.width, pos2.height])
ax4.set_position([pos4.x0, pos2.y0, pos4.width, pos4.height])
# fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.27, 0.02, 0.4])
cf = fig.colorbar(cf, cax=cbar_ax, orientation='vertical')


fig.suptitle("Occurrence Frequency of Marine Low Cloud Types", fontsize=16, y=0.84)
plt.subplots_adjust(top=0.85)

# 保存图像
plt.savefig('./analyse_result/global_frequency.png', dpi=300)
plt.close()
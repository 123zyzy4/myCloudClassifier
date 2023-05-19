# import math
# import h5py
# import numpy as np
# import os
#
#
# # 读取mat文件，返回经纬度数据，shape为scenes*1
# def read_lat_lon(filename):
#     try:
#         with h5py.File(filename, 'r') as f:
#             if 'lat_center' in f.keys():
#                 lat1 = f['lat_center'][()]
#             if 'lon_center' in f.keys():
#                 lon1 = f['lon_center'][()]
#
#             if len(np.shape(lat1)) == 2:
#                 scenes = lat1.shape[1]
#             elif len(np.shape(lat1)) == 1:
#                 scenes = 1
#             else:
#                 print("Invalid shape!")
#
#             lat = np.zeros((scenes, 1))
#             lon = np.zeros((scenes, 1))
#             if scenes == 1:
#                 lat[0, 0] = lat1
#                 lon[0, 0] = lon1
#
#             else:
#                 lat = lat1.T
#                 lon = lon1.T
#                 # for i in range(scenes):
#                 #     lat[i, 0] = lat1[i, 0]
#                 #     lon[i, 0] = lon1[i, 0]
#
#             return lat, lon
#     except OSError:
#         print(f"{filename}文件无法打开。")
#     return None
#
# # 读取mat文件，返回经纬度数据，shape为scenes*1
# def read_lat_lon2(filename):
#     try:
#         with h5py.File(filename, 'r') as f:
#             if 'lat1_1d' in f.keys():
#                 lat1 = f['lat1_1d'][()]
#             if 'lon1_1d' in f.keys():
#                 lon1 = f['lon1_1d'][()]
#
#             # if len(np.shape(lat1)) == 2:
#             #     scenes = lat1.shape[1]
#             # elif len(np.shape(lat1)) == 1:
#             #     scenes = 1
#             # else:
#             #     print("Invalid shape!")
#             scenes = 1
#             lat = np.zeros((scenes, 1))
#             lon = np.zeros((scenes, 1))
#             if scenes == 1:
#                 lat[0, 0] = lat1[64][64]
#                 lon[0, 0] = lon1[64][64]
#
#             else:
#                 lat = lat1.T
#                 lon = lon1.T
#                 # for i in range(scenes):
#                 #     lat[i, 0] = lat1[i, 0]
#                 #     lon[i, 0] = lon1[i, 0]
#
#             return lat, lon
#     except OSError:
#         print(f"{filename}文件无法打开。")
#     return None
#
#
# # 读取npy文件，返回cat和cert，shape都是scenes*1
# def read_cat_cert(filename):
#     data = np.load(filename)
#
#     cat = data['cat']
#     cert = data['cert']
#     return cat, cert
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
#
#
# if __name__ == '__main__':
#     # 创建一个 360*180*6 的数组用于存储每种类型云出现的次数
#     cloud_count = np.zeros((360, 180, 6))
#     CF = [[] for _ in range(6)]
#     CF_mean = []
#     CF_std = []
#     LWP = [[] for _ in range(6)]
#     LWP_mean = []
#     LWP_std = []
#     Re = [[] for _ in range(6)]
#     Re_mean = []
#     Re_std = []
#     COT = [[] for _ in range(6)]
#     COT_mean = []
#     COT_std = []
#
#     debug = 1
#     # 遍历所有子文件夹和文件
#     for dirpath, dirnames, filenames in os.walk('../../work10/zhangyu/app_of_cnn_model/output'):
#         for filename in filenames:
#             if filename.endswith('.npz'):
#
#                 # 读取类别数据
#                 cat, cert = read_cat_cert(os.path.join(dirpath, filename))
#                 debug = debug + 1
#                 filename = os.path.join('../../work7/jhliu/MCC_classification_test_data/output/app_data',os.path.basename(os.path.dirname(dirpath)), os.path.basename(dirpath), filename[:-4] + '.mat')
#                 # print(os.path.basename(dirpath)+filename[:-4]+" is be solved")
#                 # 读取经纬度数据
#                 lat, lon,single_CF,single_LWP,single_Re,single_COT = read_alldata(filename)
#
#                 # lat_idx = np.floor(lat + 90).astype(int)
#                 # lon_idx = np.floor(lon + 180).astype(int)
#                 # if not math.isnan(single_CF):
#                 #     CF[cat[0]].append(single_CF)
#                 # if not math.isnan(single_LWP):
#                 #     LWP[cat[0]].append(single_LWP)
#                 # if not math.isnan(single_CF):
#                 #     Re[cat[0]].append(single_Re)
#                 # if not math.isnan(single_CF):
#                 #     COT[cat[0]].append(single_COT)
#                 if not np.isnan(single_CF).any():
#                     CF[cat[0]].append(single_CF)
#                 else:
#                     print("nan")
#                 if not np.isnan(single_LWP).any():
#                     LWP[cat[0]].append(single_LWP)
#                 else:
#                     print("nan")
#                 if not np.isnan(single_Re).any():
#                     Re[cat[0]].append(single_Re)
#                 else:
#                     print("nan")
#                 if not np.isnan(single_COT).any():
#                     COT[cat[0]].append(single_COT)
#                 else:
#                     print("nan")
#
#
#                 # cloud_count[lon_idx, lat_idx, cat[0]] += 1
#
#                 # scenes = 1
#                 # for i in range(scenes):
#                 #     # 判断经纬度属于哪个网格
#                 #     lat_idx = np.floor(lat[i] + 90).astype(int)
#                 #     lon_idx = np.floor(lon[i] + 180).astype(int)
#                 #     if not math.isnan(single_CF):
#                 #         CF[cat[i]].append(single_CF)
#                 #     if not math.isnan(single_LWP):
#                 #         LWP[cat[i]].append(single_LWP)
#                 #     if not math.isnan(single_CF):
#                 #         Re[cat[i]].append(single_Re)
#                 #     if not math.isnan(single_CF):
#                 #         COT[cat[i]].append(single_COT)
#                 #
#                 #     cloud_count[lon_idx, lat_idx, cat[i]] += 1
#
#     for i, data in enumerate(CF):
#         data = np.array(data)
#         mean = np.mean(data)
#         std = np.std(data)
#         CF_mean.append(mean)
#         CF_std.append(std)
#     for i, data in enumerate(LWP):
#         data = np.array(data)
#         mean = np.mean(data)
#         std = np.std(data)
#         LWP_mean.append(mean)
#         LWP_std.append(std)
#     for i, data in enumerate(Re):
#         data = np.array(data)
#         mean = np.mean(data)
#         std = np.std(data)
#         Re_mean.append(mean)
#         Re_std.append(std)
#     for i, data in enumerate(COT):
#         data = np.array(data)
#         mean = np.mean(data)
#         std = np.std(data)
#         COT_mean.append(mean)
#         COT_std.append(std)
#
#     # with open('./analyse_result/20152016mean_std.txt', 'w') as f:
#     #     for i in range(6):
#     #         f.write(f"CF 类别 {i} 均值: {CF_mean[i]}, 标准差: {CF_std[i]}\n")
#     #         f.write(f"LWP 类别 {i} 均值: {LWP_mean[i]}, 标准差: {LWP_std[i]}\n")
#     #         f.write(f"Re 类别 {i} 均值: {Re_mean[i]}, 标准差: {Re_std[i]}\n")
#     #         f.write(f"COT 类别 {i} 均值: {COT_mean[i]}, 标准差: {COT_std[i]}\n")
#
#
#
#     # 输出结果到控制台
#     # for i in range(6):
#     #     print(f"CF 类别 {i} 均值: {CF_mean[i]}, 标准差: {CF_std[i]}\n")
#     #     print(f"LWP 类别 {i} 均值: {LWP_mean[i]}, 标准差: {LWP_std[i]}\n")
#     #     print(f"Re 类别 {i} 均值: {Re_mean[i]}, 标准差: {Re_std[i]}\n")
#     #     print(f"COT 类别 {i} 均值: {COT_mean[i]}, 标准差: {COT_std[i]}\n")
#     print("CF")
#     print(CF_mean)
#     print(CF_std)
#
#     print("LWP")
#     print(LWP_mean)
#     print(LWP_std)
#
#     print("Re")
#     print(Re_mean)
#     print(Re_std)
#
#     print("COT")
#     print(COT_mean)
#     print(COT_std)
#     # 计算每种类型云出现的频率
#     # print('cloud_count')
#     # print(cloud_count)
#     # total_count = np.sum(cloud_count, axis=2)
#     # zero_indices = np.where(total_count==0)
#     # total_count[zero_indices] = 1
#     # cloud_frequency = cloud_count / total_count[:, :, np.newaxis]
#     # print('cloud_frequency')
#     # print(cloud_frequency)
#     # np.savez('./analyse_result/20152016_frequency', cloud_frequency=cloud_frequency,cloud_count=cloud_count)
#


import h5py
import numpy as np
import os


# 读取mat文件，返回经纬度数据，shape为scenes*1
def read_lat_lon(filename):
    try:
        with h5py.File(filename, 'r') as f:
            if 'lat_center' in f.keys():
                lat1 = f['lat_center'][()]
            if 'lon_center' in f.keys():
                lon1 = f['lon_center'][()]

            if len(np.shape(lat1)) == 2:
                scenes = lat1.shape[1]
            elif len(np.shape(lat1)) == 1:
                scenes = 1
            else:
                print("Invalid shape!")

            lat = np.zeros((scenes, 1))
            lon = np.zeros((scenes, 1))
            if scenes == 1:
                lat[0, 0] = lat1
                lon[0, 0] = lon1

            else:
                lat = lat1.T
                lon = lon1.T
                # for i in range(scenes):
                #     lat[i, 0] = lat1[i, 0]
                #     lon[i, 0] = lon1[i, 0]




            return lat, lon
    except OSError:
        print(f"{filename}文件无法打开。")
    return None


# 读取npy文件，返回cat和cert，shape都是scenes*1
def read_cat_cert(filename):
    data = np.load(filename)

    cat = data['cat']
    cert = data['cert']
    return cat, cert


if __name__ == '__main__':
    # 创建一个 360*180*6 的数组用于存储每种类型云出现的次数
    cloud_count = np.zeros((360, 180, 6))

    # 遍历所有子文件夹和文件
    for dirpath, dirnames, filenames in os.walk('../../work10/zhangyu/global_app_of_model'):
        for filename in filenames:
            if filename.endswith('.npz'):
                # 读取类别数据
                cat, cert = read_cat_cert(os.path.join(dirpath, filename))

                latlonfilename = os.path.join('../../work10/jhliu/Application_CNNModel_Global_Morphology_MODIS/output',os.path.basename(os.path.dirname(dirpath)),  os.path.basename(dirpath), filename[:-4] + '.mat')
                # 读取经纬度数据
                lat, lon = read_lat_lon(latlonfilename)

                if cat.shape[0] != lat.shape[0]:
                    print("The files are inconsistent!")
                scenes = cat.shape[0]
                for i in range(scenes):
                    # 判断经纬度属于哪个网格
                    lat_idx = np.floor(lat[i] + 90).astype(int)
                    lon_idx = np.floor(lon[i] + 180).astype(int)

                    cloud_count[lon_idx, lat_idx, cat[i]] += 1

    # 计算每种类型云出现的频率
    print('cloud_count')
    print(cloud_count)
    total_count = np.sum(cloud_count, axis=2)
    zero_indices = np.where(total_count== 0)
    total_count[zero_indices] = 1
    cloud_frequency = cloud_count / total_count[:, :, np.newaxis]
    print('cloud_frequency')
    print(cloud_frequency)
    np.savez('./analyse_result/global_frequency', cloud_frequency=cloud_frequency,cloud_count=cloud_count)
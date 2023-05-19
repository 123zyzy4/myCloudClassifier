import math

import h5py
import os
import numpy as np
from frequency import read_cat_cert


def read_meteorological_condition(filename):
    try:
        with h5py.File(filename, 'r') as f:
            if 'ERA5_results' in f.keys():
                data = f['ERA5_results'][()]
            LTS = data[12]-data[36]*math.pow(1000/700, 0.286)
            SST = data[3]
            RH700hPa = data[37]
            RH1000hPa= data[13]
            w700hPa= data[39]
            w1000hPa= data[15]

            return LTS,SST,RH700hPa,RH1000hPa,w700hPa,w1000hPa
    except OSError:
        print(f"{filename}文件无法打开。")
    return None

# filename = './to_be_judged/ERA5_MYD021KM.A2015001.1830.061_indI_0237_indJ_0202.mat'
# LTS,SST,RH700hPa,RH1000hPa,w700hPa,w1000hPa = read_meteorological_condition(filename)
# print(LTS,SST,RH700hPa,RH1000hPa,w700hPa,w1000hPa)

if __name__ == '__main__':

    LTS = [[] for _ in range(6)]
    LTS_mean = []
    LTS_std = []

    SST = [[] for _ in range(6)]
    SST_mean = []
    SST_std = []

    RH700hPa = [[] for _ in range(6)]
    RH700hPa_mean = []
    RH700hPa_std = []

    RH1000hPa = [[] for _ in range(6)]
    RH1000hPa_mean = []
    RH1000hPa_std = []

    w700hPa = [[] for _ in range(6)]
    w700hPa_mean = []
    w700hPa_std = []
    w1000hPa = [[] for _ in range(6)]
    w1000hPa_mean = []
    w1000hPa_std = []

    # 遍历所有子文件夹和文件
    for dirpath, dirnames, filenames in os.walk('../../work10/zhangyu/app_of_cnn_model/output'):
        for filename in filenames:
            if filename.endswith('.npz'):
                # 读取类别数据
                cat, cert = read_cat_cert(os.path.join(dirpath, filename))

                filename = os.path.join('../../work7/jhliu/MCC_classification_test_data/output/ERA5_Collocation_CB_data',os.path.basename(os.path.dirname(dirpath)), os.path.basename(dirpath), 'ERA5_'+filename[:-4] + '.mat')
                # 读取经纬度数据
                single_LTS,single_SST,single_RH700hPa,single_RH1000hPa,single_w700hPa,single_w1000hPa = read_meteorological_condition(filename)

                if not np.isnan(single_LTS).any():
                    LTS[cat[0]].append(single_LTS)
                if not np.isnan(single_SST).any() and 400 > single_SST > 200:
                    SST[cat[0]].append(single_SST)
                if not np.isnan(single_RH700hPa).any():
                    RH700hPa[cat[0]].append(single_RH700hPa)
                if not np.isnan(single_RH1000hPa).any():
                    RH1000hPa[cat[0]].append(single_RH1000hPa)
                if not np.isnan(single_w700hPa).any():
                    w700hPa[cat[0]].append(single_w700hPa)
                if not np.isnan(single_w1000hPa).any():
                    w1000hPa[cat[0]].append(single_w1000hPa)



    for i, data in enumerate(LTS):
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        LTS_mean.append(mean)
        LTS_std.append(std)
    for i, data in enumerate(SST):
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        SST_mean.append(mean)
        SST_std.append(std)
    for i, data in enumerate(RH700hPa):
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        RH700hPa_mean.append(mean)
        RH700hPa_std.append(std)
    for i, data in enumerate(RH1000hPa):
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        RH1000hPa_mean.append(mean)
        RH1000hPa_std.append(std)
    for i, data in enumerate(w700hPa):
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        w700hPa_mean.append(mean)
        w700hPa_std.append(std)
    for i, data in enumerate(w1000hPa):
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        w1000hPa_mean.append(mean)
        w1000hPa_std.append(std)

    # with open('./analyse_result/20152016meteorological_condition.txt', 'w') as f:
    #     f.write("LTS")
    #     f.write(LTS_mean)
    #     f.write(LTS_std)
    #
    #     f.write("SST")
    #     f.write(SST_mean)
    #     f.write(SST_std)
    #
    #     f.write("RH700hPa")
    #     f.write(RH700hPa_mean)
    #     f.write(RH700hPa_std)
    #
    #     f.write("RH1000hPa")
    #     f.write(RH1000hPa_mean)
    #     f.write(RH1000hPa_std)
    #
    #     f.write("w700hPa")
    #     f.write(w700hPa_mean)
    #     f.write(w700hPa_std)
    #
    #     f.write("w1000hPa")
    #     f.write(w1000hPa_mean)
    #     f.write(w1000hPa_std)
            # f.write(f"SST 类别 {i} 均值: {SST_mean[i]}, 标准差: {SST_std[i]}\n")
            # f.write(f"RH700hPa 类别 {i} 均值: {RH700hPa_mean[i]}, 标准差: {RH700hPa_std[i]}\n")
            # f.write(f"RH1000hPa 类别 {i} 均值: {RH1000hPa_mean[i]}, 标准差: {RH1000hPa_std[i]}\n")
            # f.write(f"w700hPa 类别 {i} 均值: {w700hPa_mean[i]}, 标准差: {w700hPa_std[i]}\n")
            # f.write(f"w1000hPa 类别 {i} 均值: {w1000hPa_mean[i]}, 标准差: {w1000hPa_std[i]}\n")

    print("LTS")
    print(LTS_mean)
    print(LTS_std)

    print("SST")
    print(SST_mean)
    print(SST_std)

    print("RH700hPa")
    print(RH700hPa_mean)
    print(RH700hPa_std)

    print("RH1000hPa")
    print(RH1000hPa_mean)
    print(RH1000hPa_std)

    print("w700hPa")
    print(w700hPa_mean)
    print(w700hPa_std)

    print("w1000hPa")
    print(w1000hPa_mean)
    print(w1000hPa_std)

    # # 输出结果到控制台
    # for i in range(6):
    #     print(f"LTS 类别 {i} 均值: {LTS_mean[i]}, 标准差: {LTS_std[i]}\n")
    #     print(f"SST 类别 {i} 均值: {SST_mean[i]}, 标准差: {SST_std[i]}\n")
    #     print(f"RH700hPa 类别 {i} 均值: {RH700hPa_mean[i]}, 标准差: {RH700hPa_std[i]}\n")
    #     print(f"RH1000hPa 类别 {i} 均值: {RH1000hPa_mean[i]}, 标准差: {RH1000hPa_std[i]}\n")
    #     print(f"w700hPa 类别 {i} 均值: {w700hPa_mean[i]}, 标准差: {w700hPa_std[i]}\n")
    #     print(f"w1000hPa 类别 {i} 均值: {w1000hPa_mean[i]}, 标准差: {w1000hPa_std[i]}\n")



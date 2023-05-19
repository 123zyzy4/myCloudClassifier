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
            data1 = data[12]
            data2 = data[36]
            data3 = data2*math.pow(1000/700, 0.286)
            LTS =data3-data1
            # LTS = data[12]-data[36]*math.pow(1000/700, 0.286)


            return LTS
    except OSError:
        print(f"{filename}文件无法打开。")
    return None

# filename = './to_be_judged/ERA5_MYD021KM.A2015001.1830.061_indI_0237_indJ_0202.mat'
# LTS,SST,RH700hPa,RH1000hPa,w700hPa,w1000hPa = read_meteorological_condition(filename)
# print(LTS,SST,RH700hPa,RH1000hPa,w700hPa,w1000hPa)

if __name__ == '__main__':



    SST = [[] for _ in range(6)]
    SST_mean = []
    SST_std = []


    # 遍历所有子文件夹和文件
    for dirpath, dirnames, filenames in os.walk('../../work10/zhangyu/app_of_cnn_model/output'):
        for filename in filenames:
            if filename.endswith('.npz'):
                # 读取类别数据
                cat, cert = read_cat_cert(os.path.join(dirpath, filename))

                filename = os.path.join('../../work7/jhliu/MCC_classification_test_data/output/ERA5_Collocation_CB_data',os.path.basename(os.path.dirname(dirpath)), os.path.basename(dirpath), 'ERA5_'+filename[:-4] + '.mat')
                # 读取经纬度数据
                single_SST = read_meteorological_condition(filename)


                if not np.isnan(single_SST).any() and single_SST<50 and single_SST>0:
                    SST[cat[0]].append(single_SST)
                    # print(single_SST)







    for i, data in enumerate(SST):
        data = np.array(data,dtype=float)
        mean = np.mean(data)
        std = np.std(data)
        SST_mean.append(mean)
        SST_std.append(std)


    # with open('./analyse_result/20152016LTS.txt', 'w') as f:
    #     for i in range(6):
    #         f.write(f"LTS 类别 {i} 均值: {SST_mean[i]}, 标准差: {SST_std[i]}\n")

    # # 输出结果到控制台
    # for i in range(6):
    #     print(f"LTS 类别 {i} 均值: {SST_mean[i]}, 标准差: {SST_std[i]}\n")

    print(SST_mean)
    print(SST_std)




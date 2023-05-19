import h5py
import numpy as np
import os
import scipy.io

# 加载MAT文件
with h5py.File('./split_mat/MYD021KM.A2015001.2020.061.2018045201552.mat', 'r') as f:
    refl_01_CB = np.array(f['refl_01_CB'])
    refl_03_CB = np.array(f['refl_03_CB'])
    refl_04_CB = np.array(f['refl_04_CB'])

# 检查是否满足要求
assert refl_01_CB.shape == refl_03_CB.shape == refl_04_CB.shape == (144, 128, 128)

# 遍历场景并将它们保存为单独的MAT文件
for i in range(144):
    # 将三个变量组合成一个RGB_1d变量
    rgb_1d = np.stack((refl_01_CB[i], refl_03_CB[i], refl_04_CB[i]), axis=-1)
    assert rgb_1d.shape == (128, 128, 3)

    # 保存为MAT文件
    filename = f'scene_{i + 1}.mat'
    filepath = os.path.join('./split_mat_result', filename)
    scipy.io.savemat(filepath, {'RGB_1d': rgb_1d})

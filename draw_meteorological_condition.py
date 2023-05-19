import matplotlib.pyplot as plt

# class_names = ['cat0_Closed MCC', 'cat1_Clust Cu', 'cat2_Disorg MCC', 'cat3_Open MCC',
#                'cat4_Solid Str', 'cat5_Supp Cu']
# to do
class_names = ['cat4_Solid Stra','cat0_Closed MCC', 'cat2_Disorg MCC','cat3_Open MCC','cat1_Clust Cu',
                 'cat5_Supp Cu']

oldLTS_mean = [17.567258304051247, 15.806436023116738, 17.94655858409465, 12.683560104950466, 17.792282965070367, 16.210999153195505]
oldLTS_std =[5.151014117884324, 3.449219664078452, 4.344929815900107, 4.835189169591517, 3.919582392233743, 3.0970286206052156]

# 调整后的顺序为 1 4 2 3 0 5
new_order = [4 ,0 ,2 ,3 ,1 ,5]

# 根据新的顺序调整列表
LTS_mean = [oldLTS_mean[i] for i in new_order]
LTS_std = [oldLTS_std[i] for i in new_order]

class_names= [name[5:] for name in class_names]

# 设置图表布局
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制每个类别的平均值及标准差区间
for i, name in enumerate(class_names):

    ax.plot(i, LTS_mean[i], 'o', markersize=20, label=f'{name} (mean)')
    ax.vlines(i, LTS_mean[i] - LTS_std[i], LTS_mean[i] + LTS_std[i], color='black')
    ax.hlines([LTS_mean[i] - LTS_std[i], LTS_mean[i] + LTS_std[i]], i - 0.1, i + 0.1, color='black')

# to do(K)
ax.set_ylabel('LTS(K)',fontsize=14)

ax.set_xticks(range(len(class_names)))
ax.set_xticklabels(class_names,fontsize=14)

# to do
ax.set_title('LTS',fontsize=16)

# to do
plt.savefig('./analyse_result/LTS.png', dpi=300)
plt.close()

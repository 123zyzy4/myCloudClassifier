# # # from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # import cartopy.crs as ccrs
# # # import cartopy.feature as cfeature
# # #
# # # class_names = ['cat0_Closed_MCC', 'cat1_Clustered_Cu', 'cat2_Disorganized MCC', 'cat3_Open_MCC','cat4_Solid_stratus', 'cat5_Suppressed_Cu']
# # # filename = './analyse_result/20161012_frequency.npz'
# # # data = np.load(filename)
# # # cloud_frequency = data['cloud_frequency']
# # # cloud_count = data['cloud_count']
# # #
# # #
# # #
# # # # 获取经纬度网格
# # # lons, lats = np.meshgrid(np.linspace(-180, 180, 360), np.linspace(-90, 90, 180))
# # #
# # # land = cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='white')
# # #
# # # # 绘制云分布频率图
# # # for i in range(6):
# # #     fig = plt.figure(figsize=(12, 8))
# # #     ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
# # #
# # #     # 绘制颜色填充图
# # #     cf = ax.contourf(lons, lats, cloud_frequency[:, :, i].T, cmap='Purples', vmin=cloud_frequency.min(), vmax=cloud_frequency.max())
# # #     plt.colorbar(cf, ax=ax, orientation='horizontal')
# # #
# # #
# # #
# # #     # 绘制经纬度网格
# # #     ax.set_xticks(np.arange(-180, 180, 60))
# # #     ax.set_yticks(np.arange(-90, 90, 30))
# # #     lon_formatter = LongitudeFormatter(zero_direction_label=True)
# # #     lat_formatter = LatitudeFormatter()
# # #     ax.xaxis.set_major_formatter(lon_formatter)
# # #     ax.yaxis.set_major_formatter(lat_formatter)
# # #     ax.tick_params(axis='both', labelsize=8)
# # #
# # #     # 绘制海岸线
# # #     ax.coastlines()
# # #     ax.add_feature(land)
# # #     # ax.add_feature(cfeature.LAND, edgecolor='black')
# # #
# # #     # 设置图标题
# # #     ax.set_title('Frequency Of  {}'.format(class_names[i][5:]))
# # #
# # #     # 保存图像
# # #     plt.savefig('./analyse_result/{}_frequency.png'.format(class_names[i][5:]), dpi=300)
# # #     plt.close()
# # #
# # #
# # #
# # #
# # from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import cartopy.crs as ccrs
# # import cartopy.feature as cfeature
# # import matplotlib.cm as cm
# #
# # class_names = ['cat0_Closed MCC', 'cat1_Clustered Cumulus', 'cat2_Disorganized MCC', 'cat3_Open MCC','cat4_Solid Stratus', 'cat5_Suppressed Cumulus']
# # filename = './analyse_result/20152016_frequency.npz'
# # data = np.load(filename)
# # cloud_frequency = data['cloud_frequency']
# # cloud_count = data['cloud_count']
# #
# # # 获取有效数据的范围
# # valid_indices = np.where(np.sum(cloud_count, axis=2) > 0)
# # min_lon = valid_indices[0].min()-180
# # max_lon = valid_indices[0].max()-180
# # min_lat = valid_indices[1].min()-90
# # max_lat = valid_indices[1].max()-90
# #
# # # 获取经纬度网格
# # lons, lats = np.meshgrid(np.linspace(min_lon, max_lon, max_lon-min_lon+1), np.linspace(min_lat, max_lat, max_lat-min_lat+1))
# #
# # land = cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='white')
# #
# # # 绘制云分布频率图
# # for i in range(6):
# #     fig = plt.figure(figsize=(12, 8))
# #     ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
# #     levels = np.linspace(cloud_frequency.min(), cloud_frequency.max(), 100)
# #
# #     # 绘制颜色填充图
# #     # cf = ax.contourf(lons, lats, cloud_frequency[min_lon+180:max_lon+181, min_lat+90:max_lat+91, i].T, cmap='jet', vmin=cloud_frequency.min(), vmax=cloud_frequency.max())
# #     cf = ax.contourf(lons, lats, cloud_frequency[min_lon + 180:max_lon + 181, min_lat + 90:max_lat + 91, i].T,
# #                      levels=levels, cmap=cm.coolwarm)
# #     plt.colorbar(cf, ax=ax, orientation='horizontal')
# #
# #     # 绘制经纬度网格
# #     ax.set_xticks(np.arange(min_lon, max_lon+1, 60))
# #     ax.set_yticks(np.arange(min_lat, max_lat+1, 30))
# #     lon_formatter = LongitudeFormatter(zero_direction_label=True)
# #     lat_formatter = LatitudeFormatter()
# #     ax.xaxis.set_major_formatter(lon_formatter)
# #     ax.yaxis.set_major_formatter(lat_formatter)
# #     ax.tick_params(axis='both', labelsize=8)
# #
# #     # 绘制海岸线
# #     ax.coastlines()
# #     ax.add_feature(land)
# #
# #     # 设置图标题
# #     ax.set_title('Frequency Of  {}'.format(class_names[i][5:]))
# #
# #     # 保存图像
# #     plt.savefig('./analyse_result/{}_frequency.png'.format(class_names[i][5:]), dpi=300)
# #     plt.close()
# #
# #
# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# import numpy as np
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import matplotlib.cm as cm
#
# class_names = ['cat0_Closed MCC', 'cat1_Clustered Cumulus', 'cat2_Disorganized MCC', 'cat3_Open MCC','cat4_Solid Stratus', 'cat5_Suppressed Cumulus']
# filename = './analyse_result/20152016_frequency.npz'
# data = np.load(filename)
# cloud_frequency = data['cloud_frequency']
# cloud_count = data['cloud_count']
#
# # 获取有效数据的范围
# valid_indices = np.where(np.sum(cloud_count, axis=2) > 0)
# min_lon = valid_indices[0].min()-180
# max_lon = valid_indices[0].max()-180
# min_lat = valid_indices[1].min()-90
# max_lat = valid_indices[1].max()-90
#
# # 获取经纬度网格
# lons, lats = np.meshgrid(np.linspace(min_lon, max_lon, max_lon-min_lon+1), np.linspace(min_lat, max_lat, max_lat-min_lat+1))
#
# land = cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='white')
#
# # 创建大图和子图
# fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16, 10), subplot_kw={'projection': ccrs.PlateCarree()})
#
# # 绘制云分布频率图
# for i, ax in enumerate(axs.flat):
#     levels = np.linspace(cloud_frequency.min(), cloud_frequency.max(), 100)
#
#     # 绘制颜色填充图
#     cf = ax.contourf(lons, lats, cloud_frequency[min_lon + 180:max_lon + 181, min_lat + 90:max_lat + 91, i].T,
#                      levels=levels, cmap=cm.coolwarm)
#
#     # 绘制经纬度网格
#     ax.set_xticks(np.arange(min_lon, max_lon+1, 60))
#     ax.set_yticks(np.arange(min_lat, max_lat+1, 30))
#     lon_formatter = LongitudeFormatter(zero_direction_label=True)
#     lat_formatter = LatitudeFormatter()
#     ax.xaxis.set_major_formatter(lon_formatter)
#     ax.yaxis.set_major_formatter(lat_formatter)
#     ax.tick_params(axis='both', labelsize=8)
#
#     # 绘制海岸线
#     ax.coastlines()
#     ax.add_feature(land)
#     ax.set_title(class_names[i][5:], fontsize=12)
#
# # 添加共享的colorbar
# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
# fig.colorbar(cf, cax=cbar_ax, orientation='vertical')
# fig.suptitle("Occurrence Frequency of Marine Low Cloud Types", fontsize=16,y=0.92)
# plt.subplots_adjust(top=0.85)
#
# # 保存图像
# plt.savefig('./analyse_result/all_frequency.png', dpi=300)
# plt.close()
import math




# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# import numpy as np
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import matplotlib.cm as cm
#
# class_names = ['cat0_Closed MCC', 'cat1_Clustered Cumulus', 'cat2_Disorganized MCC', 'cat3_Open MCC(x2)','cat4_Solid Stratus', 'cat5_Suppressed Cumulus']
# filename = './analyse_result/global_frequency.npz'
# data = np.load(filename)
# cloud_frequency = data['cloud_frequency']
# cloud_count = data['cloud_count']
#
# for i in range(360):
#     for j in range(180):
#         cloud_frequency[i,j,3] = min(cloud_frequency[i,j,3] * 2, 1)
#
#
# # 获取有效数据的范围
# valid_indices = np.where(np.sum(cloud_count, axis=2) > 0)
# min_lon = valid_indices[0].min()-180
# max_lon = valid_indices[0].max()-180
# min_lat = valid_indices[1].min()-90
# max_lat = valid_indices[1].max()-90
# # min_lon = -180
# # max_lon = 180
# # min_lat = -90
# # max_lat = 90
#
# # 获取经纬度网格
# lons, lats = np.meshgrid(np.linspace(min_lon, max_lon, max_lon-min_lon+1), np.linspace(min_lat, max_lat, max_lat-min_lat+1))
#
# land = cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='white')
#
# fig = plt.figure(figsize=(12, 8))
# # ax0 = plt.subplot(321, projection=ccrs.PlateCarree())
# # ax1 = plt.subplot(324, projection=ccrs.PlateCarree())
# # ax2 = plt.subplot(322, projection=ccrs.PlateCarree())
# # ax3 = plt.subplot(323, projection=ccrs.PlateCarree())
# # ax4 = plt.subplot(325, projection=ccrs.PlateCarree())
# # ax5 = plt.subplot(326, projection=ccrs.PlateCarree())
# #
# # # 绘制云分布频率图
# # axes = [ax1, ax4, ax2, ax3, ax0, ax5]
# for i in range(6):
#     levels = np.linspace(cloud_frequency.min(), cloud_frequency.max(), 500)
#
#     # 绘制颜色填充图
#     cf = plt.contourf(lons, lats, cloud_frequency[min_lon + 180:max_lon + 181, min_lat + 90:max_lat + 91, i].T,
#                      levels=levels, cmap=cm.coolwarm)
#
#     # 绘制经纬度网格
#     plt.set_xticks(np.arange(min_lon, max_lon+1, 60))
#     plt.set_yticks(np.arange(min_lat, max_lat+1, 30))
#     lon_formatter = LongitudeFormatter(zero_direction_label=True)
#     lat_formatter = LatitudeFormatter()
#     plt.xaxis.set_major_formatter(lon_formatter)
#     plt.yaxis.set_major_formatter(lat_formatter)
#     plt.tick_params(axis='both', labelsize=10)
#
#     # 绘制海岸线
#     plt.coastlines()
#     plt.add_feature(land)
#
#     # 添加标题
#     plt.set_title("Occurrence Frequency of "+class_names[i][5:], fontsize=12)
#     # 保存图像
#     plt.savefig('./analyse_result/{}.png'.format(class_names[i][5:]), dpi=300)
#     plt.close()


#
# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# import numpy as np
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import matplotlib.cm as cm
#
# class_names = ['cat0_Closed MCC', 'cat1_Clustered Cumulus', 'cat2_Disorganized MCC', 'cat3_Open MCC(x2)','cat4_Solid Stratus', 'cat5_Suppressed Cumulus']
# filename = './analyse_result/global_frequency.npz'
# data = np.load(filename)
# cloud_frequency = data['cloud_frequency']
# cloud_count = data['cloud_count']
#
# for i in range(360):
#     for j in range(180):
#         cloud_frequency[i,j,3] = min(cloud_frequency[i,j,3] * 2, 1)
#
# # 获取有效数据的范围
# valid_indices = np.where(np.sum(cloud_count, axis=2) > 0)
# min_lon = valid_indices[0].min()-180
# max_lon = valid_indices[0].max()-180
# min_lat = valid_indices[1].min()-90
# max_lat = valid_indices[1].max()-90
#
# # 获取经纬度网格
# lons, lats = np.meshgrid(np.linspace(min_lon, max_lon, max_lon-min_lon+1), np.linspace(min_lat, max_lat, max_lat-min_lat+1))
#
# land = cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='white')
#
# # 绘制云分布频率图
# for i in range(6):
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
#     levels = np.linspace(cloud_frequency.min(), cloud_frequency.max(), 100)
#
#     # 绘制颜色填充图
#     # cf = ax.contourf(lons, lats, cloud_frequency[min_lon+180:max_lon+181, min_lat+90:max_lat+91, i].T, cmap='jet', vmin=cloud_frequency.min(), vmax=cloud_frequency.max())
#     cf = ax.contourf(lons, lats, cloud_frequency[min_lon + 180:max_lon + 181, min_lat + 90:max_lat + 91, i].T,
#                      levels=levels, cmap=cm.coolwarm)
#     plt.colorbar(cf, ax=ax, orientation='horizontal')
#
#     # 绘制经纬度网格
#     ax.set_xticks(np.arange(min_lon, max_lon+1, 60))
#     ax.set_yticks(np.arange(min_lat, max_lat+1, 30))
#     lon_formatter = LongitudeFormatter(zero_direction_label=True)
#     lat_formatter = LatitudeFormatter()
#     ax.xaxis.set_major_formatter(lon_formatter)
#     ax.yaxis.set_major_formatter(lat_formatter)
#     ax.tick_params(axis='both', labelsize=8)
#
#     # 绘制海岸线
#     ax.coastlines()
#     ax.add_feature(land)
#
#     # 设置图标题
#     ax.set_title('Occurrence Frequency of  {}'.format(class_names[i][5:]))
#
#     # 保存图像
#     plt.savefig('./analyse_result/{}_frequency.png'.format(class_names[i][5:]), dpi=300)
#     plt.close()


from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm

class_names = ['cat0_Closed MCC', 'cat1_Clustered Cumulus', 'cat2_Disorganized MCC', 'cat3_Open MCC(x2)','cat4_Solid Stratus', 'cat5_Suppressed Cumulus']
filename = './analyse_result/global_frequency.npz'
data = np.load(filename)
cloud_frequency = data['cloud_frequency']
cloud_count = data['cloud_count']

for i in range(360):
    for j in range(180):
        cloud_frequency[i,j,3] = min(cloud_frequency[i,j,3] * 2, 1)


# 获取有效数据的范围
valid_indices = np.where(np.sum(cloud_count, axis=2) > 0)
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
ax0 = plt.subplot(321, projection=ccrs.PlateCarree())
ax1 = plt.subplot(324, projection=ccrs.PlateCarree())
ax2 = plt.subplot(322, projection=ccrs.PlateCarree())
ax3 = plt.subplot(323, projection=ccrs.PlateCarree())
ax4 = plt.subplot(325, projection=ccrs.PlateCarree())
ax5 = plt.subplot(326, projection=ccrs.PlateCarree())

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
    ax.set_title(class_names[i][5:], fontsize=10)


# 调整子图之间的间距和上边距
plt.subplots_adjust(hspace=0.3, top=1.0)

# 添加共享的colorbar
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.27, 0.02, 0.4])
cf = fig.colorbar(cf, cax=cbar_ax, orientation='vertical')

fig.suptitle("Occurrence Frequency of Marine Low Cloud Types", fontsize=16, y=0.95)
plt.subplots_adjust(top=0.88)

# 保存图像
plt.savefig('./analyse_result/global_frequency.png', dpi=300)
plt.close()



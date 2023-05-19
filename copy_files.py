import shutil
import os

src_folder = '../../work10/zhangyu/MCC_classification_test_data/data/2014/0103'
dst_folder = '../../work10/zhangyu/MCC_classification_test_data/data/2014/alldata'
subfolders = [f.path for f in os.scandir(src_folder) if f.is_dir()]

for folder in subfolders:
    folder_name = os.path.basename(folder)
    dst_path = os.path.join(dst_folder, folder_name)
    shutil.copytree(folder, dst_path)

import os
import glob
import shutil

manual_dir = "../../work10/zhangyu/MCC_classification_test_data/data/2014/alldata"
data_dir = "../../work7/jhliu/MCC_classification_test_data/output/data/2014"
new_data_dir = "../../work10/zhangyu/MCC_classification_test_data/data/2014/alldata2"
subdir = "0103"
# subdir = "0406"
# figure_dir ="./delete_test/figure"
# data_dir ="./delete_test/data"
# subdir="0103"
# new_data_dir ="./data"
print("begin")
subdir_path = os.path.join(data_dir, subdir)
for subsubdir in os.listdir(subdir_path):
    subsubdir_path = os.path.join(subdir_path, subsubdir)
    if os.path.isdir(subsubdir_path):
        for mat_file in glob.glob(os.path.join(subsubdir_path, "*.mat")):
            manual_file = os.path.join(manual_dir, subsubdir, os.path.basename(os.path.basename(mat_file)))
            if os.path.exists(manual_file):
                new_dir = os.path.join(new_data_dir, subsubdir)
                shutil.copy2(mat_file, new_dir)
    print("{} is solved".format(subsubdir))
print("end")
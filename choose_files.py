import os
import glob
import shutil

# figure_dir = "../../work7/jhliu/MCC_classification_test_data/output/Figure_manual_labeled"
figure_dir = "../../work10/zhangyu/my_labels"
data_dir = "../../work7/jhliu/MCC_classification_test_data/output/data/2014"
new_data_dir = "../../work10/zhangyu/MCC_classification_test_data/data/2014"
# subdir = "0103"
subdir = "0406"
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
            jpg_file = os.path.join(figure_dir, subsubdir, os.path.splitext(os.path.basename(mat_file))[0] + "_2.jpg")
            if os.path.exists(jpg_file):
                new_dir = os.path.join(new_data_dir, subdir, subsubdir)
                shutil.copy2(mat_file, new_dir)
    print("{} is solved".format(subsubdir))
print("end")
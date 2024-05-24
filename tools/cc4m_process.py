from utils.hdfs_io import hopen, hlist_files, hexists, hcopy   
from tqdm import tqdm
import ipdb

data_path=["hdfs://haruna/home/byte_arnold_hl_vc/guolongteng.lt/data/public/vlm/coco_testset_filtered",
    "hdfs://haruna/home/byte_arnold_hl_vc/guolongteng.lt/data/public/vlm/vg_testset_filtered",
    "hdfs://haruna/home/byte_arnold_hl_vc/guolongteng.lt/data/public/vlm/sbu_bs64",
    "hdfs://haruna/home/byte_arnold_hl_vc/guolongteng.lt/data/public/vlm/cc3m_bs64"]



file_paths = hlist_files(data_path)
file_paths = [f for f in file_paths if f.find('_SUCCESS') < 0]

num_file = 0
for file_path in tqdm(file_paths):
    with hopen(file_path, 'r') as reader:
        xx = reader.readlines()
        num_file += len(xx)
print(num_file)
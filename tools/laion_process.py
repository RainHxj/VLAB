import os
import json
from dataloader import KVReader
from tqdm import tqdm
from utils.hdfs_io import hopen, hlist_files, hexists, hcopy

laion_path = "/PATH/dataset/laion400m_blip_split"

file_paths = hlist_files( laion_path.split(',') )

index_paths = [k[:-6] for k in file_paths if k.endswith("index")]

index_paths = index_paths[:25000]

with open("laion_150m_file.json", 'w') as f:
    json.dump(index_paths, f)

print(len(index_paths))

num=0
for p in tqdm(index_paths):
    reader = KVReader(p)
    num+=len(reader.list_keys())
print("all dataset: {}".format(num))


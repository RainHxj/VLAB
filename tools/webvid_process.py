from utils.hdfs_io import hopen, hlist_files, hexists, hcopy, hload_pkl
from dataloader import KVReader
import ipdb
import time
from tqdm import tqdm
from multiprocessing import Process, JoinableQueue,Value,Lock


webvid_path = "/PATH/dataset/webvid2.5M_package/webvid2.5M_pretrain" 
webvid_pkl = "/PATH/dataset/webvid2.5M_package/webvid_pretrain_anno_new.pkl" 


# webvid_path = "/PATH/datasets/webvid2.5M_package/webvid2.5M_pretrain"
# webvid_pkl = "/PATH/datasets/webvid2.5M_package//webvid_pretrain_anno_new.pkl"

keys = hload_pkl(webvid_pkl)
# ipdb.set_trace()
kv_reader = KVReader(webvid_path)
print("all_data:{}".format(len(keys)))
keys = keys[:5000]
keys = [str(k["filename"]) for k in keys]
tt=0
s_t = time.time()
for i in tqdm(range(0,len(keys),1000)):
   
    
    info =kv_reader.read_many(keys[i:i+1000])
e_t = time.time()
    
tt = (e_t-s_t)

print("evarage time{}".format(tt/len(keys)))



# filepaths = hlist_files(webvid_path.split(','))
# ipdb.set_trace()

# num=0

# index_paths = [k for k in filepaths if k.endswith("index")]

# ipdb.set_trace()

# print(len(index_paths))

# for filepath in tqdm(index_paths):

#     kv_reader = KVReader(filepath[:-6])
#     num+=len(kv_reader.list_keys())


# print("file numbers:{}".format(num))



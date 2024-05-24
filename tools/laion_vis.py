import os
import json
from dataloader import KVReader
from tqdm import tqdm
from PIL import Image
from base64 import b64decode
from utils.hdfs_io import hopen, hlist_files, hexists, hcopy
import io


laion_path = "hdfs://haruna/home/byte_ailab_litg/user/zhangxinsong/datasets/vlm/laion_aml_filtered"

file_paths = hlist_files( laion_path.split(',') )

index_paths = [k for k in file_paths if k.endswith("snappy")]


for filepath in index_paths[:1]:
    with hopen(filepath, 'r') as reader:
        i=0
        for line in reader:
            data = line.decode()
            data_item=json.loads(data)
            image_str = b64decode(data_item.get("b64_resized_binary", data_item.get("binary", data_item.get("b64_binary", data_item.get("image","")))))
            image_pil = Image.open(io.BytesIO(image_str)).convert("RGB")
            print(image_pil.size)
            image_pil.save("./save_feat/laion_aml/{}.jpg".format(i))
            i+=1
            if i>10:
                break

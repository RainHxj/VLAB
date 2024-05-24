import os
import argparse
import time 
parser = argparse.ArgumentParser()
parser.add_argument('logdir',nargs='+',type=str)
args =parser.parse_args()


output_dir = './output/tensorboard_dir'+str(time.time())
for i in args.logdir:
    if i.endswith('/'):
        i = i[:-1]
    expdir_name = i.split('/')[-1]
    src_path = os.path.join(i,'log')
    tgt_path = os.path.join(output_dir,expdir_name)
    os.makedirs(tgt_path,exist_ok=True)
    os.system(f"cp -r {src_path} {tgt_path}")

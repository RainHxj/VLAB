import os
import sys
import time
base_dir = "/PATH/VLP/output/experiments/ensemble_type"

args=sys.argv
pretrain_dir = args[1]

checkpoint_path= os.path.join(base_dir,pretrain_dir,"ckpt")
find=False
while not find:    
    checkpoints=os.listdir(checkpoint_path)
    
    for ck in checkpoints:
        if ck.startswith("model"):
            if len(args)==3:
                if str(args[2]) in ck:
                    find=True
                    print("find the model of step{}".format(str(args[2]) ))
            else:
                if "last" in ck:
                    find=True
                    print("find the model of last step")
    if find:
        pass
    else:
        print("waiting checkpint.....")
        time.sleep(60)
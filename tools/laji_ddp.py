import torch.distributed as dist 


import torch.nn as nn
import torch
import argparse
from utils.distributed import ddp_allgather, all_gather_list, any_broadcast


parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type = int, default=-1)
args =parser.parse_args()
dist.init_process_group(backend='nccl') 
n_gpu = dist.get_world_size()
local_rank = args.local_rank
device = torch.device("cuda", local_rank)
torch.cuda.set_device(local_rank)
print("ddp init")

print("device: {} n_gpu: {}, rank: {}, "
            .format(
                device, n_gpu, local_rank))


if local_rank ==0:
    a = torch.ones(1,3).cuda()
elif local_rank ==1:
    a = 2* torch.ones(2,3).cuda()



print(ddp_allgather(a))

# a=torch.ones(1,1).expand(local_rank+1,-1)
# a=a.cuda()
# print(a)

# b = ddp_allgather(a)
# print(b)


# pp=[{'video_id':2, 'caption':'he is a boy.'}]
# print(pp)

# pp = all_gather_list(pp)

# print(pp)

# pp = any_broadcast(pp,1)
# print(pp)




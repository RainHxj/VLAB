import multiprocessing as mp

import torch
from dataloader import KVReader


#index iterable batch index iterable
# chunk([4, 2, 3, 1], 2) ==> [[4, 2], [3, 1]]
def chunk(iterable, chunk_size):
    ret = []
    for record in iterable:
        ret.append(record)
        if len(ret) == chunk_size:
            yield ret
            ret = []
    if ret:
        yield ret

def get_keys(args):
    return KVReader(*args).list_keys()

class KVDataset(torch.utils.data.Dataset):
    def __init__(self, path, num_readers):
        self.path = path
        self.num_readers = num_readers
        # Use another process to avoid libhdfs.so fork issue
        with mp.Pool(1) as p:
            self.keys = p.map(get_keys, [(path, num_readers)])[0]
        # self.keys = KVReader(path).list_keys()
        # Uncomment the following lines if num_workers == 0
        # self.reader = KVReader(dataset.path, dataset.num_readers)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        return None
        index = [self.keys[i] for i in index]
        print(index)
        return self.reader.read_many(index)


class KVSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset, batch_size, num_replicas, rank, shuffle=True, drop_last=False):
        super(KVSampler, self).__init__(dataset, num_replicas= num_replicas, rank=rank, shuffle=shuffle, drop_last=drop_last)
        self.batch_size = batch_size

    def __iter__(self):
        iterable = super(KVSampler, self).__iter__()
        return chunk(iterable, self.batch_size)
    
    def __len__(self):
        return (self.num_samples+self.batch_size-1)//self.batch_size


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # Avoid "cannot pickle KVReader object" error
    dataset.reader = KVReader(dataset.path, dataset.num_readers)


if __name__ == '__main__':
    path =  "hdfs://haruna/home/byte_labcv_default/liyinan/msrvtt/all"
    batch_size = 256
    num_readers = 32
    dataset = KVDataset(path, num_readers)
    sampler = KVSampler(dataset, batch_size=batch_size, num_replicas=5,rank=4, shuffle=True)
    # See https://pytorch.org/docs/stable/data.html#disable-automatic-batching
    loader = torch.utils.data.DataLoader(dataset, batch_size=None,
                                         sampler=sampler,
                                         num_workers=4,
                                         worker_init_fn=worker_init_fn)
    for _ in range(5):  # 5 epoches
        for images in loader:
            pass # iterate through your training data
        print("Finish one epoch!")
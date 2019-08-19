# fedGpuSupport.py
# Complementary GPU support codes to Pytorch FL learning
# @Author  : wwt
# @Date    : 2019-8-18

import torch


@DeprecationWarning  # deprecated in favor of Syft built-in Cuda support
class GpuClient:
    def __init__(self, id):
        self.id = id


@DeprecationWarning  # deprecated in favor of Syft built-in Cuda support
class GpuBaseDataset:
    def __init__(self, x, y, client):
        """
        :param x: training set for a client
        :param y: test set for a client
        :param client: GpuClient object
        """
        self.x = x
        self.x = y
        self.location = client

    def __len__(self):
        return len(self.x)


@DeprecationWarning  # deprecated in favor of Syft built-in Cuda support
class GpuFedDataset:
    def __init__(self, gbd_list):
        """
        :param gbd_list: a list of GpuBaseDataset objects
        """
        self.gbd_list = gbd_list

        # count length
        self.total_datasize = 0
        for gbd in self.gbd_list:
            self.total_datasize += len(gbd)

    def __len__(self):
        return len(self.gbd_list)

    def __getitem__(self, item):
        return self.gbd_list[item]


@DeprecationWarning  # deprecated in favor of Syft built-in Cuda support
class GpuFedDataloader:
    def __init__(self, fed_set, shuffle=False, batch_size=1):
        self.fed_set = fed_set  # a GpuFedDataset
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __iter__(self):
        self.set_id = 0
        self.batch_id = 0
        self.next_set = self.fed_set[self.set_id]  # one set per client, the 1st client's data frag here
        self.next_batch_x = self.next_set.x[self.batch_id*self.batch_size:
                                            min((self.batch_id+1)*self.batch_size, len(self.next_set))]
        self.next_batch_y = self.next_set.y[self.batch_id * self.batch_size:
                                            min((self.batch_id + 1) * self.batch_size, len(self.next_set))]
        self.next_batch = (self.next_batch_x, self.next_batch_y)
        return self

    def __next__(self):  #TODO
        self.batch_id += 1
        if self.set_id > len(self.fed_set):
            raise StopIteration
        else:
            batch = self.next_batch


#test area
# c1 = GpuClient('client_0')
# c2 = GpuClient('client_1')
# gbd1 = GpuBaseDataset(torch.tensor([1,2,3]), torch.tensor([4,5,6]), c1)
# gbd2 = GpuBaseDataset(torch.tensor([5,6,7]), torch.tensor([5,6,2]), c2)
# gfd = GpuFedDataset([gbd1, gbd2])
#
# for bid, gbd in enumerate(gfd):
#     print(bid, gbd)
#     print(gbd.c_train_set, gbd.c_test_set, gbd.location.id)

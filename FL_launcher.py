# FL_launcher.py
# A script to launch Federated Learning procedure using specified FL algorithm
# @Author  : wwt
# @Date    : 2019-8-7

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets as torch_datasets
from torchvision import transforms
import copy
import sys
import gc
import time
import random
import numpy as np
import syft as sy
from sklearn import datasets as sk_datasets
from sklearn.svm import SVC
import fedGpuSupport
import fullyLocalFL
import primal_FedAvg
import semiAysnc_FedAvg
from learning_tasks import MLmodelReg, MLmodelCNN
from learning_tasks import MLmodelSVM
import utils


class EnvSettings:
    """
    Environment settings for FL
        mode: Primal FedAvg, Semi-Asynchronous FedAvg
        n_clients: # of clients/models
        n_rounds: # of global rounds
        n_epochs: # of local epochs (identical for each client)
        batch_size: size of a local batch (identical for each client)
        train_pct: training data percentage
        sf: shuffle if True
        pick_pct: percentage of clients picked in each round
        data_dist: local data size distribution, valid options include:
            ('E',None): equal-size partition, local size = total_size / n_clients
            ('N',rlt_sigma): partition with local sizes following normal distribution, mu = total_size/n_clients,
                sigma = rlt_sigma * mu
            ('X',None): partition with local sizes following exponential distribution, lambda = n_clients/total_size
        perf_dist: client performance distribution (unit: virtual time per batch), valid options include:
            ('E',None): equal performance, perf = 1 unit
            ('N',rlt_sigma): performances follow normal distribution, mu = 1, sigma = rlt_sigma * mu
            ('X',None): performances follow exponential distribution, lambda = 1/1
        crash_dist: client crash prob. distribution, valid options include:
            ('E',prob): equal probability to crash, crash_prob = prob
            ('U',(low, high)): uniform distribution between low and high
        keep_best: keep so-far best global model if True, otherwise update anyway after aggregation
        dev: running device
    """
    def __init__(self, n_clients=3, n_rounds=3, n_epochs=1, batch_size=1, train_pct=0.7, sf=False,
                 pick_pct=1.0, data_dist=None, perf_dist=None, crash_dist=None, keep_best=False, dev='cpu'):
        self.mode = None
        self.n_clients = n_clients
        self.n_rounds = n_rounds
        self.n_epochs = n_epochs
        self.batch_size = batch_size  # batch_size = -1 means full local data set as a mini-batch
        self.train_pct = train_pct
        self.test_pct = 1 - self.train_pct
        self.shuffle = sf
        self.pick_pct = pick_pct

        # client and data settings
        self.data_dist = data_dist  # data (size) distribution
        self.perf_dist = perf_dist  # client performance distribution
        self.crash_dist = crash_dist  # client crash probability distribution

        self.keep_best = keep_best

        # runtime
        if dev == 'cpu':
            self.device = torch.device('cpu')
        elif dev == 'gpu' and torch.cuda.is_available():
            self.device = torch.device('cuda:0')


class TaskSettings:
    """
    Machine learning task settings
        task_type: 'Reg' = regression, 'SVM' = (linear) support vector machine, 'CNN' = Convolutional NN
        dataset: data set name
        path: dataset file path
        in_dim: input feature dimension
        out_dim: output dimension
        optimizer: optimizer, 'SGD', 'Adam'
        loss: loss function to use: 'mse' for regression, 'svm_loss' for SVM,
        lr: learning rate
        lr_decay: learning rate decay per round
    """
    def __init__(self, task_type, dataset, path, in_dim, out_dim, optimizer='SGD', loss=None, lr=0.01, lr_decay=1.0):
        self.task_type = task_type
        self.dataset = dataset
        self.path = path
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.optimizer = optimizer
        self.loss = loss
        self.lr = lr
        self.lr_decay = lr_decay


class SimpleFedDataLoader:
    def __init__(self, fed_dataset, client2idx, batch_size, shuffle=False):
        self.fed_dataset = fed_dataset.datasets  # FedDatasets.datasets is a clientName-BaseDataset dict
        self.c_idx2data = [0 for _ in range(len(client2idx))]  # client_idx to dataset mapping list
        for k, v in self.fed_dataset.items():  # k is client name, v is the associated dataset
            self.c_idx2data[client2idx[k]] = v  # client2idx: client_name->client/model index
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_ptr = -1  # used for splitting into batches
        self.baseDataset_ptr = None

        # shuffle  TODO: untested
        if self.shuffle:
            for ds in self.c_idx2data:
                ds_size = ds.data.shape[0]
                ds.data = ds.data[torch.randperm(ds_size)]

    def __iter__(self):
        self.client_idx = 0
        self.batch_ptr = -1  # used for splitting into batches
        self.baseDataset_ptr = self.c_idx2data[self.client_idx]  # client_0's BaseDataset
        return self

    def __next__(self):
        self.batch_ptr += 1  # this batch
        # update batch location
        if self.batch_ptr * self.batch_size >= len(self.baseDataset_ptr.data):  # if no more batch for the current client
            self.batch_ptr = 0  # reset
            self.client_idx += 1
            if self.client_idx >= len(self.c_idx2data):  # no more client to iterate through
                self.stop()
            self.baseDataset_ptr = self.c_idx2data[self.client_idx]  # next client's BaseDataSet

        right_bound = len(self.baseDataset_ptr.data)
        this_batch_x = self.baseDataset_ptr.data[self.batch_ptr * self.batch_size:
                                                 min(right_bound, (self.batch_ptr + 1) * self.batch_size)]
        this_batch_y = self.baseDataset_ptr.targets[self.batch_ptr * self.batch_size:
                                                    min(right_bound, (self.batch_ptr + 1) * self.batch_size)]

        return this_batch_x, this_batch_y

    def stop(self):
        raise StopIteration


def save_KddCup99_tcpdump_tcp_tofile(fpath):
    """
    Fetch (from sklearn.datasets) the KddCup99 tcpdump dataset, extract tcp samples, and save to local as csv
    :param fpath: local file path to save the dataset
    :return: KddCup99 dataset, tcp-protocol samples only
    """
    xy = utils.fetch_KddCup99_10pct_tcpdump(return_X_y=False)
    np.savetxt(fpath, xy, delimiter=',', fmt='%.6f',
               header='duration, src_bytes, dst_bytes, land, urgent, hot, #failed_login, '
                      'logged_in, #compromised, root_shell, su_attempted, #root, #file_creations, #shells, '
                      '#access_files, is_guest_login, count, srv_cnt, serror_rate, srv_serror_rate, rerror_rate, '
                      'srv_rerror_rate, same_srv_rate, diff_srv_rate, srv_diff_host_rate, dst_host_cnt,'
                      'dst_host_srv_cnt, dst_host_same_srv_rate, dst_host_diff_srv_rate, dst_host_same_src_port_rt,'
                      'dst_host_srv_diff_host_rate, dst_host_serror_rate, dst_host_srv_serror_rate, '
                      'dst_host_rerror_rate, dst_host_srv_rerror_rate, label')


def init_syft_clients(nc, hook):
    """
    # Create clients and a client-index map
    :param nc:  # of clients
    :param hook:  Syft hook
    :return: clients list, and a client-index map
    """
    clients = []
    c_name2idx = {}
    for i in range(nc):
        clients.append(sy.VirtualWorker(hook, id='client_' + str(i)))
        c_name2idx['client_' + str(i)] = i  # client i with model i
    return clients, c_name2idx


def clear_syft_memory_and_reinit(fed_data_train, fed_data_test, env_cfg, hook):
    """
    Clear syft memory leakage by deleting and re-initializing all client objects and corresponding data sets
    :param fed_data_train: FederatedDataSet of training
    :param fed_data_test: FederatedDataSet of test
    :param env_cfg: environment config
    :param hook: Pysyft hook
    :return: new clients, FederatedDataLoader of training and test data
    """
    # init new client objects
    clients, c_name2idx = init_syft_clients(env_cfg.n_clients, hook)

    # rebuild FederatedDatasets and Loaders
    # train set
    for c, d in fed_data_train.datasets.items():
        d.get()
        d.send(clients[c_name2idx[c]])  # bind data to newly-built clients
    for c, d in fed_data_test.datasets.items():
        d.get()
        d.send(clients[c_name2idx[c]])
    fed_loader_train = sy.FederatedDataLoader(fed_data_train, shuffle=env_cfg.shuffle, batch_size=env_cfg.batch_size)
    fed_loader_test = sy.FederatedDataLoader(fed_data_test, shuffle=env_cfg.shuffle, batch_size=env_cfg.batch_size)

    return clients, fed_loader_train, fed_loader_test



@DeprecationWarning  # deprecated in favor of Syft built-in Cuda support
def init_cuda_clients(nc):
    """
    # Create clients and a client-index map
    :param nc:  # of cuda clients
    :return: clients list, and a client-index map
    """
    clients = []
    cm_map = {}
    for i in range(nc):
        clients.append(fedGpuSupport.GpuClient(id='client_' + str(i)))
        cm_map['client_' + str(i)] = i  # client i with model i
    return clients, cm_map


def init_models(env_cfg, task_cfg):
    """
    Initialize models as per the settings of FL and machine learning task
    :param env_cfg:
    :param task_cfg:
    :return: models
    """
    models = []
    dev = env_cfg.device
    # have to transiently set default tensor type to cuda.float, otherwise model.to(dev) fails on GPU
    if dev.type == 'cuda':
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # instantiate models, one per client
    for i in range(env_cfg.n_clients):
        if task_cfg.task_type == 'Reg':
            models.append(MLmodelReg(in_features=task_cfg.in_dim, out_features=task_cfg.out_dim).to(dev))
        elif task_cfg.task_type == 'SVM':
            models.append(MLmodelSVM(in_features=task_cfg.in_dim).to(dev))
        elif task_cfg.task_type == 'CNN':
            models.append(MLmodelCNN(classes=10).to(dev))

    torch.set_default_tensor_type('torch.FloatTensor')
    return models


def generate_clients_perf(env_cfg):
    """
    Generate a series of client performance values (in virtual time unit) following specified distribution
    :param env_cfg: environment config file
    :return: a list of client's performance, measured in virtual unit
    """
    n_clients = env_cfg.n_clients
    perf_vec = None
    # Case 1: Equal performance
    if env_cfg.perf_dist[0] == 'E':  # ('E', None)
        perf_vec = [1.0 for _ in range(n_clients)]

    # Case 2: eXponential distribution of performance
    elif env_cfg.perf_dist[0] == 'X':  # ('X', None), lambda = 1/1, mean = 1
        perf_vec = [random.expovariate(1.0) for _ in range(n_clients)]

    # Case 3: Normal distribution of performance
    elif env_cfg.perf_dist[0] == 'N':  # ('N', rlt_sigma), mu = 1, sigma = rlt_sigma * mu
        perf_vec = [0.0 for _ in range(n_clients)]
        for i in range(n_clients):
            perf_vec[i] = random.gauss(1.0, env_cfg.perf_dist[1] * 1.0)
            while perf_vec[i] <= 0:  # in case of negative
                perf_vec[i] = random.gauss(1.0, env_cfg.perf_dist[1] * 1.0)
    else:
        print('Error> Invalid client performance distribution option')
        exit(0)

    return perf_vec


def generate_clients_crash_prob(env_cfg):
    """
    Generate a series of probability that the corresponding client would crash (including device and network down)
    :param env_cfg: environment config file
    :return: a list of client's crash probability, measured in virtual unit
    """
    n_clients = env_cfg.n_clients
    prob_vec = None
    # Case 1: Equal prob
    if env_cfg.crash_dist[0] == 'E':  # ('E', prob)
        prob_vec = [env_cfg.crash_dist[1] for _ in range(n_clients)]

    # Case 2: uniform distribution of crashing prob
    elif env_cfg.crash_dist[0] == 'U':  # ('U', (low, high))
        low = env_cfg.crash_dist[1][0]
        high = env_cfg.crash_dist[1][1]
        # check
        if low < 0 or high < 0 or low > 1 or high > 1 or low >= high:
            print('Error> Invalid crash prob interval')
            exit(0)
        prob_vec = [random.uniform(low, high) for _ in range(n_clients)]
    else:
        print('Error> Invalid crash prob distribution option')
        exit(0)

    return prob_vec


def generate_crash_trace(env_cfg, clients_crash_prob_vec):
    """
    Generate a crash trace (length=# of rounds) for simulation,
    making every FA algorithm shares the same trace for fairness
    :param env_cfg: env config
    :param clients_crash_prob_vec: client crash prob. vector
    :return: crash trace as a list of lists, and a progress trace
    """
    crash_trace = []
    progress_trace = []
    for r in range(env_cfg.n_rounds):
        crash_ids = []  # crashed ones this round
        progress = [1.0 for _ in range(env_cfg.n_clients)]  # 1.0 denotes well progressed
        for c_id in range(env_cfg.n_clients):
            rand = random.random()
            if rand <= clients_crash_prob_vec[c_id]:  # crash
                crash_ids.append(c_id)
                progress[c_id] = rand / clients_crash_prob_vec[c_id]  # progress made before crash

        crash_trace.append(crash_ids)
        progress_trace.append(progress)

    return crash_trace, progress_trace


def main():
    hook = sy.TorchHook(torch)  # hook PyTorch with PySyft to support Federated Learning
    ''' Boston housing regression settings (3s per epoch)'''
    # env_cfg = EnvSettings(n_clients=5, n_rounds=100, n_epochs=3, batch_size=5, train_pct=0.7, sf=False,
    #                       pick_pct=0.5, data_dist=('N', 0.3), perf_dist=('X', None), crash_dist=('E', float(argv[1])),
    #                       dev='cpu', keep_best=False)
    # task_cfg = TaskSettings(task_type='Reg', dataset='Boston', path='data/boston_housing.csv',
    #                         in_dim=12, out_dim=1, optimizer='SGD', loss='mse', lr=1e-4, lr_decay=1.0)
    ''' KddCup99 tcpdump SVM classification settings (~15s per epoch on CPU, optimized)'''
    env_cfg = EnvSettings(n_clients=500, n_rounds=100, n_epochs=5, batch_size=100, train_pct=0.7, sf=False,
                          pick_pct=0.5, data_dist=('N', 0.3), perf_dist=('X', None), crash_dist=('E', float(sys.argv[1])),
                          dev='cpu', keep_best=False)
    task_cfg = TaskSettings(task_type='SVM', dataset='tcpdump99', path='data/kddcup99_tcp.csv',
                            in_dim=35, out_dim=1, optimizer='SGD', loss='svmLoss', lr=1e-2, lr_decay=1.0)
    ''' MNIST digits classification task settings (3~4min per epoch on CPU, 1.5min on GPU)'''
    # env_cfg = EnvSettings(n_clients=100, n_rounds=30, n_epochs=5, batch_size=20, train_pct=6.0/7.0, sf=True,
    #                       pick_pct=0.5, data_dist=('E', None), perf_dist=('X', None), crash_dist=('E', 0.5),
    #                       dev='gpu', keep_best=False)
    # task_cfg = TaskSettings(task_type='CNN', dataset='mnist', path='data/MNIST/',
    #                         in_dim=None, out_dim=None, optimizer='SGD', loss='nllLoss', lr=1e-3, lr_decay=1.0)

    utils.show_settings(env_cfg, task_cfg, detail=False, detail_info=None)

    # load data
    if task_cfg.dataset == 'Boston':
        data = np.loadtxt(task_cfg.path, delimiter=',', skiprows=1)
        data = utils.normalize(data)
        data_merged = True
    elif task_cfg.dataset == 'tcpdump99':
        data = np.loadtxt(task_cfg.path, delimiter=',', skiprows=1)
        data = utils.normalize(data, expt=-1)  # normalize features but not labels (+1/-1 for SVM)
        data_merged = True
    elif task_cfg.dataset == 'mnist':
        # ref: https://github.com/pytorch/examples/blob/master/mnist/main.py
        mnist_train = torch_datasets.MNIST('data/mnist/', train=True, download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        mnist_test = torch_datasets.MNIST('data/mnist/', train=False, download=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        data_train_x = mnist_train.data.view(-1, 1, 28, 28).float()
        data_train_y = mnist_train.targets.long()
        data_test_x = mnist_test.data.view(-1, 1, 28, 28).float()
        data_test_y = mnist_test.targets.long()

        train_data_size = len(data_train_x)
        test_data_size = len(data_test_x)
        data_size = train_data_size + test_data_size
        data_merged = False
    else:
        print('E> Invalid dataset specified')
        exit(-1)
    # partition into train/test set, for Boston and Tcpdump data
    if data_merged:
        data_size = len(data)
        train_data_size = int(data_size * env_cfg.train_pct)
        test_data_size = data_size - train_data_size
        data = torch.tensor(data).float()
        data_train_x = data[0:train_data_size, 0:task_cfg.in_dim]  # training data, x
        data_train_y = data[0:train_data_size, task_cfg.out_dim * -1:].reshape(-1, task_cfg.out_dim)  # training data, y
        data_test_x = data[train_data_size:, 0:task_cfg.in_dim]  # test data following, x
        data_test_y = data[train_data_size:, task_cfg.out_dim * -1:].reshape(-1, task_cfg.out_dim)  # test data, x

    clients, c_name2idx = init_syft_clients(env_cfg.n_clients, hook)  # create clients and a client-index map
    fed_data_train, fed_data_test, client_shard_sizes = utils.get_FL_datasets(data_train_x, data_train_y,
                                                                              data_test_x, data_test_y,
                                                                              env_cfg, clients)
    # pseudo distributed data loaders, by Syft
    # fed_loader_train = sy.FederatedDataLoader(fed_data_train, shuffle=env_cfg.shuffle, batch_size=env_cfg.batch_size)
    # fed_loader_test = sy.FederatedDataLoader(fed_data_test, shuffle=env_cfg.shuffle, batch_size=env_cfg.batch_size)
    fed_loader_train = SimpleFedDataLoader(fed_data_train, c_name2idx,
                                           batch_size=env_cfg.batch_size, shuffle=env_cfg.shuffle)
    fed_loader_test = SimpleFedDataLoader(fed_data_test, c_name2idx,
                                          batch_size=env_cfg.batch_size, shuffle=env_cfg.shuffle)
    print('> %d clients data shards (data_dist = %s):' % (env_cfg.n_clients, env_cfg.data_dist[0]), client_shard_sizes)

    # prepare simulation
    # clients performance
    clients_perf_vec = generate_clients_perf(env_cfg)
    print('> Clients perf vec:', clients_perf_vec)
    # max round interval is reached when any crash occurs
    max_round_interval = max(
        env_cfg.n_epochs / env_cfg.batch_size * np.array(client_shard_sizes) / np.array(clients_perf_vec))
    # client crash probability
    clients_crash_prob_vec = generate_clients_crash_prob(env_cfg)
    print('> Clients crash prob. vec:', clients_crash_prob_vec)
    # crash trace simulation
    crash_trace, progress_trace = generate_crash_trace(env_cfg, clients_crash_prob_vec)

    # specify learning task, for Fully Local training
    # models = init_models(env_cfg, task_cfg)
    # print('> Launching Fully Local FL...')
    # # run FL with Fully local training
    # env_cfg.mode = 'Fully Local'
    # best_model, best_rd, final_loss = fullyLocalFL. \
    #     run_fullyLocal(env_cfg, task_cfg, models, c_name2idx, data_size, fed_loader_train, fed_loader_test,
    #                    client_shard_sizes, clients_perf_vec, clients_crash_prob_vec, crash_trace, progress_trace,
    #                    max_round_interval)
    #
    # # reinitialize, for FedAvg
    # models = init_models(env_cfg, task_cfg)
    # print('> Launching FedAvg FL...')
    # # run FL with FedAvg
    # env_cfg.mode = 'Primal FedAvg'
    # best_model, best_rd, final_loss = primal_FedAvg. \
    # run_FL(env_cfg, task_cfg, models, c_name2idx, data_size, fed_loader_train, fed_loader_test, client_shard_sizes,
    #        clients_perf_vec, clients_crash_prob_vec, crash_trace, progress_trace, max_round_interval)

    # reinitialize, for SAFA
    models = init_models(env_cfg, task_cfg)
    print('> Launching SAFA-FL (lag tolerance = %d)' % int(sys.argv[2]))
    # run FL with SAFA
    env_cfg.mode = 'Semi-Async. FedAvg'
    best_model, best_rd, final_loss = semiAysnc_FedAvg. \
        run_FL_SAFA(env_cfg, task_cfg, models, c_name2idx, data_size, fed_loader_train, fed_loader_test,
                    client_shard_sizes, clients_perf_vec, clients_crash_prob_vec, crash_trace, max_round_interval,
                    lag_t=int(sys.argv[2]))


# test area
# m = MLmodelCNN(10).to(torch.device('cuda'))
# print(next(m.parameters()).is_cuda)
# exit(0)

if __name__ == '__main__':
    main()


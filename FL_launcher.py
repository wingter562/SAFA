# FL_launcher.py
# A script to launch Federated Learning procedure using specified FL algorithm
# @Author  : wwt
# @Date    : 2019-8-7

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import sys
import os
import random
import numpy as np
import syft as sy
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from learning_tasks import MLmodelReg
import utils
import primal_FedAvg
import semiAysnc_FedAvg


class EnvSettings:
    """
    Environment settings for FL
        mode: Primal FedAvg, Semi-Asynchronous FedAvg
        n_clients: # of clients/models
        n_rounds: # of global rounds
        n_epochs: # of local epochs (identical for each client)
        batch_size: size of a local batch (identical for each client)
        train_pct: training data percentage
        subset_pct: percentage of clients picked in each round
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
    def __init__(self, n_clients=3, n_rounds=3, n_epochs=1, batch_size=1, train_pct=0.7, subset_pct=1.0,
                 data_dist=None, perf_dist=None, crash_dist=None, keep_best=False, dev='cpu'):
        self.mode = None
        self.n_clients = n_clients
        self.n_rounds = n_rounds
        self.n_epochs = n_epochs
        self.batch_size = batch_size  # batch_size = -1 means full local data set as a mini-batch
        self.train_pct = train_pct
        self.test_pct = 1 - self.train_pct
        self.subset_pct = subset_pct

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
    """
    def __init__(self, task_type, dataset, path, in_dim, out_dim, optimizer='SGD', loss=None, lr=0.01):
        self.task_type = task_type
        self.dataset = dataset
        self.path = path
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.optimizer = optimizer
        self.loss = loss
        self.lr = lr


def init_models(env_cfg, task_cfg):
    """
    Initialize models as per the settings of FL and machine learning task
    :param env_cfg:
    :param task_cfg:
    :return: models
    """
    models = []
    for i in range(env_cfg.n_clients):
        if task_cfg.task_type == 'Reg':
            models.append(MLmodelReg(in_features=task_cfg.in_dim, out_features=task_cfg.out_dim))

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


# test area
# X, y = datasets.fetch_kddcup99(subset=None, data_home='data/', shuffle=False, percent10=True, return_X_y=True)
# y = np.reshape(y,(-1 ,1))
# xy = np.concatenate((X, y), axis=1)
# print(xy)
# n_xy = utils.filter_matrix(xy, -1, b'normal.')
# print('total=', len(xy), 'normal=', len(n_xy))  # overall data view
#
# # tcp data
# tcp_xy = utils.filter_matrix(xy, 1, b'tcp')
# tcp_n_xy = utils.filter_matrix(tcp_xy, -1, b'normal.')
# print(tcp_xy)
# print(type(tcp_xy[0][0]))
# print('total_tcp=', len(tcp_xy), 'normal_tcp=', len(tcp_n_xy))  # tcp data view
x, y = utils.fetch_KddCup99_10pct_tcpdump(return_X_y=True)
x = utils.normalize(x)  # SVM can't converge with normalization
print(x)
print('total=', len(x))
clf = SVC(kernel='linear')
clf.fit(x, y)
decs = clf.decision_function(x)
neg = 0
precision = 0
for k in range(len(y)):
    print(decs[k], y[k])

exit(0)

if __name__ == '__main__':
    # initialization
    hook = sy.TorchHook(torch)  # hook PyTorch with PySyft to support Federated Learning
    ''' Boston housing regression settings'''
    env_cfg = EnvSettings(n_clients=5, n_rounds=20, n_epochs=2, batch_size=1, train_pct=0.7, subset_pct=0.4,
                          data_dist=('X', None), perf_dist=('X', None), crash_dist=('E', 0.5),
                          dev='cpu', keep_best=False)
    task_cfg = TaskSettings(task_type='Reg', dataset='Boston', path='data/boston_housing.csv',
                            in_dim=12, out_dim=1, optimizer='SGD', loss='mse', lr=1e-2)  # for Boston reg
    task_cfg = TaskSettings(task_type='Reg', dataset='Boston', path='data/boston_housing.csv',
                            in_dim=12, out_dim=1, optimizer='SGD', loss='mse', lr=1e-2)  # for KddCup99 tcpdump

    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    utils.show_settings(env_cfg, task_cfg, detail=False, detail_info=None)

    # create clients and client-model-mapping
    clients = []
    cm_map = {}
    for i in range(env_cfg.n_clients):
        clients.append(sy.VirtualWorker(hook, id='client_'+str(i)))
        cm_map['client_'+str(i)] = i  # client i with model i

    # load data
    data = np.loadtxt(task_cfg.path, delimiter=',', skiprows=1)
    data_size = len(data)
    train_data_size = int(data_size * env_cfg.train_pct)
    test_data_size = data_size - train_data_size
    data = utils.normalize(data)

    data = torch.tensor(data).float()
    # data_x = data[:, 0:task_cfg.in_dim]
    # data_y = data[:, task_cfg.out_dim * -1:].reshape(-1, task_cfg.out_dim)  # reshape labels to a column
    data_train_x = data[0:train_data_size, 0:task_cfg.in_dim]  # training data, x
    data_train_y = data[0:train_data_size, task_cfg.out_dim*-1:].reshape(-1, task_cfg.out_dim)  # training data , y
    data_test_x = data[train_data_size:, 0:task_cfg.in_dim]  # test data, x
    data_test_y = data[train_data_size:, task_cfg.out_dim*-1:].reshape(-1, task_cfg.out_dim)  # test data, x

    fed_data_train, fed_data_test, client_shard_sizes = utils.get_FL_datasets(data_train_x, data_train_y,
                                                                              data_test_x, data_test_y,
                                                                              env_cfg, clients)
    print('> %d clients data shards (data_dist = %s):' % (env_cfg.n_clients, env_cfg.data_dist[0]), client_shard_sizes)

    # pseudo distributed data loaders
    fed_loader_train = sy.FederatedDataLoader(fed_data_train, shuffle=False, batch_size=env_cfg.batch_size)
    fed_loader_test = sy.FederatedDataLoader(fed_data_test, shuffle=False, batch_size=env_cfg.batch_size)

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

    # specify learning task
    models = init_models(env_cfg, task_cfg)
    print('> Launching FL...')
    # run FL with FedAvg
    env_cfg.mode = 'Primal FedAvg'
    best_model, best_rd, final_loss = primal_FedAvg.\
        run_FL(env_cfg, task_cfg, models, cm_map, data_size, fed_loader_train, fed_loader_test, client_shard_sizes,
               clients_perf_vec, clients_crash_prob_vec, crash_trace, progress_trace, max_round_interval)

    # reinitialize
    models = init_models(env_cfg, task_cfg)
    print('> Launching SAFA-FL...')
    # run FL with SAFA
    env_cfg.mode = 'Semi-Async. FedAvg'
    best_model, best_rd, final_loss = semiAysnc_FedAvg.\
        run_FL_SAFA(env_cfg, task_cfg, models, cm_map, data_size, fed_loader_train, fed_loader_test,
                    client_shard_sizes, clients_perf_vec, clients_crash_prob_vec, crash_trace, max_round_interval,
                    lag_t=2)


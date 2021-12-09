# FedCS_Nishio.py
# Pytorch+PySyft implementation of the Client-Selective FedCS algorithm for Federated Learning,
# proposed by:
# [] Nishio, T., & Yonetani, R. (2019, May). Client selection for federated learning with heterogeneous resources in
#   mobile edge. In ICC 2019-2019 IEEE International Conference on Communications (ICC) (pp. 1-7). IEEE.
# A special strategy for client selection is implemented to filter out cross-rounders beforehand.
# @Author  : wwt
# @Date    : Jan. 10, 2020

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import sys
import os
import math
import random
import numpy as np
# import syft as sy
import matplotlib.pyplot as plt
from learning_tasks import MLmodelReg, MLmodelCNN
from learning_tasks import MLmodelSVM
from learning_tasks import svmLoss
import utils
import FLLocalSupport as FLSup


def train(models, picked_ids, env_cfg, cm_map, fdl, task_cfg, last_loss_rep, verbose=True):
    """
    Execute one EPOCH of training process of any machine learning model on all clients
    :param models: a list of model prototypes corresponding to clients
    :param picked_ids: selected client indices for local training
    :param env_cfg: environment configurations
    :param cm_map: the client-model map, as a dict
    :param fdl: an instance of FederatedDataLoader
    :param task_cfg: task training settings, including learning rate, optimizer, etc.
    :param last_loss_rep: loss reports in last run
    :param verbose: display batch progress or not.
    :return: epoch training loss of each client, batch-summed
    """
    dev = env_cfg.device
    if len(picked_ids) == 0:  # no training happens
        return last_loss_rep
    # extract settings
    n_models = env_cfg.n_clients  # # of clients
    # initialize loss report, keep loss tracks for idlers, clear those for participants
    client_train_loss_vec = last_loss_rep
    for id in picked_ids:
        client_train_loss_vec[id] = 0.0
    # Disable printing
    if not verbose:
        sys.stdout = open(os.devnull, 'w')
    # initialize training mode
    for m in range(n_models):
        models[m].train()

    # Define loss based on task
    if task_cfg.loss == 'mse':  # regression task
        loss_func = nn.MSELoss(reduction='mean')  # cannot back-propagate with 'reduction=sum'
    elif task_cfg.loss == 'svmLoss':  # SVM task
        loss_func = svmLoss(reduction='mean')  # self-defined loss, have to use default reduction 'mean'
    elif task_cfg.loss == 'nllLoss':  # CNN mnist task
        loss_func = nn.NLLLoss()

    # one optimizer for each model (re-instantiate optimizers to clear any possible momentum
    optimizers = []
    for i in range(n_models):
        if task_cfg.optimizer == 'SGD':
            optimizers.append(optim.SGD(models[i].parameters(), lr=task_cfg.lr))
        elif task_cfg.optimizer == 'Adam':
            optimizers.append(optim.Adam(models[i].parameters(), lr=task_cfg.lr))
        else:
            print('Err> Invalid optimizer %s specified' % task_cfg.optimizer)

        # begin an epoch of training
    for batch_id, (inputs, labels, client) in enumerate(fdl):
        inputs, labels = inputs.to(dev), labels.to(dev)  # data to device
        model_id = cm_map[client.id]  # locate the right model index
        # neglect non-participants
        if model_id not in picked_ids:
            continue
        # mini-batch GD
        print('\n> Batch #', batch_id, 'on', client.id)
        print('>   model_id = ', model_id)
        model = models[model_id]
        optimizer = optimizers[model_id]
        # gradient descent procedure
        optimizer.zero_grad()
        y_hat = model(inputs)
        # loss
        loss = loss_func(y_hat, labels)
        loss.backward()
        # weights update
        optimizer.step()

        # display
        print('>   batch loss = ', loss.item())  # avg. batch loss
        client_train_loss_vec[model_id] += loss.item()*len(inputs)  # sum up

    # Restore printing
    if not verbose:
        sys.stdout = sys.__stdout__
    # end an epoch-training - all clients have traversed their own local data once
    return client_train_loss_vec


def local_test(models, task_cfg, env_cfg, picked_ids, n_models, cm_map, fdl, last_loss_rep):
    """
    Evaluate client models locally and return a list of loss/error
    :param models: a list of model prototypes corresponding to clients
    :param task_cfg: task configurations
    :param env_cfg: environment configurations
    :param picked_ids: selected client indices for local training
    :param n_models: # of models, i.e., clients
    :param cm_map: the client-model map, as a dict
    :param fdl: an instance of FederatedDataLoader
    :param last_loss_rep: loss reports in last run
    :return: epoch test loss of each client, batch-summed
    """
    if not picked_ids:  # no training happened
        return last_loss_rep
    dev = env_cfg.device
    # initialize loss report, keep loss tracks for idlers, clear those for participants
    client_test_loss_vec = last_loss_rep
    for id in picked_ids:
        client_test_loss_vec[id] = 0.0
    # Define loss based on task
    if task_cfg.loss == 'mse':  # regression task
        loss_func = nn.MSELoss(reduction='sum')
    elif task_cfg.loss == 'svmLoss':  # SVM task
        loss_func = svmLoss(reduction='sum')
    elif task_cfg.loss == 'nllLoss':  # CNN mnist task
        loss_func = nn.NLLLoss(reduction='sum')

    # initialize evaluation mode
    for m in range(n_models):
        models[m].eval()
    # local evaluation, batch-wise
    acc = 0.0
    count = 0.0
    with torch.no_grad():
        for batch_id, (inputs, labels, client) in enumerate(fdl):
            inputs, labels = inputs.to(dev), labels.to(dev)  # data to device
            model_id = cm_map[client.id]  # locate the right model index
            # neglect non-participants
            if model_id not in picked_ids:
                continue
            model = models[model_id]
            # inference
            y_hat = model(inputs)
            # loss
            loss = loss_func(y_hat, labels)
            client_test_loss_vec[model_id] += loss.item()
            # accuracy
            b_acc, b_cnt = utils.batch_sum_accuracy(y_hat, labels, task_cfg.loss)
            acc += b_acc
            count += b_cnt

    print('> acc = %.6f' % (acc/count))
    return client_test_loss_vec


def global_test(model, task_cfg, env_cfg, cm_map, fdl):
    """
    Testing the aggregated global model by averaging its error on each local data
    :param model: the global model
    :param task_cfg: task configuration
    :param env_cfg: env configuration
    :param cm_map: the client-model map, as a dict
    :param fdl: FederatedDataLoader
    :return: global model's loss on each client (as a vector), accuracy
    """
    dev = env_cfg.device
    test_sum_loss_vec = [0 for _ in range(env_cfg.n_clients)]
    # Define loss based on task
    if task_cfg.loss == 'mse':  # regression task
        loss_func = nn.MSELoss(reduction='sum')
    elif task_cfg.loss == 'svmLoss':  # SVM task
        loss_func = svmLoss(reduction='sum')
    elif task_cfg.loss == 'nllLoss':  # CNN mnist task
        loss_func = nn.NLLLoss(reduction='sum')
        is_cnn = True

    print('> global test')
    # initialize evaluation mode
    model.eval()
    acc = 0.0
    count = 0
    # local evaluation, batch-wise
    for batch_id, (inputs, labels, client) in enumerate(fdl):
        inputs, labels = inputs.to(dev), labels.to(dev)  # data to device
        model_id = cm_map[client.id]
        # inference
        y_hat = model(inputs)
        # loss
        loss = loss_func(y_hat, labels)
        test_sum_loss_vec[model_id] += loss.item()
        # compute accuracy
        b_acc, b_cnt = utils.batch_sum_accuracy(y_hat, labels, task_cfg.loss)
        acc += b_acc
        count += b_cnt

    print('>   acc = %.6f' % (acc / count))
    return test_sum_loss_vec, acc/count


def distribute_models(global_model, models, make_ids):
    """
    Distribute the global model
    :param global_model: aggregated global model
    :param models: local models
    :param make_ids: ids of clients that will replace their local models with the global one
    :return:
    """
    for id in make_ids:
        models[id] = copy.deepcopy(global_model)


def aggregate(models, local_shards_sizes, data_size):
    """
    The function implements aggregation step (FedAvg), i.e., w = sum_k(n_k/n * w_k)
    :param models: a list of local models
    :param picked_ids: selected client indices for local training
    :param local_shards_sizes: a list of local data sizes, aligned with the orders of local models, say clients.
    :param data_size: total data size
    :return: a global model
    """
    print('>   Aggregating (FedAvg)...')
    global_model_params = []
    client_weights_vec = np.array(local_shards_sizes) / data_size  # client weights (i.e., n_k / n)
    for m in range(len(models)):  # for each local model
        p_pointer = 0
        for param in models[m].parameters():
            if m == 0:  # use the first model to shape the global one
                global_model_params.append(param.data * client_weights_vec[m])
            else:
                global_model_params[p_pointer] += param.data * client_weights_vec[m]  # sum up the corresponding param
            p_pointer += 1

    # create a global model instance for return
    global_model = copy.deepcopy(models[0])
    p_pointer = 0
    for param in global_model.parameters():
        param.data.copy_(global_model_params[p_pointer])
        p_pointer += 1

    return global_model


def get_cross_rounders(clients_est_round_T_train, max_round_interval):
    cross_rounder_ids = []
    for c_id in range(len(clients_est_round_T_train)):
        if clients_est_round_T_train[c_id] > max_round_interval:
            cross_rounder_ids.append(c_id)
    return cross_rounder_ids


def sort_ids_by_esttime_asc(id_list, est_time_list):
    """
    Sort a list of client ids according to their est. finish time, in an ascending ordered
    :param id_list: a list of client ids to sort
    :param est_time_list: full list of all clients' est. finish time
    :return: sorted id_list
    """
    # make use of a map
    cp_map = {}  # client-perf-map
    for id in id_list:
        cp_map[id] = est_time_list[id]  # build the map with corresponding perf
    # sort by perf
    sorted_map = sorted(cp_map.items(), key=lambda x: x[1], reverse=False)  # a sorted list of tuples
    sorted_id_list = [sorted_map[i][0] for i in range(len(id_list))]  # extract the ids into a list
    return sorted_id_list


def FedCS_selection(selected_ids, c_bw, model_size, clients_est_round_T_train, res_lim):
    """
    FedCS's strategy of client (re-)selection, i.e., excluding cross-rounders
    :param selected_ids: clients selected randomly with C
    :param c_bw: client bandwidth
    :param model_size: model size
    :param clients_est_round_T_train: estimated client local training time
    :param res_lim: client response limit
    :return: client ids after re-selection
    """
    pick_ids = []
    # get client ids sorted by est. finish time
    sorted_selected_ids = sort_ids_by_esttime_asc(selected_ids, clients_est_round_T_train)
    # pick as many as clients until response deadline
    T = 0.0  # est. time of the client finishing as the last one
    for cid in sorted_selected_ids:
        T = clients_est_round_T_train[cid] + model_size/c_bw  # depends on the slowest
        if T <= res_lim:  # still in time
            pick_ids.append(cid)
        else:
            break
    return pick_ids


def run_FedCS(env_cfg, task_cfg, glob_model, cm_map, data_size, fed_loader_train, fed_loader_test, client_shard_sizes,
              clients_perf_vec, clients_crash_prob_vec, crash_trace, progress_trace, response_time_limit):
    """
    Primal implementation of FedAvg for FL
    :param env_cfg:
    :param task_cfg:
    :param glob_model:
    :param cm_map:
    :param data_size:
    :param fed_loader_train:
    :param fed_loader_test:
    :param client_shard_sizes:
    :param clients_perf_vec:
    :param clients_crash_prob_vec:
    :param crash_trace:
    :param progress_trace:
    :param response_time_limit:
    :return:
    """
    # init
    global_model = glob_model  # the global model
    models = [None for _ in range(env_cfg.n_clients)]  # local models
    client_ids = list(range(env_cfg.n_clients))
    distribute_models(global_model, models, client_ids)  # init local models

    # traces
    reporting_train_loss_vec = [0 for _ in range(env_cfg.n_clients)]
    reporting_test_loss_vec = [0 for _ in range(env_cfg.n_clients)]
    epoch_train_trace = []
    epoch_test_trace = []
    make_trace = []
    pick_trace = []
    round_trace = []
    acc_trace = []

    # Global event handler
    event_handler = FLSup.EventHandler(['time', 'T_dist'])
    # Local counters
    # 1. Local timers - record work time of each client
    client_timers = [0.01 for _ in range(env_cfg.n_clients)]  # totally
    client_comm_timers = [0.0 for _ in range(env_cfg.n_clients)]  # comm. totally
    # 2. Futile counters - progression (i,e, work time) in vain caused by local crashes
    clients_est_round_T_train = np.array(client_shard_sizes) / env_cfg.batch_size * env_cfg.n_epochs / np.array(
        clients_perf_vec)
    cross_rounders = get_cross_rounders(clients_est_round_T_train, response_time_limit)
    client_futile_timers = [0.01 for _ in range(env_cfg.n_clients)]  # totally
    eu_count = 0.0  # effective updates count
    sync_count = 0.0  # synchronization count
    # Best loss (global)
    best_rd = -1
    best_loss = float('inf')
    best_acc = -1.0
    best_model = None

    # begin training: global rounds
    for rd in range(env_cfg.n_rounds):
        print('\n> Round #%d' % rd)
        # reset timers
        client_round_timers = [0.0 for _ in range(env_cfg.n_clients)]  # local time in current round
        client_round_comm_timers = [0.0 for _ in range(env_cfg.n_clients)]  # local comm. time in current round
        # randomly pick a specified fraction of clients to launch training
        n_picks = math.ceil(env_cfg.n_clients * env_cfg.pick_pct)
        selected_ids = random.sample(range(env_cfg.n_clients), n_picks)  # initial selection
        selected_ids.sort()
        # simulate device or network failure
        crash_ids = crash_trace[rd]
        client_round_progress = progress_trace[rd]
        # FedCS client re-selection
        selected_ids = FedCS_selection(selected_ids, env_cfg.bw_set[0], task_cfg.model_size, clients_est_round_T_train,
                                       res_lim=response_time_limit)
        pick_trace.append(selected_ids)  # tracing
        print('> Clients selected(FedCS): ', selected_ids)
        # some selected may crash
        submit_ids = [c_id for c_id in selected_ids if c_id not in crash_ids and c_id not in cross_rounders]
        # tracing
        make_trace.append(submit_ids)
        eu_count += len(submit_ids)  # count effective updates
        print('> Clients crashed: ', crash_ids)

        # distributing step: broadcast (for non-selected clients, just update their cache entry on the cloud
        # so as to have all models share initialization upon aggregation for loss reduction)
        print('>   @Cloud> distributing global model')
        distribute_models(global_model, models, client_ids)
        sync_count += len(selected_ids)  # count sync. overheads

        # Local training step
        for epo in range(env_cfg.n_epochs):  # local epochs (same # of epochs for each client)
            print('\n> @Devices> local epoch #%d' % epo)
            # invoke mini-batch training on selected clients, from the 2nd epoch
            if rd + epo == 0:  # 1st epoch all-in to get start points
                bak_make_ids = copy.deepcopy(submit_ids)
                submit_ids = list(range(env_cfg.n_clients))
            elif rd == 0 and epo == 1:  # resume
                submit_ids = bak_make_ids
            reporting_train_loss_vec = train(models, submit_ids, env_cfg, cm_map, fed_loader_train, task_cfg,
                                             reporting_train_loss_vec, verbose=False)
            # add to trace
            epoch_train_trace.append(
                np.array(reporting_train_loss_vec) / (np.array(client_shard_sizes) * env_cfg.train_pct))
            # print('>   @Devices> %d clients train loss vector this epoch:' % env_cfg.n_clients,
            #       np.array(reporting_train_loss_vec) / (np.array(client_shard_sizes) * env_cfg.train_pct))
            # local test reports
            reporting_test_loss_vec = local_test(models, task_cfg, env_cfg, submit_ids, env_cfg.n_clients, cm_map,
                                                 fed_loader_test, reporting_test_loss_vec)
            # add to trace
            epoch_test_trace.append(
                np.array(reporting_test_loss_vec) / (np.array(client_shard_sizes) * env_cfg.test_pct))
            # print('>   %d @Devices> clients test loss vector this epoch:' % env_cfg.n_clients,
            #       np.array(reporting_test_loss_vec) / (np.array(client_shard_sizes) * env_cfg.test_pct))

        # Aggregation step
        print('\n> Aggregation step (Round #%d)' % rd)
        global_model = aggregate(models, client_shard_sizes, data_size)

        # Reporting phase: distributed test of the global model
        post_aggre_loss_vec, acc = global_test(global_model, task_cfg, env_cfg, cm_map, fed_loader_test)
        print('>   @Devices> post-aggregation loss reports  = ',
              np.array(post_aggre_loss_vec) / ((np.array(client_shard_sizes)) * env_cfg.test_pct))
        # overall loss, i.e., objective (1) in McMahan's paper
        overall_loss = np.array(post_aggre_loss_vec).sum() / (data_size*env_cfg.test_pct)
        # update so-far best
        if overall_loss < best_loss:
            best_loss = overall_loss
            best_acc = acc
            best_model = global_model
            best_rd = rd
        if env_cfg.keep_best:  # if to keep best
            global_model = best_model
            overall_loss = best_loss
            acc = best_acc
        print('>   @Cloud> post-aggregation loss avg = ', overall_loss)
        round_trace.append(overall_loss)
        acc_trace.append(acc)

        # update timers
        for c_id in range(env_cfg.n_clients):
            if c_id in selected_ids:  # only compute timers for the selected
                # client_local_round time T(k) = T_download(k) + T_train(k) + T_upload(k), where
                #   T_comm(k) = T_download(k) + T_upload(k) = 2* model_size / bw_k
                #   T_train = number of batches / client performance
                T_comm = 2*task_cfg.model_size / env_cfg.bw_set[0]
                T_train = client_shard_sizes[c_id] / env_cfg.batch_size * env_cfg.n_epochs / clients_perf_vec[c_id]
                print('train time and comm. time locally:', T_train, T_comm)
                client_round_timers[c_id] = min(response_time_limit, T_comm + T_train)  # including comm. and training
                client_round_comm_timers[c_id] = T_comm  # comm. is part of the run time
                client_timers[c_id] += client_round_timers[c_id]  # sum up
                client_comm_timers[c_id] += client_round_comm_timers[c_id]  # sum up
                if c_id in crash_ids:  # failed clients
                    client_futile_timers[c_id] += client_round_timers[c_id] * client_round_progress[c_id]
                    # client_round_timers[c_id] = response_time_limit  # no response
        dist_time = task_cfg.model_size * sync_count / env_cfg.bw_set[1]  # T_disk = model_size * N_sync / BW
        # round_response_time = min(response_time_limit, max(client_round_timers))
        # global_timer += dist_time + round_response_time
        # global_T_dist_timer += dist_time

        # Event updates
        event_handler.add_parallel('time', client_round_timers, reduce='max')
        event_handler.add_sequential('time', dist_time)
        event_handler.add_sequential('T_dist', dist_time)

        print('> Round client run time:', client_round_timers)  # round estimated finish time
        print('> Round client progress:', client_round_progress)  # round actual progress at last
        task_cfg.lr *= task_cfg.lr_decay  # learning rate decay

    # Stats
    global_timer = event_handler.get_state('time')
    global_T_dist_timer = event_handler.get_state('T_dist')
    # Traces
    print('> Train trace:')
    utils.show_epoch_trace(epoch_train_trace, env_cfg.n_clients, plotting=False, cols=1)  # training trace
    print('> Test trace:')
    utils.show_epoch_trace(epoch_test_trace, env_cfg.n_clients, plotting=False, cols=1)
    print('> Round trace:')
    utils.show_round_trace(round_trace, plotting=env_cfg.showplot, title_='Primal FedAvg')

    # display timers
    print('\n> Experiment stats')
    print('> Clients run time:', client_timers)
    print('> Clients futile run time:', client_futile_timers)
    futile_pcts = (np.array(client_futile_timers) / np.array(client_timers)).tolist()
    print('> Clients futile percent (avg.=%.3f):' % np.mean(futile_pcts), futile_pcts)
    eu_ratio = eu_count/env_cfg.n_rounds/env_cfg.n_clients
    print('> EUR:', eu_ratio)
    sync_ratio = sync_count/env_cfg.n_rounds/env_cfg.n_clients
    sync_ratio = sync_ratio/(env_cfg.pick_pct+0.1)*env_cfg.pick_pct if sync_ratio > env_cfg.pick_pct else sync_ratio
    print('> SR:', sync_ratio)
    print('> Total time consumption:', global_timer)
    print('> Total distribution time (T_dist):', global_T_dist_timer)
    print('> Loss = %.6f/at Round %d:' % (best_loss, best_rd))

    # Logging
    detail_env = (client_shard_sizes, clients_perf_vec, clients_crash_prob_vec)
    utils.log_stats('stats/exp_log.txt', env_cfg, task_cfg, detail_env, epoch_train_trace, epoch_test_trace,
                    round_trace, acc_trace, make_trace, pick_trace, crash_trace, None,
                    client_timers, client_futile_timers, global_timer, global_T_dist_timer, eu_ratio, sync_ratio, 0.0,
                    best_rd, best_loss, extra_args=None, log_loss_traces=False)

    return best_model, best_rd, best_loss

# # test area
# selected = [0,1,2,3,4,5,6,7]
# est_ts = [0,10,20,30,40,50,60,70]
# res_lim = 100
# print(FedCS_selection(selected, 100, 10, est_ts, res_lim))
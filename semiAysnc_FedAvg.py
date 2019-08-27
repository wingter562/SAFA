# primal_FedAvg.py
# Pytorch+PySyft implementation of the semi-asynchronous Federated Averaging (SAFA) algorithm for Federated Learning,
# proposed by: wwt
# @Author  : wwt
# @Date    : 2019-8-1

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
from learning_tasks import MLmodelReg, svmLoss, MLmodelSVM, MLmodelCNN
import utils



def select_clients_CFCFM(make_ids, last_round_pick_ids, clients_perf_vec, quota):
    """
    Select clients to aggregate their models according to Compensatory First-Come-First-Merge principle.
    :param make_ids: ids of clients finishing their training this round
    :param last_round_pick_ids: ids of clients picked last round, low priority in this round
    :param clients_perf_vec: clients' performances
    :param quota: number of clients to draw this round
    :return: ids of selected clients
    """
    picks = []
    cp_make_ids = copy.deepcopy(make_ids)  # keep the original copy of make_ids
    # priority to clients not picked last round ("Compensatory")
    for id in make_ids:
        if id not in last_round_pick_ids:  # not picked last round, including crashed and undrafted
            picks.append(id)  # pick it
            cp_make_ids.remove(id)
            if len(picks) >= quota:  # quota met
                return picks

    # the rest are picked by order of performance ("FCFM")
    cp_map = {}  # client-perf-map
    for id in cp_make_ids:
        cp_map[id] = clients_perf_vec[id]  # build the map
    # sort by performance
    sorted_map = sorted(cp_map.items(), key=lambda x: x[1], reverse=True)

    # start picking, FCFM
    for i in range(min(quota - len(picks), len(cp_map))):
        picks.append(sorted_map[i][0])

    return picks


def train(models, picked_ids, env_cfg, cm_map, fdl, task_cfg, last_loss_rep, verbose=True):
    """
    Execute one EPOCH of training process of any machine learning model on all clients
    :param models: a list of model prototypes corresponding to clients
    :param picked_ids: participating client indices for local training
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
    for batch_id, (inputs, labels) in enumerate(fdl):
        inputs, labels = inputs.to(dev), labels.to(dev)  # data to device
        client = inputs.location  # training location (i.e.,the client) recorded by Syft, an object
        model_id = cm_map[client.id]  # locate the right model index
        # neglect non-participants
        if model_id not in picked_ids:
            continue
        # mini-batch GD
        print('\n> Batch #', batch_id, 'on', inputs.location.id)
        print('>   model_id = ', model_id)
        model = models[model_id]
        optimizer = optimizers[model_id]
        # send out for local training
        model.send(client)
        # gradient descent procedure
        optimizer.zero_grad()
        y_hat = model(inputs)
        # loss
        loss = loss_func(y_hat, labels)
        loss.backward()
        # weights
        optimizer.step()

        # display
        loss_ = loss.get()  # batch-total loss
        print('>   batch loss = ', loss_.item())  # avg. batch loss
        if np.isnan(loss_.item()):
            print(y_hat.get(), labels.get())
        client_train_loss_vec[model_id] += loss_.detach().item()*len(inputs)  # sum up
        # get back before next send
        models[model_id] = model.get()

    # Restore printing
    if not verbose:
        sys.stdout = sys.__stdout__
    # end an epoch-training - all clients have traversed their own local data once
    return client_train_loss_vec


def local_test(models, picked_ids, task_cfg, env_cfg,  cm_map, fdl, last_loss_rep):
    """
    Evaluate client models locally and return a list of loss/error
    :param models: a list of model prototypes corresponding to clients
    :param task_cfg: task configurations
    :param env_cfg: environment configurations
    :param picked_ids: selected client indices for local training
    :param cm_map: the client-model map, as a dict
    :param fdl: an instance of FederatedDataLoader
    :param last_loss_rep: loss reports in last run
    :return: epoch test loss of each client, batch-summed
    """
    if not picked_ids:  # no training happens
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
        is_cnn = True

    # initialize evaluation mode
    for m in range(env_cfg.n_clients):
        models[m].eval()
    # local evaluation, batch-wise
    acc = 0.0
    count = 0.0
    with torch.no_grad():
        for batch_id, (inputs, labels) in enumerate(fdl):
            inputs, labels = inputs.to(dev), labels.to(dev)  # data to device
            client = inputs.location  # training location (i.e.,the client) recorded by Syft
            model_id = cm_map[client.id]  # locate the right model index
            # neglect non-participants
            if model_id not in picked_ids:
                continue
            model = models[model_id]
            # send out for local test
            model.send(client)
            # inference
            y_hat = model(inputs)
            # loss
            loss = loss_func(y_hat, labels)
            client_test_loss_vec[model_id] += loss.get().detach().item()
            models[model_id] = model.get()  # get model back
            # accuracy
            b_acc, b_cnt = utils.batch_sum_accuracy(y_hat, labels, task_cfg.loss)
            acc += b_acc
            count += b_cnt

    print('> acc = %.6f' % (acc / count))
    return client_test_loss_vec


def global_test(model, task_cfg, env_cfg, cm_map, fdl):
    """
    Testing the aggregated global model by averaging its error on each local data
    :param model: the global model
    :param task_cfg: task configurations
    :param env_cfg: environment configurations
    :param cm_map: the client-model map, as a dict
    :param fdl: FederatedDataLoader
    :return: global model's loss on each client (as a vector), accuracy
    """
    dev = env_cfg.device
    test_sum_loss_vec = [0 for i in range(env_cfg.n_clients)]
    # Define loss based on task
    if task_cfg.loss == 'mse':  # regression task
        loss_func = nn.MSELoss(reduction='sum')
    elif task_cfg.loss == 'svmLoss':  # SVM task
        loss_func = svmLoss(reduction='sum')
    elif task_cfg.loss == 'nllLoss':  # CNN mnist task
        loss_func = nn.NLLLoss(reduction='sum')
        is_cnn = True

    # initialize evaluation mode
    print('> global test')
    model.eval()
    # local evaluation, batch-wise
    acc = 0.0
    count = 0
    for batch_id, (inputs, labels) in enumerate(fdl):
        inputs, labels = inputs.to(dev), labels.to(dev)  # data to device
        client = inputs.location  # training location (i.e.,the client) recorded by Syft
        model_id = cm_map[client.id]
        # send model to data
        model.send(client)
        # inference
        y_hat = model(inputs)
        # loss
        loss = loss_func(y_hat, labels)
        test_sum_loss_vec[model_id] += loss.get().detach().item()
        model.get()  # get model back
        # compute accuracy
        b_acc, b_cnt = utils.batch_sum_accuracy(y_hat, labels, task_cfg.loss)
        acc += b_acc
        count += b_cnt

    print('>   acc = %.6f' % (acc/count))
    return test_sum_loss_vec, acc/count


def update_cloud_cache(cache, models, the_ids):
    """
    Update the model cache residing on the cloud, it contains the latest non-aggregated models
    :param cache: the model cache
    :param models: latest local model set containing picked, undrafted and deprecated models
    :param the_ids: ids of clients to update cache
    :return:
    """
    # use deepcopy to decouple cloud cache and local models
    for id in the_ids:
        cache[id] = copy.deepcopy(models[id])


def update_cloud_cache_deprecated(cache, global_model, straggler_ids):
    """
    Update entries of those clients lagging too much behind with the latest global model
    :param cache: the model cache
    :param global_model: the aggregated global model
    :param straggler_ids: ids of clients to update cache
    :return:
    """
    # use deepcopy to decouple cloud cache and local models
    for id in straggler_ids:
        cache[id] = copy.deepcopy(global_model)


def get_versions(ids, versions):
    """
    Show versions of specified clients, as a dict
    :param ids: clients ids
    :param versions: versions vector of all clients
    :return:
    """
    cv_map = {}
    for id in ids:
        cv_map[id] = versions[id]

    return cv_map


def update_versions(versions, make_ids, rd):
    """
    Update versions of local models that successfully perform training in the current round
    :param versions: version vector
    :param make_ids: well-progressed clients ids
    :param rd: round number
    :return: na
    """
    for id in make_ids:
        versions[id] = rd


def version_filter(versions, the_ids, base_v, lag_tolerant=1):
    """
    Apply a filter to client ids by checking their model versions. If the version is lagged behind the latest version
    (i.e., round number) by a number > lag_tolarant, then it will be filtered out.
    :param versions: client versions vector
    :param the_ids: client ids to check version
    :param base_v: latest base version
    :param lag_tolerant: maximum tolerance of version lag
    :return: non-straggler ids, deprecated ids
    """
    good_ids = []
    deprecated_ids = []
    for id in the_ids:
        if base_v - versions[id] <= lag_tolerant:
            good_ids.append(id)
        else:  # stragglers
            deprecated_ids.append(id)

    return good_ids, deprecated_ids


def distribute_to_local(global_model, models, make_ids):
    """
    Distribute the global model to local clients
    :param global_model: aggregated global model
    :param models: local models
    :param make_ids: ids of clients that will replace their local models with the global one
    :return:
    """
    for id in make_ids:
        models[id] = copy.deepcopy(global_model)


def safa_aggregate(models, local_shards_sizes, data_size):
    """
    The function implements aggregation step (Semi-Async. FedAvg), allowing cross-round com
    :param models: a list of local models
    :param picked_ids: selected client indices for local training
    :param local_shards_sizes: a list of local data sizes, aligned with the orders of local models, say clients.
    :param data_size: total data size
    :return: a global model
    """
    print('>   Aggregating (SAFA)...')
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


def run_FL_SAFA(env_cfg, task_cfg, models, cm_map, data_size, fed_loader_train, fed_loader_test, client_shard_sizes,
                clients_perf_vec, clients_crash_prob_vec, crash_trace, max_round_interval, lag_t=1):
    """
    Run FL with SAFA algorithm
    :param env_cfg: environment config
    :param task_cfg: task config
    :param models: local models
    :param cm_map: client-model mapping
    :param data_size: total data size
    :param fed_loader_train: federated training set
    :param fed_loader_test: federated test set
    :param client_shard_sizes: sizes of clients' shards
    :param clients_perf_vec: performance values of clients
    :param clients_crash_prob_vec: crash probs of clients
    :param crash_trace: simulated crash trace
    :param max_round_interval: maximum round interval
    :param lag_t: tolerance of lag
    :return:
    """
    # traces
    reporting_train_loss_vec = [0.0 for _ in range(env_cfg.n_clients)]
    reporting_test_loss_vec = [0.0 for _ in range(env_cfg.n_clients)]
    versions = [-1 for _ in range(env_cfg.n_clients)]
    epoch_train_trace = []
    epoch_test_trace = []
    pick_trace = []
    make_trace = []
    undrafted_trace = []
    deprecated_trace = []
    round_trace = []
    acc_trace = []

    # Counters
    # 1. Global timers, 1 unit = # of batches / client performance, where performance is defined as batch efficiency
    global_timer = 0.0
    # 2. Client timers - record work time of each client
    client_timers = [0.0 for _ in range(env_cfg.n_clients)]  # totally
    client_round_timers = []  # in current round
    picked_client_round_timers = []  # in current round, we need to wait them
    # 3. Futile counters - progression (i,e, work time) in vain caused by denial
    picked_ids = []
    client_futile_timers = [0.0 for _ in range(env_cfg.n_clients)]  # totally
    # 4. best loss (global)
    best_rd = -1
    best_loss = float('inf')
    best_model = None

    # global cache, storing models to merge before aggregation and latest models after aggregation.
    cache = copy.deepcopy(models)  # initial
    # begin training: global rounds
    for rd in range(env_cfg.n_rounds):
        print('\n> Round #%d' % rd)
        # reset timers
        client_round_timers = [0.0 for _ in range(env_cfg.n_clients)]
        picked_client_round_timers = [0.0 for _ in range(env_cfg.n_clients)]
        # randomly pick a specified fraction of clients to launch training
        quota = int(env_cfg.n_clients * env_cfg.pick_pct)  # the quota
        # simulate device or network failure
        crash_ids = crash_trace[rd]
        make_ids = [c_id for c_id in range(env_cfg.n_clients) if c_id not in crash_ids]
        # compensatory first-come-first-merge selection, last-round picks are considered low priority
        picked_ids = select_clients_CFCFM(make_ids, picked_ids, clients_perf_vec, quota)
        # also record well-progressed but undrafted ones
        undrafted_ids = [c_id for c_id in make_ids if c_id not in picked_ids]
        # tracing
        make_trace.append(make_ids)
        pick_trace.append(picked_ids)
        undrafted_trace.append(undrafted_ids)
        print('> Clients crashed: ', crash_ids)
        print('> Clients undrafted: ', undrafted_ids)
        print('> Clients picked: ', picked_ids)  # first-come-first-merge

        # Local training step
        for epo in range(env_cfg.n_epochs):  # local epochs (same # of epochs for each client)
            print('\n> local epoch #%d' % epo)
            # invoke mini-batch training on selected clients, from the 2nd epoch
            if rd + epo == 0:  # 1st epoch all-in to get start points
                bak_make_ids = copy.deepcopy(make_ids)
                make_ids = list(range(env_cfg.n_clients))
            elif rd == 0 and epo == 1:  # resume
                cache = copy.deepcopy(models)  # as the very 1st epoch changes everything
                make_ids = bak_make_ids
            reporting_train_loss_vec = train(models, make_ids, env_cfg, cm_map, fed_loader_train, task_cfg,
                                             reporting_train_loss_vec, verbose=False)
            # add to trace
            epoch_train_trace.append(
                np.array(reporting_train_loss_vec) / (np.array(client_shard_sizes) * env_cfg.train_pct))
            print('>   %d clients train loss vector this epoch:' % env_cfg.n_clients,
                  np.array(reporting_train_loss_vec) / (np.array(client_shard_sizes) * env_cfg.train_pct))
            # local test reports
            reporting_test_loss_vec = local_test(models, make_ids, task_cfg, env_cfg, cm_map, fed_loader_test,
                                                 reporting_test_loss_vec)
            # add to trace
            epoch_test_trace.append(
                np.array(reporting_test_loss_vec) / (np.array(client_shard_sizes) * env_cfg.test_pct))
            print('>   %d clients test loss vector this epoch:' % env_cfg.n_clients,
                  np.array(reporting_test_loss_vec) / (np.array(client_shard_sizes) * env_cfg.test_pct))

        # Aggregation step
        # discriminative update cloud cache and aggregate
        # case 1: update cache from picked clients, before aggregation
        good_picked_ids, deprecated_ids1 = version_filter(versions, picked_ids, rd-1, lag_tolerant=lag_t)  # by version
        update_cloud_cache(cache, models, good_picked_ids)
        print('\n> Aggregation step (Round #%d)' % rd)
        global_model = safa_aggregate(cache, client_shard_sizes, data_size)  # aggregation
        # case 2: update cache from undrafted clients, after aggregation
        good_undrafted_ids, deprecated_ids2 = version_filter(versions, undrafted_ids, rd-1, lag_tolerant=lag_t)
        update_cloud_cache(cache, models, good_undrafted_ids)
        # case 3: deny well-progressed but deprecated clients, replace these cache entries with the global model
        deprecated_ids = deprecated_ids1 + deprecated_ids2
        deprecated_trace.append(deprecated_ids)
        print('\n> Deprecated clients (models denied):', get_versions(deprecated_ids, versions))
        update_cloud_cache_deprecated(cache, global_model, deprecated_ids)  # replace deprecated with the latest global
        # versioning
        update_versions(versions, make_ids, rd)
        print('\n> Versions updated:', versions)

        # Reporting phase: distributed test of the global model
        post_aggre_loss_vec, acc = global_test(global_model, task_cfg, env_cfg, cm_map, fed_loader_test)
        print('>   post-aggregation loss reports  = ',
              np.array(post_aggre_loss_vec) / ((np.array(client_shard_sizes)) * env_cfg.test_pct))
        # overall loss, i.e., objective (1) in McMahan's paper
        overall_loss = np.array(post_aggre_loss_vec).sum() / (data_size*env_cfg.test_pct)
        # update so-far best
        if overall_loss < best_loss:
            best_loss = overall_loss
            best_model = global_model
            best_rd = rd
        if env_cfg.keep_best:  # if to keep best
            global_model = best_model
            overall_loss = best_loss
        print('>   post-aggregation loss avg = ', overall_loss)
        round_trace.append(overall_loss)
        acc_trace.append(acc)

        # dispatch global model back to clients
        print('>   Dispatching global model to well-progressed clients')
        distribute_to_local(global_model, models, make_ids)

        # update timers
        for c_id in range(env_cfg.n_clients):
            if c_id in make_ids:
                # time = # of batches run / perf
                client_round_timers[c_id] = env_cfg.n_epochs / env_cfg.batch_size * client_shard_sizes[c_id] \
                                            / clients_perf_vec[c_id]
                client_timers[c_id] += client_round_timers[c_id]
                if c_id in picked_ids:
                    picked_client_round_timers[c_id] = client_round_timers[c_id]  # we need to wait the picked
                if c_id in deprecated_ids:  # denied clients
                    client_futile_timers[c_id] += client_round_timers[c_id]
        round_time = max_round_interval if len(make_ids) < quota else max(picked_client_round_timers)
        global_timer += round_time

        print('> Round client run time:', client_round_timers)  # round estimated finish time

    # Traces
    print('> Train trace:')
    utils.show_epoch_trace(epoch_train_trace, env_cfg.n_clients, plotting=False, cols=1)  # training trace
    print('> Test trace:')
    utils.show_epoch_trace(epoch_test_trace, env_cfg.n_clients, plotting=False, cols=1)
    print('> Round trace:')
    utils.show_round_trace(round_trace, plotting=True, title_='SAFA')

    # display timers
    print('\n> Experiment stats')
    print('> Clients run time:', client_timers)
    print('> Clients futile run time:', client_futile_timers)
    futile_pcts = (np.array(client_futile_timers) / np.array(client_timers)).tolist()
    print('> Clients futile percent (avg.=%.3f):' % np.mean(futile_pcts), futile_pcts)
    print('> Total time consumption:', global_timer)
    print('> Loss = %.6f/at Round %d:' % (best_loss,best_rd))

    # Logging
    detail_env = (client_shard_sizes, clients_perf_vec, clients_crash_prob_vec)
    utils.log_stats('stats/exp_log.txt', env_cfg, task_cfg, detail_env, epoch_train_trace, epoch_test_trace,
                    round_trace, acc_trace, make_trace, pick_trace, crash_trace, deprecated_trace,
                    client_timers, client_futile_timers, global_timer,
                    best_rd, best_loss, extra_args={'lag_tolerance': lag_t}, log_loss_traces=False)

    return best_model, best_rd, best_loss


# test area
# picked_ids = select_clients_CFCFM([0, 1, 3, 4], [0],[0.9,1.2,3.2,0.4,0.8],3)
# print(picked_ids)
# exit(0)





import copy
import numpy as np
import torch
import torchvision
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F

from scipy.spatial.distance import pdist, squareform
import scipy.spatial as sp


from beam_greedy_search import get_beam_kept_gks

import os
import sys
import copy
import random

import collections
import logging
logger = logging.getLogger("Test")



def get_permutation_matrix(X, X_labels):
    n_clusters = len(set(X_labels))
    index_label_LUT = [[i, v] for i, v in enumerate(X_labels)]
    index_label_LUT.sort(key=lambda t: (t[1], t[0]))

    permutation_matrix = np.zeros((X.shape[0], X.shape[0]))
    for order, (original_index, _) in enumerate(index_label_LUT):
        permutation_matrix[order, original_index] = 1

    return permutation_matrix



def prune_kernels(current_block, conv_num, new_conv, new_weights, old_out_channels, pruning_rate = 0.4375, n_clusters = 8, pruning_strategy = 'smooth_cost_beam', setting = None):



    weights = new_weights.cuda()
    conv_prune_mask = torch.zeros(weights.data.size()).cuda()
    d_out = old_out_channels // n_clusters


    conv_preserved_kernel_index = []
    for i in range(n_clusters):
        wi = weights[i*d_out:(i+1)*d_out, :, :, :]
        wi_copy = copy.deepcopy(wi)
        _, wi_copy_in_channels, _, _ = wi_copy.data.size()
        wi_copy = wi_copy.data.cpu().numpy()
        wi = wi.transpose(1,0).contiguous()
        in_channels, out_channels, kernel_size, kernel_size = wi.data.size()
        # print('layer shape',in_channels, out_channels, kernel_size, kernel_size )
        wi = wi.view(in_channels, out_channels*kernel_size*kernel_size)
        wi = wi.data.cpu().numpy()


        # Greedy Pruning
        if pruning_strategy == 'greedy':

            sim_matrix = 1 - sp.distance.cdist(wi, wi, metric = setting['prune_params']['metric'])
            di = select_kernel_from_group(sim_matrix,
                        pruning_rate = pruning_rate,
                        eval_pruned_kernel_relationship = setting['prune_params']['eval_pruned_kernel_relationship'],
                        eval_kept_kernel_ratio = setting['prune_params']['eval_kept_kernel_ratio'],
                        cost_balancer = setting['prune_params']['cost_balancer']
                ) # kept indices


        elif pruning_strategy == 'L2':
            wi_tensor = torch.from_numpy(wi)
            norm_list = torch.norm(wi_tensor, p=2, dim=1)
            norm_list = [(i, v) for i, v in enumerate(norm_list)]
            norm_list.sort(reverse = True, key = lambda t: t[1])
            number_of_grouped_kernel_to_keep = int((1 - pruning_rate) * len(norm_list))

            di = [i for i, v in norm_list[:number_of_grouped_kernel_to_keep]]


        elif pruning_strategy == 'smooth_kernel':

            wi_tensor = torch.from_numpy(wi)
            smoothness_list = get_kernel_smoothness(wi_tensor, in_channels, old_out_channels // n_clusters)
            smoothness_list = [(i, v) for i, v in enumerate(smoothness_list)]
            smoothness_list.sort(key = lambda t: t[1])
            number_of_grouped_kernel_to_keep = int((1 - pruning_rate) * len(smoothness_list))

            di = [i for i, v in smoothness_list[:number_of_grouped_kernel_to_keep]]

        elif pruning_strategy == 'mean_kernel':
            wi_tensor = torch.from_numpy(wi)
            mean_list = torch.mean(wi_tensor, dim=1)
            mean_list = [(i, abs(v)) for i, v in enumerate(mean_list)]
            mean_list.sort(reverse = True, key = lambda t: t[1])

            number_of_grouped_kernel_to_keep = int((1 - pruning_rate) * len(mean_list))

            di = [i for i, v in mean_list[:number_of_grouped_kernel_to_keep]]


        elif pruning_strategy == 'greedy_smooth_kernel':
            sim_matrix = 1 - sp.distance.cdist(wi, wi, metric = setting['prune_params']['metric'])

            wi_tensor = torch.from_numpy(wi)
            smoothness_list = get_kernel_smoothness(wi_tensor, in_channels, old_out_channels // n_clusters)
            smoothness_list = [(i, v) for i, v in enumerate(smoothness_list)]


            di = select_kernel_from_group(sim_matrix,
                        pruning_rate = pruning_rate,
                        pruning_strategy = pruning_strategy,
                        eval_pruned_kernel_relationship = setting['prune_params']['eval_pruned_kernel_relationship'],
                        eval_kept_kernel_ratio = setting['prune_params']['eval_kept_kernel_ratio'],
                        cost_balancer = setting['prune_params']['cost_balancer'],
                        smoothness_list = smoothness_list
                ) # kept indices

        elif pruning_strategy == 'kmeanspp_smooth_kernel':
            sim_matrix = 1 - sp.distance.cdist(wi, wi, metric = setting['prune_params']['metric'])

            wi_tensor = torch.from_numpy(wi)
            smoothness_list = get_kernel_smoothness(wi_tensor, in_channels, old_out_channels // n_clusters)
            smoothness_list = [(i, v) for i, v in enumerate(smoothness_list)]

            di = select_kernel_from_group(sim_matrix,
                        pruning_rate = pruning_rate,
                        pruning_strategy = pruning_strategy,
                        eval_pruned_kernel_relationship = setting['prune_params']['eval_pruned_kernel_relationship'],
                        eval_kept_kernel_ratio = setting['prune_params']['eval_kept_kernel_ratio'],
                        cost_balancer = setting['prune_params']['cost_balancer'],
                        smoothness_list = smoothness_list,
                        kmeanspp_flag = True
                )

        elif pruning_strategy == 'kmeanspp_cost':
            sim_matrix = 1 - sp.distance.cdist(wi, wi, metric = setting['prune_params']['metric'])

            di = select_kernel_from_group(sim_matrix,
                        pruning_rate = pruning_rate,
                        pruning_strategy = pruning_strategy,
                        eval_pruned_kernel_relationship = setting['prune_params']['eval_pruned_kernel_relationship'],
                        eval_kept_kernel_ratio = setting['prune_params']['eval_kept_kernel_ratio'],
                        cost_balancer = setting['prune_params']['cost_balancer'],
                        kmeanspp_flag = True
                )

        elif pruning_strategy == 'debug':
            logger.info(f'\tKernel Size: {kernel_size}')

            di = [i for i in range(len(wi))]
            random.shuffle(di)
            di = di[:int(len(wi) * (1 - pruning_rate))]


        elif pruning_strategy == 'smooth_cost_beam' or pruning_strategy == 'smooth_cost_beam_smooth_select':
            sim_matrix = 1 - sp.distance.cdist(wi, wi, metric = setting['prune_params']['metric'])
            wi_tensor = torch.from_numpy(wi)

            if kernel_size == 3:
                smoothness_list = get_kernel_smoothness(wi_tensor, in_channels, old_out_channels // n_clusters, p = 2)
                smoothness_list = [(i, v) for i, v in enumerate(smoothness_list)]


                # di = get_smooth_beam_kept_gks(sim_matrix,
                di = get_beam_kept_gks(sim_matrix,
                            pruning_rate = pruning_rate,
                            pruning_strategy = pruning_strategy,
                            smoothness_list = smoothness_list,
                            inner_outer_balancer = setting['prune_params']['inner_outer_balancer'],
                            cost_smooth_balancer = setting['prune_params']['cost_smooth_balancer'],
                            eval_kept_kernel_number = setting['prune_params']['eval_kept_kernel_number'],
                            beam_width = setting['prune_params']['beam_width'],
                            smoothness_check_step = setting['prune_params']['smoothness_check_step'],
                            eval_outer_cost_during_beam_search = setting['prune_params']['eval_outer_cost_during_beam_search']
                        ) # kept indices

                # di = [i for i in range(len(wi))]
                # random.shuffle(di)
                # di = di[:int(len(wi) * (1 - pruning_rate))]

            elif kernel_size == 1:

                di = get_greedy_kept_gks(sim_matrix,
                            pruning_rate = pruning_rate,
                            pruning_strategy = 'greedy',
                            eval_kept_kernel_number = setting['prune_params']['eval_kept_kernel_number'],
                            inner_outer_balancer = setting['prune_params']['inner_outer_balancer'],
                            show_analysis = False)

            else:
                logger.error(f'kernel_size = {kernel_size} is not supported.')


            # di = get_smooth_beam_greedy_search_kept_gks(sim_matrix,
            #             pruning_rate = pruning_rate,
            #             pruning_strategy = pruning_strategy,
            #             smoothness_list = smoothness_list,
            #             inner_outer_balancer = setting['prune_params']['inner_outer_balancer'],
            #             cost_smooth_balancer = setting['prune_params']['cost_smooth_balancer'],
            #             eval_kept_kernel_number = setting['prune_params']['eval_kept_kernel_number'],
            #             beam_width = setting['prune_params']['beam_width'],
            #             smoothness_check_step = setting['prune_params']['smoothness_check_step'],
            #         ) # kept indices
            # print(f"di: {di}")

            # if sorted(di_brute) != sorted(di_optim):
            #     print(f'di_brute: {sorted(di_optim)}')
            #     print(f'di_optim: {sorted(di_brute)}')
            #     sys.exit()
            # else:
            #     di = di_brute

            # sorted_smoothness_list = sorted(smoothness_list, key = lambda t: t[1])
            # number_of_grouped_kernel_to_keep = int((1 - pruning_rate) * len(sorted_smoothness_list))
            #
            # di_smooth_alt = [i for i, v in sorted_smoothness_list[:number_of_grouped_kernel_to_keep]]
            #
            # print(f'smooth_beam_greedy di len: {len(di)}; smoothness di len: {len(di_smooth_alt)}')
            # print(f'di: {di}')
            # print(f'smoothness di: {di_smooth_alt}')

            # sys.exit()


        else:
            logger.error(f"Invalid input on pruning_strategy: {pruning_strategy}")
            sys.exit()

        conv_preserved_kernel_index.append(di)
        for d in di:
            conv_prune_mask[i*d_out:(i+1)*d_out, d, :, :].fill_(1)

    conv_prune_mask = conv_prune_mask.double()
    new_weights = torch.mul(new_weights.double(), conv_prune_mask, out=None)
    #print(new_weights)
    new_conv.weight = torch.nn.Parameter(new_weights)
    new_conv.weight.data = new_conv.weight.type(torch.FloatTensor)
    new_conv.weight.data = new_conv.weight.data.cuda()


    return new_conv, conv_prune_mask, conv_preserved_kernel_index


def get_filter_smoothness(w, filter_num, kernel_size = (3, 3)):

    # print('w before reshape:', w.shape)

    w = w.reshape((filter_num, kernel_size[0], kernel_size[1])).tolist()

    # print('w after reshape:', torch.FloatTensor(w).shape)
    #
    # print(torch.FloatTensor(w))

    # print('w.reshape', w)

    filter_diff_sum = 0
    for a_kernel in w:
        for i in range(len(a_kernel)):
            for j in range(len(a_kernel[i])):
                filter_diff_sum += adj_diff_sum(a_kernel, i, j)


    return filter_diff_sum


def get_smoothness_ordering_labels(w, filter_num, n_clusters, kernel_size = (3, 3)):
    w_smoothness_LUT = [(filter_i, filter_w, get_filter_smoothness(filter_w, filter_num, kernel_size)) for filter_i, filter_w in enumerate(w)]

    w_smoothness_LUT.sort(key = lambda t: t[2])
    w_label_list = []

    w_smoothness_LUT = [(filter_i, filter_w, filter_smoothness, (cluster_label)%n_clusters) for cluster_label, (filter_i, filter_w, filter_smoothness) in enumerate(w_smoothness_LUT)]

    w_smoothness_LUT.sort(key = lambda t:t[0])


    return [i[-1] for i in w_smoothness_LUT]



def get_smoothness_snaking_labels(w, filter_num, n_clusters, kernel_size = (3, 3)):
    w_smoothness_LUT = [(filter_i, filter_w, get_filter_smoothness(filter_w, filter_num, kernel_size)) for filter_i, filter_w in enumerate(w)]

    w_smoothness_LUT.sort(key = lambda t: t[2])

    smoothness_ordering_labels = [[i for i in range(n_clusters)]] * (len(w_smoothness_LUT) // n_clusters)

    for line_num, line in enumerate(smoothness_ordering_labels):
        if line_num % 2 != 0:
            line = line[::-1]
            smoothness_ordering_labels[line_num] = line

    smoothness_ordering_labels = torch.IntTensor(smoothness_ordering_labels).flatten().tolist()

    w_smoothness_LUT = [(filter_i, filter_w, filter_smoothness, smoothness_ordering_labels[cluster_label]) for cluster_label, (filter_i, filter_w, filter_smoothness) in enumerate(w_smoothness_LUT)]

    w_smoothness_LUT.sort(key = lambda t:t[0])


    return [i[-1] for i in w_smoothness_LUT]








def adj_diff_sum(target_kernel, i, j, kernel_size = (3, 3), p = 2):
    def get_diff(target_kernel, i, j, target):
        if i < 0 or j < 0 or i >= kernel_size[0] or j >= kernel_size[1]:
            return 0
        else:
            return abs(target_kernel[i][j]**p - target**p)
    target = target_kernel[i][j]
    diff_sum = get_diff(target_kernel, i-1, j-1, target) + \
                get_diff(target_kernel, i-1, j, target) + \
                get_diff(target_kernel, i-1, j+1, target) + \
                get_diff(target_kernel, i, j-1, target) + \
                get_diff(target_kernel, i, j+1, target) + \
                get_diff(target_kernel, i+1, j-1, target) + \
                get_diff(target_kernel, i+1, j, target) + \
                get_diff(target_kernel, i+1, j+1, target)
    return diff_sum




def get_kernel_smoothness(w, out_channels, group_size, kernel_size = (3, 3), p = 2):

    # print('w before reshape:', w.shape)
    w = w.reshape((out_channels, group_size, kernel_size[0], kernel_size[1])).tolist()

    # print('w after reshape:', torch.FloatTensor(w).shape)

    # print('w.reshape', w)
    total_diff_sum_list = []

    for a_kernel_group in w:
        group_diff_sum = 0
        for a_kernel in a_kernel_group:
            for i in range(len(a_kernel)):
                for j in range(len(a_kernel[i])):
                    group_diff_sum += adj_diff_sum(a_kernel, i, j, p = p)
        total_diff_sum_list.append(group_diff_sum)


    return total_diff_sum_list


def make_new_conv(module, group_info = None):
    in_channels = module.in_channels
    groups = module.groups

    if group_info:
        n_clusters, pruning_rate, kernel_gcd = group_info
        number_of_unpruned_kernels = float((1 - pruning_rate) * kernel_gcd)

        if not number_of_unpruned_kernels.is_integer:
            logger.error(f'Should have int amount of unpruned kernels, now with: (1 - {pruning_rate}) * {kernel_gcd} = {number_of_unpruned_kernels}')
            os.exit()



        number_of_unpruned_kernels = int(number_of_unpruned_kernels)
        in_channels = module.in_channels * number_of_unpruned_kernels // (kernel_gcd / n_clusters)

        in_channels = int(in_channels)
        groups = n_clusters


    new_conv = torch.nn.Conv2d(in_channels = in_channels,
                       out_channels=module.out_channels,
                       kernel_size=module.kernel_size,
                       stride=module.stride,
                       padding=module.padding,
                       padding_mode='zeros',
                       dilation=module.dilation,
                       groups = groups,
                       bias=False)

    if group_info:
        return new_conv, number_of_unpruned_kernels
    else:
        return new_conv



def get_magnitude_snaking_labels(w, n_clusters, kernel_size = (1, 1)):
    w_magnitude_LUT = [(filter_i, filter_w, sum(filter_w)) for filter_i, filter_w in enumerate(w)]
    # print(w_magnitude_LUT)

    # w_magnitude_LUT = [(filter_i, filter_w, get_filter_smoothness(filter_w, filter_num, kernel_size)) for filter_i, filter_w in enumerate(w)]
    #
    w_magnitude_LUT.sort(key = lambda t: t[-1], reverse = True)

    magnitude_ordering_labels = [[i for i in range(n_clusters)]] * (len(w_magnitude_LUT) // n_clusters)

    for line_num, line in enumerate(magnitude_ordering_labels):
        if line_num % 2 != 0:
            line = line[::-1]
            magnitude_ordering_labels[line_num] = line

    magnitude_ordering_labels = torch.IntTensor(magnitude_ordering_labels).flatten().tolist()

    w_magnitude_LUT = [(filter_i, filter_w, filter_magnitude, magnitude_ordering_labels[cluster_label]) for cluster_label, (filter_i, filter_w, filter_magnitude) in enumerate(w_magnitude_LUT)]

    w_magnitude_LUT.sort(key = lambda t:t[0])


    return [i[-1] for i in w_magnitude_LUT]


def get_cluster_permutation_matrix(old_weights, old_out_channels, n_clusters = 8, clustering_method = 'Smoothness Snaking', one_by_one_conv_flag = False):
    permutation_matrices = []
    score_dicts = []

    old_weights_normalized = torch.from_numpy(old_weights).float()
    old_weights_normalized = F.normalize(old_weights_normalized, p=2, dim=1).numpy()


    if clustering_method == 'Smoothness Snaking':

        if not one_by_one_conv_flag:
            equal_group_labels = get_smoothness_snaking_labels(old_weights, old_weights.shape[1]//9, n_clusters)
        else:
            equal_group_labels = get_magnitude_snaking_labels(old_weights, n_clusters)

        permutation_matrix = get_permutation_matrix(old_weights, equal_group_labels)

    elif clustering_method == 'Smoothness Ordering':
        equal_group_labels = get_smoothness_ordering_labels(old_weights, old_weights.shape[1]//9, n_clusters)
        # print(f'equal_group_labels: {equal_group_labels}')
        permutation_matrix = get_permutation_matrix(old_weights, equal_group_labels)

    elif clustering_method == 'debug':
        equal_group_labels = [(cluster_label)%n_clusters for cluster_label, filter_i in enumerate(old_weights)]
        random.shuffle(equal_group_labels)

        # print(f'equal_group_labels: {equal_group_labels}')
        permutation_matrix = get_permutation_matrix(old_weights, equal_group_labels)


    best_permutation_matrix = permutation_matrix
    best_cluster_method = clustering_method

    return best_permutation_matrix, best_cluster_method



def get_out_index(permutation_matrix):
    q = []
    n, m = permutation_matrix.shape
    for j in range(n):
        for i in range(m):
            if permutation_matrix[j, i] == 1:
                q.append(i)

    q = np.array(q)
    q = torch.from_numpy(q)
    return q





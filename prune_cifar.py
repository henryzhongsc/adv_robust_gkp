import torch
from torch.autograd import Variable
from torchvision import models
#import cv2
import argparse
import sys
import time
import math
import copy
import logging
import json

import numpy as np
import torchvision
import torch.nn.functional as F

# from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import pdist, squareform
import scipy.spatial as sp

# from sklearn.decomposition import PCA
# from sklearn.decomposition import KernelPCA
# from Spectral_Clustering.spectral_clustering import Spectral_Clustering
# from same_size_dbscan import Same_Size_DBSCAN

from gkp_utils import *

logger = logging.getLogger("Test")


logger.info('All modules imported')

class ModifiedResNet(torch.nn.Module):
    def __init__(self, model, original_model, pruning_rate = 0.4375, setting = None):
        super(ModifiedResNet, self).__init__()

        self.conv1 = model.conv_1_3x3
        self.bn1 = model.bn_1
        self.layer1 = model.stage_1
        self.layer2 = model.stage_2
        self.layer3 = model.stage_3
        self.avgpool = model.avgpool
        self.linear = model.classifier


        self.original_model = original_model
        self.kernel_gcd = float('inf')

        # self.model = model
        self.pruning_rate = pruning_rate
        self.pruning_target_layers = setting['prune_params']['pruning_target_layers']
        self.grouping_target_layers = setting['prune_params']['grouping_target_layers']
        self.n_clusters = setting['prune_params']['n_clusters']
        self.clustering_method = setting['prune_params']['clustering_method']
        self.pruning_strategy = setting['prune_params']['pruning_strategy']


        logger.info(f"clustering_method: {self.clustering_method}; n_clusters: {self.n_clusters}; pruning_target_layers: {self.pruning_target_layers}; grouping_target_layers: {self.grouping_target_layers}.")
        logger.info(f"pruning_strategy: {self.pruning_strategy}; pruning_rate: {self.pruning_rate}.")

        if self.pruning_strategy =='smooth_cost_beam':
            logger.info(f"\tmetric: {setting['prune_params']['metric']}; inner_outer_balancer: {setting['prune_params']['inner_outer_balancer']}; cost_smooth_balancer: {setting['prune_params']['cost_smooth_balancer']}; eval_kept_kernel_number: {setting['prune_params']['eval_kept_kernel_number']}; beam_width: {setting['prune_params']['beam_width']}; smoothness_check_step: {setting['prune_params']['smoothness_check_step']}.")

        for current_block, layer, sublayer, modules, submodule in self.gen_unpruned_block(model):
            modules[sublayer] = self.prune_block(current_block, submodule, (layer, sublayer), setting = setting)


    def gen_unpruned_block(self, model):

        # imagenet_target_layers = [4, 5, 6, 7]
        # tiny_imagenet_target_layers = [2, 3, 4]
        # cifar10_target_layers = [2, 3, 4]
        #
        # if self.dataset == 'imagenet':
        #     target_layers = imagenet_target_layers
        # elif self.dataset == 'tiny_imagenet':
        #     target_layers = tiny_imagenet_target_layers
        # elif self.dataset == 'cifar10':
        #     target_layers = cifar10_target_layers
        # else:
        #     logger.error(f'Invalid dataset input {dataset}.')
        #     sys.exit(0)



        for layer, (name, modules) in enumerate(model._modules.items()):
            if layer in self.pruning_target_layers:
                for sublayer, (name, submodule) in enumerate(modules._modules.items()):
                    current_block = modules[sublayer]
                    yield current_block, layer, sublayer, modules, submodule

    def prune_block(self, current_block, submodule, block_info, setting = None):
        layer, sublayer = block_info
        block_out_list = []
        block_layer_list = []
        # if pruned_flag:
        #     block_in_planes = None
        #     block_planes = None
        # else:
        block_in_planes = submodule.conv_a.in_channels
        block_planes = submodule.conv_a.out_channels

        block_prune_masks = []
        block_candidate_methods_list = []
        block_preserved_kernel_index = []
        block_layer_info = []

        for subsublayer, (name, module) in enumerate(submodule._modules.items()):
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                new_conv = make_new_conv(module)


                old_weights = module.weight.cuda()
                old_out_channels, old_in_channels, old_kernel_size, old_kernel_size = old_weights.data.size()
                old_weights = old_weights.data.cpu().numpy()
                original_shape = old_weights.shape

                self.kernel_gcd = min(self.kernel_gcd, original_shape[0])
                old_weights_float = torch.from_numpy(old_weights).float()

                layer_info = (layer, sublayer, subsublayer)

                # if not pruned_flag:
                block_layer_info.append(layer_info)

                old_weights = old_weights.reshape(old_out_channels, old_in_channels*old_kernel_size*old_kernel_size)

                preferred_permutation_matrix, preferred_clustering_method = get_cluster_permutation_matrix(old_weights, old_out_channels, n_clusters = self.n_clusters, clustering_method = self.clustering_method)

                clustering_info = (preferred_permutation_matrix, preferred_clustering_method)

                block_candidate_methods_list.append(clustering_info)


                logger.info(f'Layer {layer}-{sublayer}-{subsublayer}; Shape {original_shape} -> {old_weights.shape}; Method: {preferred_clustering_method}.')


                block_out_index = get_out_index(preferred_permutation_matrix.transpose(1,0)).cuda()
                block_out_index = Variable(block_out_index)
                block_out_list.append(block_out_index)
                new_weights = np.dot(preferred_permutation_matrix, old_weights)
                new_weights = new_weights.reshape(old_out_channels, old_in_channels, old_kernel_size, old_kernel_size)
                new_weights = Variable(torch.from_numpy(new_weights)).cuda()


                if subsublayer == 0:
                    conv_num = 0
                elif subsublayer == 2:
                    conv_num = 1
                else:
                    logger.error(f'subsublayer: {subsublayer} is neither 0 or 2.')
                    sys.exit(0)

                new_conv, new_conv_prune_mask, new_conv_preserved_kernel_index = prune_kernels(current_block, conv_num, new_conv, new_weights, old_out_channels,
                                pruning_rate = self.pruning_rate,
                                n_clusters = self.n_clusters,
                                pruning_strategy = setting['prune_params']['pruning_strategy'],
                                setting = setting
                            )

                block_prune_masks.append(new_conv_prune_mask)
                block_layer_list.append(new_conv)
                block_preserved_kernel_index.append(new_conv_preserved_kernel_index)

        return NewBasicblock(submodule, submodule.bn_a, submodule.bn_b, block_out_list, block_layer_list, block_prune_masks, block_candidate_methods_list, block_preserved_kernel_index, in_planes = block_in_planes, planes = block_planes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x




class NewBasicblock(torch.nn.Module):
    expansion = 1
    def __init__(self, module, bn1, bn2, out_list, layer_list, prune_mask, candidate_methods_list, preserved_kernel_index, in_planes = None, planes = None, stride=1, shortcut=None):
        super(NewBasicblock, self).__init__()

        self.out_list = out_list
        self.conv1 = layer_list[0]
        self.bn1 = bn1
        self.conv2 = layer_list[1]
        self.bn2 = bn2
        self.shortcut = shortcut
        self.prune_mask = prune_mask
        self.candidate_methods_list = candidate_methods_list
        self.preserved_kernel_index = preserved_kernel_index

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = module.downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = torch.index_select(out, 1, self.out_list[0])
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = torch.index_select(out, 1, self.out_list[1])
        out = self.bn2(out)
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out += residual
        out = F.relu(out)
        return out
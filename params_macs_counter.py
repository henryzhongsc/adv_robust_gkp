

import torch
import argparse



def parse_args():

    parser = argparse.ArgumentParser(description='PyTorch FLOPs counter')
  
    parser.add_argument('--method', default='macs', type=str, help='macs/thop/params/torchscan/CC')


    parser.add_argument('--model_dir', help='The model directory with weights')
    parser.add_argument('--device', type=str, help='cpu', default='cuda')
    parser.add_argument('--dataset', type=str, help='cifar10/imagenet/mnist', default='cifar10')
    parser.add_argument('--multiply_adds', type=str, default='')
    parser.add_argument('--ignore_bn', type=str, default='')
    parser.add_argument('--ignore_relu', type=str, default='')
    parser.add_argument('--ignore_maxpool', type=str, default='')
    parser.add_argument('--ignore_bias', type=str, default='')


    args = parser.parse_args()

    return args



def get_macs_dpf(current_device, model, dataset, multiply_adds,
                 ignore_bn, ignore_relu, ignore_maxpool, ignore_bias
                 , ignore_zero=True, display_log=True):


    import torch

    # Inspired from DPF code (Lin et al 2020)
    # ---------------
    # Code from https://github.com/simochen/model-tools.

    import numpy as np

    import torch.nn as nn

    """for cv tasks."""

    data = dataset
    device = current_device

    if "cifar10" == data:
        input_res = [3, 32, 32]
    elif 'tinyimagenet' == data:
        input_res = [3, 64, 64]
    elif "imagenet" == data:
        input_res = [3, 224, 224]
    elif "mnist" == data:
        input_res = [1, 28, 28]
    else:
        raise RuntimeError("not supported imagenet type.")

    prods = {}

    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)

        return hook_per

    list_1 = []

    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))

    list_2 = {}

    def simple_hook2(self, input, output):
        list_2["names"] = np.prod(input[0].shape)

    list_conv = []
    module_names = []

    def conv_hook(self, input, output):
        # print(self.weight.shape)
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = (
                self.kernel_size[0] * self.kernel_size[1] *
                (self.in_channels / self.groups)
        )
        bias_ops = 1 if not ignore_bias and self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        num_weight_params = (
            (self.weight.data != 0).float().sum()
            if ignore_zero
            else self.weight.data.nelement()
        )
        assert self.weight.numel() == kernel_ops * output_channels, "Not match"
        flops = (
                (
                        num_weight_params * (2 if multiply_adds else 1)
                        + bias_ops * output_channels
                )
                * output_height
                * output_width
                * batch_size
        )

        list_conv.append(flops)
        module_names.append(self.name)

    list_linear = []


    def linear_hook(self, input, output):
        # print(self.weight.shape)
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        num_weight_params = (
            (self.weight.data != 0).float().sum()
            if ignore_zero
            else self.weight.data.nelement()
        )
        weight_ops = num_weight_params * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement() if not ignore_bias else 0

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)
        module_names.append(self.name)

    list_bn = []

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu = []

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling = []

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (
                (kernel_ops + bias_ops)
                * output_channels
                * output_height
                * output_width
                * batch_size
        )

        list_pooling.append(flops)

    list_upsample = []

    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)


    def foo(net, name=''):

        children = list(net.named_children())
        if not children:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
                setattr(net, 'name', name)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
                setattr(net, 'name', name)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(
                    net, torch.nn.AvgPool2d
            ):
                net.register_forward_hook(pooling_hook)
            if isinstance(net, torch.nn.Upsample):
                net.register_forward_hook(upsample_hook)
            return
        for child_name, child in children:
            foo(child, name="{}.{}".format(name, child_name))

    assert model is not None
    # print(model)
    foo(model)
    # 1, 3, 224, 224
    _input = torch.rand(*input_res).unsqueeze(0).to(device)

    model(_input)

    total_flops = (
            sum(list_conv)
            + sum(list_linear)
            + (sum(list_bn) if not ignore_bn else 0)
            + (sum(list_relu) if not ignore_relu else 0)
            + (sum(list_pooling) if not ignore_maxpool else 0)
            + sum(list_upsample)
    )
    total_flops = (
        total_flops.item() if isinstance(total_flops, torch.Tensor) else total_flops
    )
    list_conv = [x.item() for x in list_conv]
    list_linear = [x.item() for x in list_linear]
    # print("list conv is ", list_conv)
    # print("list linear is ", list_linear)
    # print("list module_names is ", module_names)

    # print(sum(list_linear) + sum(list_conv))
    print("Output:")
    if display_log:
        print(
            "  + Number of {}: {:.3f}M".format(
                "flop" if multiply_adds else "macs", 1.0 * total_flops / 1e6
            )
        )
    return total_flops, list_conv + list_linear, module_names



def get_total_sparsity_unwrapped(module):
    if hasattr(module, "weight"):
        num_sparse = (module.weight.data == 0).float().sum()
        num_params = module.weight.numel()
        if hasattr(module, "bias") and module.bias is not None:
            num_params += module.bias.numel()
            num_sparse += (module.bias.data == 0).float().sum()
        return num_sparse, num_params
    num_zeros, num_params = 0, 0
    for child_module in module.children():
        num_zeros_child, total_child = get_total_sparsity_unwrapped(child_module)
        # print(f"for child_module {child_module}, num_zeros: {num_zeros_child} & total_child: {total_child}")
        num_zeros += num_zeros_child
        num_params += total_child
    return num_zeros, num_params


def thop_flops(model, data):

    if "cifar10" == data:
        input_res = torch.randn(1, 3, 32, 32)
    elif "imagenet" == data:
        input_res = torch.randn(1, 3, 224, 224)
    elif "mnist" == data:
        input_res = torch.randn(1, 1, 28, 28)
    else:
        raise RuntimeError("not supported imagenet type.")

    macs, params = profile(model, inputs=(input_res, ))
    macs, params = clever_format([macs, params], "%.3f")
    print("Output of thop:")
    print("The macs is:", macs)
    print("The params is:", params)


def torch_scan(model, data):

    if "cifar10" == data:
        input_res = (3, 32, 32)
    elif "imagenet" == data:
        input_res = (3, 224, 224)
    elif "mnist" == data:
        input_res = (1, 28, 28)
    else:
        raise RuntimeError("not supported imagenet type.")

    summary(model, input_res, max_depth=2)


def macs_print(device_p, model_p, dataset_p, multiply_adds_p, ignore_relu, 
        ignore_bn, ignore_maxpool, ignore_bias):

    get_macs_dpf(current_device=device_p, model=model_p, dataset=dataset_p, 
    multiply_adds=multiply_adds_p, ignore_relu=ignore_relu, ignore_bn=ignore_bn, 
    ignore_maxpool=ignore_maxpool, ignore_bias=ignore_bias)


def params_print(model):
    num_zeros, num_params = get_total_sparsity_unwrapped(module=model)
    print(
            "  + Number of {}: {:.3f}M".format(
                "parameters", 1.0 * (num_params - num_zeros).item() / 1e6
            )
        )


def args_switcher(argument):

    switcher = {
        '': False,
        'False':  False,
        'True': True
    }

    return switcher.get(argument, False)


def CC_flops(model, params=False):
    import pdb
    from operator import mul
    from functools import reduce
    import operator
    from collections import OrderedDict, namedtuple
    import functools
    import itertools
    import random
    import math
    import pickle

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn import Parameter

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    import multiprocessing as mp
    from multiprocessing import Pool, Manager

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from scipy.optimize import leastsq

    _PER_FLOPS = 10**7

    count_ops = []
    layer_inputsize = []
    flop_ops = 0

    def get_num_gen(gen):
        return sum(1 for x in gen)


    def is_leaf(model):
        return get_num_gen(model.children()) == 0


    def get_layer_info(layer):
        layer_str = str(layer)
        type_name = layer_str[:layer_str.find('(')].strip()
        return type_name


    def compress(model, com_ratio=0):
        for child in model.children():
            if is_leaf(child):
                if get_layer_info(child) in [
                        'Conv2d_Pruning', 'Conv2d_TD', 'Conv2d_Quant'
                ]:
                    child.compress(com_ratio)
                    print(child, 'Compression finish!')
            else:
                compress(child, com_ratio=com_ratio)

    def cal_metric(org_weight,
                    now_weight,
                    grad,
                    com_ops,
                    pruning_index=[],
                    eigenvalue_zero_num=[],
                    com_gamma=0.9):

        num = now_weight.shape[1] - len(pruning_index) + min(
            now_weight.shape[0], reduce(
                mul, now_weight.shape[1:])) - eigenvalue_zero_num

        metrics = []
        for j in range(len(com_ops)):
            if com_ops[j] == 'pruning':
                for k in range(now_weight.shape[1]):
                    if k not in pruning_index:
                        keep_index = [
                            i for i in range(org_weight.shape[1])
                            if i not in pruning_index and i != k
                        ]

                        this_weight_k = now_weight[:, k].data.cpu().numpy()
                        now_weight_k = now_weight
                        now_weight_k[:, k] = 0
                        backward_metric = torch.sum(
                            torch.pow(grad * (now_weight_k - org_weight), 2))

                        forward_metric = torch.sum(
                            torch.pow(grad * (now_weight_k - org_weight), 2) +
                            2 / (num - 1) * (2 * (grad * grad *
                                                (now_weight_k - org_weight) *
                                                -now_weight_k)))

                        try:
                            u, s, v = torch.svd(now_weight_k.cpu().reshape(
                                now_weight_k.shape[0], -1))
                            u = torch.pow(u, 2).cuda()
                            s = torch.pow(s, 2).cuda()
                            v = torch.pow(v, 2).cuda()
                            new_weight = torch.mm(torch.mm(u, torch.diag(s)),
                                                v.t())
                            new_weight = new_weight.reshape(now_weight.size())
                            forward_metric += torch.sum(
                                (torch.pow(grad * now_weight_k, 2) +
                                (torch.pow(grad, 2) *
                                new_weight))) / (num - 1)
                        except:
                            forward_metric += torch.sum(
                                (torch.pow(grad * now_weight_k,
                                        2))) * 2 / (num - 1)

                        forward_metric = com_gamma * forward_metric
        
                        metrics.append(
                            ('pruning', k,
                            backward_metric.item() + forward_metric.item()))
                        now_weight[:, k] = torch.from_numpy(this_weight_k).cuda()
                        del this_weight_k

            elif com_ops[j] == 'td':
                try:
                    u, s, v = torch.svd(now_weight.cpu().reshape(
                        now_weight.shape[0], -1))
                except:
                    continue
                u = u.cuda()
                s = s.cuda()
                v = v.cuda()
                use_rank_num = min(
                    now_weight.shape[0], reduce(
                        mul, now_weight.shape[1:])) - eigenvalue_zero_num
                for k in range(use_rank_num):
                    org_value = s[k].item()
                    s[k] = 0
                    now_weight_k = torch.mm(torch.mm(u, torch.diag(s)), v.t())
                    now_weight_k = now_weight_k.reshape(now_weight.size())
                    backward_metric = torch.sum(
                        torch.pow(grad * (now_weight_k - org_weight), 2))

                    keep_index = [
                        i for i in range(org_weight.shape[1])
                        if i not in pruning_index
                    ]

                    forward_metric = torch.sum(
                        torch.pow(grad *
                                (now_weight_k - org_weight), 2) +
                        2 / (num - 1) * (2 * (grad * grad *
                                            (now_weight_k - org_weight) *
                                            -now_weight_k)))

                    try:
                        un = torch.pow(u, 2)
                        sn = torch.pow(s, 2)
                        vn = torch.pow(v, 2)
                        new_weight = torch.mm(torch.mm(un, torch.diag(sn)), vn.t())
                        new_weight = new_weight.reshape(now_weight.size())
                        forward_metric += torch.sum(
                            (torch.pow(grad * now_weight_k, 2) +
                            (torch.pow(grad, 2) * new_weight))) / (
                                num - 1)
                    except:
                        forward_metric += torch.sum(
                            (torch.pow(grad * now_weight_k,
                                    2))) * 2 / (num - 1)

                    forward_metric = com_gamma * forward_metric

                    metrics.append(
                        ('td', k, backward_metric.item() + forward_metric.item()))
                    s[k] = org_value
        
        return metrics

    def get_layer_param(model):
        return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])


    def get_conv_flop(layer, x):
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        delta_params = get_layer_param(layer)
        delta_ops = delta_params * out_h * out_w
        return delta_ops


    def get_fc_flop(layer, x):
        delta_params = get_layer_param(layer)
        delta_ops = x.size()[0] * delta_params
        return delta_ops


    def measure_flop_layer(layer, x):
        global count_ops, flop_ops, layer_inputsize
        delta_ops = 0
        type_name = get_layer_info(layer)

        if type_name in ['Conv2d', 'Conv2d_compress']:
            delta_ops = get_conv_flop(layer, x)
            # print(layer, delta_ops)

        elif type_name in ['Linear']:
            delta_ops = get_fc_flop(layer, x)
            # print(layer, delta_ops)

        elif type_name in ['BasicBlock_Compress', 'ResNetBasicblock']:
            pruning_ratio = len(
                layer.conv2.keep_index) / layer.conv2.input_channels
            delta_ops = get_conv_flop(layer.conv1, x) * pruning_ratio
            delta_ops += get_conv_flop(layer.conv2, x)
            # print(delta_ops)

        elif type_name in [
                'ReLU', 'ReLU6', 'Sigmoid', 'AvgPool2d', 'MaxPool2d',
                'AdaptiveAvgPool2d', 'BatchNorm2d', 'Dropout2d', 'DropChannel',
                'Dropout', 'Sequential', 'LambdaLayer'
        ]:
            pass

        # unknown layer type
        else:
            raise TypeError('unknown layer type: %s' % type_name)

        if delta_ops != 0:
            flop_ops += delta_ops / _PER_FLOPS
            if type_name in ['Conv2d'] and layer.groups != 1:
                return
            count_ops.append(delta_ops / _PER_FLOPS)
            if len(x.shape) == 4:
                layer_inputsize.append((x.shape[2], x.shape[3]))
            else:
                layer_inputsize.append((x.shape[1]))
        return


    def measure_flop_model(model, H, W):
        global count_ops, flop_ops, layer_inputsize
        count_ops = []
        layer_inputsize = []
        flop_ops = 0
        data = torch.zeros(1, 3, H, W).cuda()

        def should_measure(x):
            return is_leaf(x)

        def modify_forward(model):
            for child in model.children():
                type_name = get_layer_info(child)
                if should_measure(child) or type_name in ['BasicBlock_Compress']:

                    def new_forward(m):
                        def lambda_forward(x):
                            measure_flop_layer(m, x)
                            return m.old_forward(x)

                        return lambda_forward

                    child.old_forward = child.forward
                    child.forward = new_forward(child)
                else:
                    modify_forward(child)

        def restore_forward(model):
            for child in model.children():
                # leaf node
                type_name = get_layer_info(child)
                if (should_measure(child)
                        or type_name in ['BasicBlock_Compress']) and hasattr(
                            child, 'old_forward'):
                    child.forward = child.old_forward
                    child.old_forward = None
                else:
                    restore_forward(child)

        modify_forward(model)
        model.forward(data)
        restore_forward(model)



        return flop_ops
    
    if args.dataset == 'cifar10':
        flops1 = measure_flop_model(model, 32, 32)
    else:
        flops1 = measure_flop_model(model, 224, 224)

    a1 = flops1 * (10**7)

    if params:
        return a1, get_layer_param(model)
    
    else:
        return a1

def get_flops_params_dhp(model):
    import torch.nn as nn
    import numpy as np
    from DHP.model_dhp.dhp_base import conv_dhp
# from IPython import embed

    def set_output_dimension(model, input_res):
        assert type(input_res) is tuple, 'Please provide the size of the input image.'
        assert len(input_res) >= 3, 'Input image should have 3 dimensions.'
        feat_model = add_feature_dimension(model)
        feat_model.eval().start_dimension_add()
        device = list(feat_model.parameters())[-1].device
        batch = torch.FloatTensor(1, *input_res).to(device)
        _ = feat_model(batch)
        feat_model.stop_dimension_add()

    
    def get_flops(model):
        flops = 0
        for module in model.modules():
            if is_supported_instance(module):
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, conv_dhp)):
                    flops += conv_calc_flops(module)
                elif isinstance(module, (nn.ReLU, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6)):
                    flops += relu_calc_flops(module)
                    # if isinstance(module, nn.ReLU):
                    #     print(module)
                elif isinstance(module, (nn.Linear)):
                    flops += linear_calc_flops(module)
                elif isinstance(module, (nn.BatchNorm2d)):
                    flops += bn_calc_flops(module)
        return flops
    
    
    def get_parameters(model):
        parameters = 0
        for module in model.modules():
            if is_supported_instance(module):
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.ReLU, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6)):
                    for p in module.parameters():
                        parameters += p.nelement()
                elif isinstance(module, nn.Linear):
                    in_features = module.in_features_remain if hasattr(module, 'in_features_remain') else module.in_features
                    out_features = module.out_features_remain if hasattr(module, 'out_features_remain') else module.out_features
                    parameters += in_features * out_features
                    if module.bias is not None:
                        parameters += module.out_features
                elif isinstance(module, (conv_dhp)):
                    in_channels = module.in_channels_remain if hasattr(module, 'in_channels_remain') else module.in_channels
                    out_channels = module.out_channels_remain if hasattr(module, 'out_channels_remain') else module.out_channels
                    groups = module.groups_remain if hasattr(module, 'groups_remain') else module.groups
                    parameters += in_channels // groups * out_channels * module.kernel_size ** 2
                    if module.bias is not None:
                        parameters += out_channels
                elif isinstance(module, nn.BatchNorm2d):
                    if module.affine:
                        num_features = module.num_features_remain if hasattr(module, 'num_features_remain') else module.num_features
                        parameters += num_features * 2
        return parameters
    
    
    def add_feature_dimension(net_main_module):
        # adding additional methods to the existing module object,
        # this is done this way so that each function has access to self object
        net_main_module.start_dimension_add = start_dimension_add.__get__(net_main_module)
        net_main_module.stop_dimension_add = stop_dimension_add.__get__(net_main_module)
    
        return net_main_module
    
    
    def start_dimension_add(self):
        """
        A method that will be available after add_flops_counting_methods() is called
        on a desired net object.
    
        Activates the computation of mean flops consumption per image.
        Call it before you run the network.
    
        """
        self.apply(add_feat_dim_hook_function)
    
    
    def stop_dimension_add(self):
        """
        A method that will be available after add_flops_counting_methods() is called
        on a desired net object.
    
        Stops computing the mean flops consumption per image.
        Call whenever you want to pause the computation.
    
        """
        self.apply(remove_feat_dim_hook_function)
    
    
    def add_feat_dim_hook_function(module):
        if is_supported_instance(module):
            if hasattr(module, '__flops_handle__'):
                return
    
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, conv_dhp)):
                handle = module.register_forward_hook(conv_feat_dim_hook)
            elif isinstance(module, (nn.ReLU, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6)):
                handle = module.register_forward_hook(relu_feat_dim_hook)
            elif isinstance(module, nn.Linear):
                handle = module.register_forward_hook(linear_feat_dim_hook)
            elif isinstance(module, nn.BatchNorm2d):
                handle = module.register_forward_hook(bn_feat_dim_hook)
            else:
                raise NotImplementedError('FLOPs calculation is not implemented for class {}'.format(module.__class__.__name__))
            module.__flops_handle__ = handle
    
    
    def remove_feat_dim_hook_function(module):
        if is_supported_instance(module):
            if hasattr(module, '__flops_handle__'):
                module.__flops_handle__.remove()
                del module.__flops_handle__
    
    
    # ---- Internal functions
    def is_supported_instance(module):
        if isinstance(module,
                      (
                              conv_dhp,
                              nn.Conv2d, nn.ConvTranspose2d,
                              nn.BatchNorm2d,
                              nn.Linear,
                              # nn.ReLU, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6,
                      )):
            if hasattr(module, '__exclude_complexity__'):
                return False
            else:
                return True
    
        return False
    
    
    def conv_feat_dim_hook(module, input, output):
        module.__output_dims__ = output.shape[2:]
    
    
    def conv_calc_flops(self):
        # Do not count bias addition
        batch_size = 1
        output_dims = np.prod(self.__output_dims__)
    
        kernel_dims = np.prod(self.kernel_size) if isinstance(self.kernel_size, tuple) else self.kernel_size ** 2
        in_channels = self.in_channels_remain if hasattr(self, 'in_channels_remain') else self.in_channels
        out_channels = self.out_channels_remain if hasattr(self, 'out_channels_remain') else self.out_channels
        groups = self.groups_remain if hasattr(self, 'groups_remain') else self.groups
        # groups = self.groups
    
        filters_per_channel = out_channels // groups
        conv_per_position_flops = kernel_dims * in_channels * filters_per_channel
    
        active_elements_count = batch_size * output_dims
    
        overall_conv_flops = conv_per_position_flops * active_elements_count
    
        return int(overall_conv_flops)
    
    
    def relu_feat_dim_hook(module, input, output):
        s = output.shape
        module.__output_dims__ = s[2:]
        module.__output_channel__ = s[1]
    
    
    def relu_calc_flops(self):
        batch = 1
        channels = self.channels if hasattr(self, 'channels') else self.__output_channel__
        active_elements_count = batch * np.prod(self.__output_dims__) * channels
        # print(active_elements_count, id(self))
        # print(self)
        return int(active_elements_count)
    
    
    def linear_feat_dim_hook(module, input, output):
        if len(output.shape[2:]) == 2:
            module.__additional_dims__ = 1
        else:
            module.__additional_dims__ = output.shape[1:-1]
    
    
    def linear_calc_flops(self):
        # Do not count bias addition
        batch_size = 1
        in_features = self.in_features_remain if hasattr(self, 'in_features_remain') else self.in_features
        out_features = self.out_features_remain if hasattr(self, 'out_features_remain') else self.out_features
        linear_flops = batch_size * np.prod(self.__additional_dims__) * in_features * out_features
        # print(self.in_features, in_features)
        return int(linear_flops)
    
    
    def bn_feat_dim_hook(module, input, output):
        module.__output_dims__ = output.shape[2:]
    
    
    def bn_calc_flops(self):
        # Do not count bias addition
        batch = 1
        output_dims = np.prod(self.__output_dims__)
        channels = self.num_features_remain if hasattr(self, 'num_features_remain') else self.num_features
        batch_flops = batch * channels * output_dims
        # print(self.num_features, channels)
        if self.affine:
            batch_flops *= 2
        return int(batch_flops)
        
    print(f'flops: {get_flops(model) * 0.000001 :.3f}\n Params: {get_parameters(model) * 0.000001 :.3f}')


if __name__ == '__main__':

    args = parse_args()
    device = args.device
    model_dir = args.model_dir
    dataset = args.dataset
    ma = args.multiply_adds
    
    model = torch.load(model_dir, map_location=torch.device(device))
    
    ig_relu = args_switcher(args.ignore_relu)
    ig_bn = args_switcher(args.ignore_bn)
    ig_mp = args_switcher(args.ignore_maxpool)
    ig_bias = args_switcher(args.ignore_bias)
    ma = args_switcher(args.multiply_adds)


    if args.method == 'macs':
        macs_print(device_p=device, model_p=model, dataset_p=dataset, multiply_adds_p=ma,
            ignore_relu=ig_relu, ignore_bn=ig_bn, ignore_maxpool=ig_mp, ignore_bias=ig_bias)
    
    elif args.method == 'params':
        params_print(model=model)

    elif args.method == 'thop':
        from thop import profile
        from thop import clever_format
        thop_flops(model=model, data=dataset)
   
    elif args.method == 'torchscan': 
        from torchscan import summary
        torch_scan(model=model, data=dataset)
        
    elif args.method == 'CC':
        cc_macs, params = CC_flops(model, True)
        print(
            "  + Number of {}: {:.3f}M".format(
                "macs", 1.0 * cc_macs / 1e6
            ))
        print(
            "  + Number of {}: {:.3f}M".format(
                "parameters", 1.0 * params / 1e6
            )
        )

    elif args.method == 'DHP':
        get_flops_params_dhp(model)
        

    elif args.method == 'params_macs_counter':
        macs_print(device_p=device, model_p=model, dataset_p=dataset, multiply_adds_p=ma,
            ignore_relu=ig_relu, ignore_bn=ig_bn, ignore_maxpool=ig_mp, ignore_bias=ig_bias)
        params_print(model=model)















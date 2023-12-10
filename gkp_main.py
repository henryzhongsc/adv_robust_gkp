import torch
import torch.nn as nn
import torch.optim as optim

import os
import sys
import argparse
import copy
import json
import datetime
import logging
import numpy as np

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import random
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


# from ModifiedPyTorchClassifier import PyTorchClassifier

import utils
import pruning_utils




start_time = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
start_time_int = datetime.datetime.now()
args = utils.parse_args()
logger = utils.set_logger(args.output_folder_dir, args)
setting = utils.register_args_to_setting(args, logger)
if args.gpu != '':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Experiment {setting['management']['exp_desc']} (task: {setting['management']['task']}) started at {start_time}.")

if setting['management']['dataset'] == 'cifar10':
    if args.model_type == 'basicblock':
        from prune_cifar import *
        from group_cifar import *
    if args.model_type == 'vgg':
        from prune_cifar_vgg import *
        from group_cifar_vgg import *
elif setting['management']['dataset'] == 'imagenet' or setting['management']['dataset'] == 'imagenet_cifar_mock':
    if setting['prune_params']['prune_conv_target'] == '3x3':
        from prune_imagenet import *
        from group_imagenet import *
    elif setting['prune_params']['prune_conv_target'] == 'all':
        from prune_imagenet_multi import *
        from group_imagenet_multi import *
    else:
        logger.error(f"Invalid setting['prune_params']['prune_conv_target'] == {setting['prune_params']['prune_conv_target']} input received.")
elif setting['management']['dataset'] == 'tiny_imagenet':
    if args.model_type == 'basicblock':
        from prune_cifar import *
        from group_cifar import *
    elif args.model_type == 'bottleneck':
        if setting['prune_params']['prune_conv_target'] == '3x3':
            from prune_imagenet import *
            from group_imagenet import *
        elif setting['prune_params']['prune_conv_target'] == 'all':
            from prune_imagenet_multi import *
            from group_imagenet_multi import *
        else:
            logger.error(f"Invalid setting['prune_params']['prune_conv_target'] == {setting['prune_params']['prune_conv_target']} input received.")
    else:
        logger.error(f"Invalid args.model_type == {args.model_type} input received.")

model = utils.load_model(setting, logger)

logger.info(f"Model {setting['management']['model_dir'] } loaded (type: {type(model)}).")

if setting['management']['task'] == 'gkp_prune':

    logger.info(f"Performing one-shot GKP pruning...")

    if args.model_type == 'basicblock' or args.model_type == 'bottleneck':
        original_model = copy.deepcopy(model)
        model = ModifiedResNet(model, original_model,
                    pruning_rate = setting['prune_params']['pruning_rate'],
                    setting = setting,
                ).cuda()
    elif args.model_type == 'vgg':
        model = ModifiedVGG(model,
                    pruning_rate = setting['prune_params']['pruning_rate'],
                    setting = setting,
                ).cuda()

    utils.save_trained_model(model, 'pruned', setting, logger)
    logger.info(f'Model {setting["management"]["model_dir"]} {type(model)} now pruned.')

    if args.model_type == 'basicblock' or args.model_type == 'bottleneck':
        model = GroupResNet(model).cuda()
    elif args.model_type == 'vgg':
        model = GroupVGG(model).cuda()

    utils.save_trained_model(model, 'grouped', setting, logger)
    logger.info(f'Model {setting["management"]["model_dir"]} {type(model)} now grouped.')

else:
    logger.error(f"Input task <{setting['management']['task']}> is not supported.")


end_time = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
end_time_int = datetime.datetime.now()
setting_output_path = setting['management']['output_folder_dir'] + 'output_setting.json'

with open(setting_output_path, "w+") as out_setting_f:
    logger.info(f"Saving output setting file to {setting_output_path}...")
    json.dump(setting, out_setting_f, indent = 4)

total_time =  str(end_time_int - start_time_int)
logger.info(f"Experiment {setting['management']['exp_desc']} (task: {setting['management']['task']}) ended at {end_time} duration: {total_time}.")




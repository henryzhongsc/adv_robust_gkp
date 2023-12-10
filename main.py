import torch
import torch.nn as nn
import torch.optim as optim
# import models.resnet_cifar

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

# from art.estimators.classification import PyTorchClassifier
from ModifiedPyTorchClassifier import PyTorchClassifier
# from art.utils import load_dataset

import utils
import pruning_utils
import models.resnet



start_time = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
start_time_int = datetime.datetime.now()
args = utils.parse_args()
logger = utils.set_logger(args.output_folder_dir, args)
setting = utils.register_args_to_setting(args, logger)
if args.gpu != '':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'



logger.info(f"Experiment {setting['management']['exp_desc']} (task: {setting['management']['task']}) started at {start_time}.")

# train_data_gen, x_test, y_test, clip_values, input_shape, nb_classes, trainloader = utils.load_dataset(setting)
train_data_gen, testloader, clip_values, input_shape, nb_classes, trainloader = utils.load_dataset(setting)
logger.info(f"Dataset {setting['management']['dataset']} loaded.")
criterion = nn.CrossEntropyLoss()

# model = models.resnet.resnet20()
if setting['management']['task'] == 'iterative_prune':
    model = pruning_utils.load_model(setting["prune_params"]["prune_method_name"],utils.load_model(setting, logger), setting, logger)
    vanilla_model = model.model
    optimizer = pruning_utils.load_optimizer(setting["prune_params"]["prune_method_name"], model, setting)
else:
    model = utils.load_model(setting, logger)
    vanilla_model = model
    optimizer = optim.SGD(vanilla_model.parameters(), lr = setting['train_params']['lr'], momentum = setting['train_params']['momentum'], weight_decay = setting['train_params']['weight_decay'])

classifier = PyTorchClassifier(
    model = vanilla_model,
    clip_values = clip_values,
    loss = criterion,
    optimizer = optimizer,
    input_shape = input_shape,
    nb_classes = nb_classes,
    setting = setting,
)
logger.info(f"Model {setting['management']['model_dir']} successfully wrapped as an ART classifier.")

setting['results'] = dict()
setting['results']['best'] = dict()


if setting['management']['task'] == 'test':
    logger.info(f"Testing performance of assigned model {setting['management']['model_dir']}...")
    utils.eval_classifier(classifier, epoch_num = 0, testloader = testloader, setting = setting, logger = logger, attack_flag = args.adv_attack)


if setting['management']['task'] == 'train' or setting['management']['task'] == 'finetune':

    logger.info(f"Testing baseline performance of model {setting['management']['model_dir']}...")
    utils.eval_classifier(classifier, epoch_num = 0, testloader = testloader, setting = setting, logger = logger, attack_flag = args.adv_attack)

    logger.info(f"Training classifier for {setting['train_params']['epoch_num']} epochs...")

    classifier.fit_generator(train_data_gen, nb_epochs = setting['train_params']['epoch_num'], testloader = testloader, setting = setting, logger = logger, attack_flag = args.adv_attack, trainloader = trainloader)

    if setting['management']['task'] == 'train':
        utils.save_trained_model(model, 'trained', setting, logger)
    elif setting['management']['task'] == 'fintune':
        utils.save_trained_model(model, 'fintuned', setting, logger)
    else:
        logger.info(f"Model {setting['management']['model_dir']} finetuned, please check {setting['management']['save_subdir']}/best_ckpts/ folder for best models.")
    utils.pprint_best_models(setting, logger)

elif setting['management']['task'] == 'iterative_prune':
    logger.info(f"Training classifier using {setting['prune_params']['prune_method_name']} for {setting['train_params']['epoch_num']} epochs...")

    #classifier.fit_generator(train_data_gen, nb_epochs = setting['train_params']['epoch_num'],  = x_test, y_test = y_test, setting = setting, logger = logger, attack_flag = args.adv_attack, trainloader = trainloader, shell = model)
    classifier.fit_generator(train_data_gen, nb_epochs = setting['train_params']['epoch_num'], testloader = testloader, setting = setting, logger = logger, attack_flag = args.adv_attack, trainloader = trainloader, shell = model)

    utils.save_trained_model(classifier.model, 'iterative_pruned', setting, logger)
    utils.pprint_best_models(setting, logger)

elif setting['management']['task'] == 'offline_loop_test':
    del classifier
    eval_worthy_state_dict_full_path_list = utils.get_eval_worthy_epochs(setting, logger)

    for eval_epoch_i, eval_epoch_path in eval_worthy_state_dict_full_path_list:

        vanilla_model.load_state_dict(torch.load(eval_epoch_path))
        classifier = PyTorchClassifier(
            model = vanilla_model,
            clip_values = clip_values,
            loss = criterion,
            optimizer = optimizer,
            input_shape = input_shape,
            nb_classes = nb_classes,
            setting = setting,
        )
        logger.info(f"Epoch #{eval_epoch_i} \t ({eval_epoch_path}) successfully wrapped as an ART classifier.")
        utils.eval_classifier(classifier, epoch_num = eval_epoch_i, testloader = testloader, setting = setting, logger = logger, attack_flag = args.adv_attack)
        del classifier

    logger.info(f"Offline loop tests finished.")
    utils.pprint_best_models(setting, logger)


end_time = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
end_time_int = datetime.datetime.now()
setting_output_path = setting['management']['output_folder_dir'] + 'output_setting.json'

with open(setting_output_path, "w+") as out_setting_f:
    logger.info(f"Saving setting file to {setting_output_path}...")
    json.dump(setting, out_setting_f, indent = 4)

total_time =  str(end_time_int - start_time_int)
logger.info(f"Experiment {setting['management']['exp_desc']} (task: {setting['management']['task']}) ended at {end_time} duration: {total_time}.")




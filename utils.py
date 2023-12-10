import argparse
from inspect import ArgSpec
import logging
from multiprocessing import current_process
import sys
import os
import copy
import json
import numpy as np
import dill

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import importlib

from art.data_generators import PyTorchDataGenerator
from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.evasion import HopSkipJump
from art.attacks.evasion import SaliencyMapMethod


def parse_args():

    parser = argparse.ArgumentParser(description='PyTorch training/testing')
    parser.add_argument('--exp_desc', type=str, help='experiment description')
    parser.add_argument('--setting_dir', type=str, help='path of setting JSON file')
    parser.add_argument('--dataset', default='cifar10', type=str, help='decide which dataset to run on.')
    parser.add_argument('--dataset_dir', default='../data', type=str, help='folder path of dataset.')
    parser.add_argument('--model_dir', default='', type=str, help='path of pretrain model')
    parser.add_argument('--output_folder_dir', default='', type=str, help='path of output model')
    parser.add_argument('--task', default='test', type=str, help='Select from one of the following tasks: finetune, train, test, iterative_prune, offline_loop_test.')
    parser.add_argument('--adv_attack', default='no_attack', type=str, help='Select an adversarial attack method to perturb the dataset, e.g., FGSM')
    parser.add_argument('--gpu', default='', help='gpu available')
    parser.add_argument('--debug', default=False, type=bool, help='Include logger from ART')
    parser.add_argument('--prune_method', default='GKP', type=str, help='Declare pruning method to facilitate model import')
    parser.add_argument('--adv_attack_init_epoch', default=0, type=int, help='Declare the starting epoch to register adv acc')

    parser.add_argument('--output_setting_dir', default='', type=str, help='path of state dict folder')
    parser.add_argument('--state_dict_dir', default='', type=str, help='path of state dict folder')
    parser.add_argument('--state_dict_eval_init_epoch', default='', type=str, help='path of state dict folder')
    parser.add_argument('--state_dict_eval_end_epoch', default='300', type=str, help='path of state dict folder')
    parser.add_argument('--state_dict_eval_inter', default='1', type=str, help='path of state dict folder')
    parser.add_argument('--state_dict_prefix', default='epoch_', type=str, help='path of state dict folder')
    parser.add_argument('--state_dict_suffix', default='.state_dict', type=str, help='path of state dict folder')

    parser.add_argument('--model_type', default='basicblock', type=str, help='Declare model type, basicblock or bottleneck')

    args = parser.parse_args()

    args.adv_attack = False if args.adv_attack == 'no_attack' else True


    if args.output_folder_dir != '':
        if args.output_folder_dir[-1] != '/':
            args.output_folder_dir  += '/'
    else:
        if args.task == 'offline_loop_test':
            eval_base_folder_path = '/'.join(args.model_dir.split('/')[:-3]) + '/'
            args.output_folder_dir = eval_base_folder_path + 'offline_loop_test_output/'
    if not os.path.isdir(args.output_folder_dir):
        os.makedirs(args.output_folder_dir)

    return args


def set_logger(output_folder_dir, args):
    log_formatter = logging.Formatter("%(asctime)s | %(levelname)s : %(message)s")
    if args.debug:
        logger = logging.getLogger()
    else:
        logger = logging.getLogger("Test")
    file_handler = logging.FileHandler(output_folder_dir + 'experiment.log', mode = 'w')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    return logger


def register_args_to_setting(args, logger):

    with open(args.setting_dir) as setting_f:
        setting = json.load(setting_f)
        logger.info(f'Input setting file {args.setting_dir} loaded.')

    input_setting_path = args.output_folder_dir + 'input_setting.json'
    with open(input_setting_path, "w+") as input_setting_f:
        json.dump(setting, input_setting_f, indent = 4)
        logger.info(f'Input setting file {args.setting_dir} saved to {input_setting_path}.')

    setting['results'] = dict()
    setting['management']['output_folder_dir'] = args.output_folder_dir
    setting['management']['exp_desc'] = args.exp_desc
    setting['management']['model_dir'] = args.model_dir
    setting['management']['task'] = args.task
    setting['management']['dataset'] = args.dataset
    setting['management']['dataset_dir'] = args.dataset_dir
    setting['management']['adv_attack'] = args.adv_attack
    setting['management']['prune_method'] = args.prune_method
    setting['management']['adv_attack_init_epoch'] = args.adv_attack_init_epoch

    if 'scheduler_type' not in setting['train_params'].keys():
        setting['train_params']['scheduler_type'] = 'StepLR'

    if 'using_normalization' not in setting['management'].keys():
        setting['management']['using_normalization'] = True
        logger.info('Using normalized dataset.')

    if 'prune_method_name' not in setting['prune_params'].keys():
        setting['prune_params']['prune_method_name'] = None


    if setting['management']['task'] == 'offline_loop_test':
        setting['management']['output_setting_dir'] = args.output_setting_dir
        setting['management']['state_dict_dir'] = args.state_dict_dir
        setting['management']['state_dict_eval_init_epoch'] = int(args.state_dict_eval_init_epoch)
        setting['management']['state_dict_eval_end_epoch'] = int(args.state_dict_eval_end_epoch)
        setting['management']['state_dict_eval_inter'] = int(args.state_dict_eval_inter)
        setting['management']['state_dict_prefix'] = args.state_dict_prefix
        setting['management']['state_dict_suffix'] = args.state_dict_suffix

        eval_base_folder_path = '/'.join(setting['management']['model_dir'].split('/')[:-3]) + '/'
        if setting['management']['output_setting_dir'] == '':
            setting['management']['output_setting_dir'] = eval_base_folder_path + 'output_setting.json'
            logger.info(f"setting['management']['output_setting_dir'] constructed as {setting['management']['output_setting_dir']}")

        if setting['management']['state_dict_dir'] == '':
            setting['management']['state_dict_dir'] = eval_base_folder_path + 'exp_ckpts/epoch_ckpts/'

            logger.info(f"setting['management']['state_dict_dir'] constructed as {setting['management']['state_dict_dir']}")

        if setting['management']['output_folder_dir'] == '':
            setting['management']['output_folder_dir'] = eval_base_folder_path + 'eval/'

        setting['management']['state_dict_dir'] =  setting['management']['state_dict_dir'] + '/' if setting['management']['state_dict_dir'][-1] != '/' else setting['management']['state_dict_dir']

    for k, v in setting['management'].items():
        if v == '':
            setting['management'][k] = None

    logger.info(f"Experiment ({setting['management']['exp_desc']}) input registered to setting.")

    return setting


def load_dataset(setting):

    class OneHot:
        def __init__(self, num_classes):
            self.num_classes = num_classes

        def __call__(self, labels):
            y = torch.eye(self.num_classes)
            return y[labels]

    def get_test_sample_label(testloader):

        itertest = iter(testloader)
        x_test, y_test = next(itertest)
        for x_test_batch, y_test_batch in itertest:
            x_test = torch.vstack((x_test, x_test_batch))
            y_test = torch.cat((y_test, y_test_batch))
        x_test = x_test.numpy()
        y_test = y_test.numpy()

        return x_test, y_test


    if setting['management']['dataset'] == 'cifar10':

        clip_values = (-2.4291, 2.7537)
        input_shape = (3, 32, 32)
        nb_classes = 10

        if setting['management']['using_normalization'] == True:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),])
            transform_test = transforms.Compose([
                transforms.ToTensor(),])

        onehot_target_transform = transforms.Compose([OneHot(num_classes=nb_classes)])

        trainset = torchvision.datasets.CIFAR10(root=setting['management']['dataset_dir'] , train=True, download=True, transform=transform_train, target_transform=onehot_target_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=setting['train_params']['train_batch_size'], shuffle=True, num_workers=2)

        train_data_gen = PyTorchDataGenerator(trainloader, size=len(trainset), batch_size=setting['train_params']['train_batch_size'])

        testset = torchvision.datasets.CIFAR10(root=setting['management']['dataset_dir'] , train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=setting['train_params']['test_batch_size'], shuffle=False, num_workers=2)

    elif setting['management']['dataset'] == 'tiny_imagenet':

        clip_values = None
        input_shape = (3, 32, 32)
        nb_classes = 200

        transform_train = transforms.Compose([
            # transforms.RandomSizedCrop(32),
            transforms.Resize(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
        ])

        transform_test = transforms.Compose([
            # transforms.Scale(32),
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
        ])

        onehot_target_transform = transforms.Compose([OneHot(num_classes=nb_classes)])

        trainset = torchvision.datasets.ImageFolder(root=setting['management']['dataset_dir']+'/train', transform=transform_train, target_transform=onehot_target_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=setting['train_params']['train_batch_size'], shuffle=True, num_workers=4, pin_memory=True)

        train_data_gen = PyTorchDataGenerator(trainloader, size=len(trainset), batch_size=setting['train_params']['train_batch_size'])

        testset = torchvision.datasets.ImageFolder(root=setting['management']['dataset_dir']+'/val', transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=setting['train_params']['test_batch_size'], shuffle=False, num_workers=4, pin_memory=True)



    elif setting['management']['dataset'] == 'imagenet':

        clip_values = None
        input_shape = (3, 224, 224)
        nb_classes = 1000


        if setting['management']['using_normalization'] == True:
            transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4824, 0.4467), (0.2471, 0.2435, 0.2616)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4824, 0.4467), (0.2471, 0.2435, 0.2616)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        onehot_target_transform = transforms.Compose([OneHot(num_classes=nb_classes)])

        trainset = torchvision.datasets.ImageFolder(root=setting['management']['dataset_dir']+'/train', transform=transform_train, target_transform=onehot_target_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=setting['train_params']['train_batch_size'], shuffle=True, num_workers=4, pin_memory=True)

        train_data_gen = PyTorchDataGenerator(trainloader, size=len(trainset), batch_size=setting['train_params']['train_batch_size'])

        testset = torchvision.datasets.ImageFolder(root=setting['management']['dataset_dir']+'/val', transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=setting['train_params']['test_batch_size'], shuffle=False, num_workers=4, pin_memory=True)


    elif setting['management']['dataset'] == 'imagenet_cifar_mock':
        clip_values = (-2.4291, 2.7537)
        input_shape = (3, 224, 224)
        nb_classes = 10


        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        onehot_target_transform = transforms.Compose([OneHot(num_classes=nb_classes)])

        trainset = torchvision.datasets.CIFAR10(root=setting['management']['dataset_dir'] , train=True, download=True, transform=transform_train, target_transform=onehot_target_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=setting['train_params']['train_batch_size'], shuffle=True, num_workers=2)

        train_data_gen = PyTorchDataGenerator(trainloader, size=len(trainset), batch_size=setting['train_params']['train_batch_size'])

        testset = torchvision.datasets.CIFAR10(root=setting['management']['dataset_dir'] , train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=setting['train_params']['test_batch_size'], shuffle=False, num_workers=2)


    elif setting['management']['dataset'] == 'imagenet_val_only':

        clip_values = None
        input_shape = (3, 224, 224)
        nb_classes = 1000

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4467), (0.2471, 0.2435, 0.2616)),
        ])

        testset = torchvision.datasets.ImageFolder(root=setting['management']['dataset_dir']+'/val', transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=setting['train_params']['test_batch_size'], shuffle=False, num_workers=4, pin_memory=True)

        train_data_gen = None
        trainloader = None


    else:
        logger.error(f"Dataset <{setting['management']['dataset']}> is not supported.")

    # x_test, y_test = get_test_sample_label(testloader)
    return train_data_gen, testloader, clip_values, input_shape, nb_classes, trainloader

def load_model(setting, logger):
    if setting['management']['load_state_dict'] == False:
        print(setting['management']['model_dir'])
        model = torch.load(setting['management']['model_dir'])
        input_model_saving_path = setting['management']['output_folder_dir'] + 'input_model.pt'
        torch.save(model, input_model_saving_path, pickle_module=dill)
        logger.info(f"Model {setting['management']['model_dir']} (type: {type(model)} loaded and saved to {input_model_saving_path}.")
    else:
        model_module = importlib.import_module(setting['management']['load_state_dict']['model_import_command'])
        model_method = getattr(model_module, setting['management']['load_state_dict']['model_method_name'])

        model = model_method()
        model.load_state_dict(torch.load(setting['management']['model_dir']))

        input_state_dict_saving_path = setting['management']['output_folder_dir'] + 'input_model.state_dict'
        torch.save(model.state_dict(), input_state_dict_saving_path)
        logger.info(f"Model (module: {setting['management']['load_state_dict']['model_import_command']}; method: {setting['management']['load_state_dict']['model_method_name']}; type: {type(model)}) loaded from state dict {setting['management']['model_dir']} and saved to {input_state_dict_saving_path}.")

    return model

def process_best_acc(classifier, setting, acc_key, current_acc, current_epoch_num):
    return_msg = ''
    if acc_key not in setting['results']['best'] or (acc_key in setting['results']['best'] and current_acc >= setting['results']['best'][acc_key][0]):
        setting['results']['best'][acc_key] = (current_acc, current_epoch_num)

        best_ckpt_save_path = setting['management']['output_folder_dir'] + setting['management']['save_subdir'] + '/best_ckpts/'
        if not os.path.isdir(best_ckpt_save_path):
            os.makedirs(best_ckpt_save_path)
        if setting['management']['load_state_dict'] == False:
            best_ckpt_filename = f'best_{acc_key}.pt'
            classifier.save(filename = best_ckpt_filename, path = best_ckpt_save_path)
        else:
            best_ckpt_filename = f'best_{acc_key}.pt'
            best_ckpt_save_path += f'best_{acc_key}.state_dict'
            torch.save(classifier.model.state_dict(), best_ckpt_save_path)

        return_msg = f'\tNew best model ckpt {best_ckpt_filename} saved.'

    epoch_ckpt_save_path = setting['management']['output_folder_dir'] + setting['management']['save_subdir'] + '/epoch_ckpts/'
    if not os.path.isdir(epoch_ckpt_save_path):
        os.makedirs(epoch_ckpt_save_path)
    torch.save(classifier.model.state_dict(), f'{epoch_ckpt_save_path}epoch_{current_epoch_num}.state_dict')
    return return_msg

def save_most_recent_art_classifier(classifier, epoch_num, setting):
    most_recent_epoch_info_dict = {
        'art_classifier': classifier,
        'model': classifier._model._model,
        'optimizer': classifier._optimizer,
        'scheduler': classifier._scheduler,
        'epoch_num': epoch_num,
    }

    recent_epoch_ckpt_save_path = setting['management']['output_folder_dir'] + setting['management']['save_subdir'] + '/most_recent_epoch/'
    if not os.path.isdir(recent_epoch_ckpt_save_path):
        os.makedirs(recent_epoch_ckpt_save_path)

    # torch.save(most_recent_epoch_info_dict, recent_epoch_ckpt_save_path + 'most_recent_epoch.info_dict', pickle_module=dill)
    torch.save(most_recent_epoch_info_dict, recent_epoch_ckpt_save_path + 'most_recent_epoch.info_dict')
    classifier.save(filename = 'most_recent_epoch.pt', path = recent_epoch_ckpt_save_path)

    with open(recent_epoch_ckpt_save_path + 'most_recent_epoch_output_setting.json', "w+") as recent_epoch_out_setting_f:
        json.dump(setting, recent_epoch_out_setting_f, indent = 4)


def classifier_predict(classifier, testloader, test_batch_size = 64):
    correct = 0
    total = 0
    for batch_idx, (x_test, y_test) in enumerate(testloader):
        x_test, y_test = x_test.numpy(), y_test.numpy()

        predictions = classifier.predict(x_test, batch_size = test_batch_size)
        correct += np.sum(np.argmax(predictions, axis=1) == y_test, axis=0)
        total += len(y_test)

    test_acc = correct/total
    return test_acc


def classifier_adv_predict(classifier, testloader, logger, test_batch_size, attack_method_name = None, attack_method_params = None):
    correct = 0
    total = 0
    attack_info = ''
    for batch_idx, (x_test, y_test) in enumerate(testloader):
        x_test, y_test = x_test.numpy(), y_test.numpy()
        x_test_adv, attack_info = generate_attack(attack_method_name, attack_method_params, classifier, x_test, logger, target = y_test)

        predictions = classifier.predict(x_test_adv, batch_size = test_batch_size)
        correct += np.sum(np.argmax(predictions, axis=1) == y_test, axis=0)
        total += len(y_test)

    adv_test_acc = correct/total
    return adv_test_acc, attack_info

def eval_classifier(classifier, epoch_num = -1, **kwargs):

    testloader = kwargs['testloader']
    # x_test = kwargs['x_test']
    # y_test = kwargs['y_test']
    setting = kwargs['setting']
    logger = kwargs['logger']
    attack_flag = kwargs['attack_flag']
    if 'current_lr' in kwargs:
        current_lr = kwargs['current_lr']
        logger.info(f'Epoch #{epoch_num} trained (lr = {current_lr}).')
    print_flag = kwargs.get('print_flag', False)

    setting['results'][epoch_num] = dict()
    # epoch_train_acc = kwargs['epoch_train_acc']

    # predictions = classifier.predict(x_test, batch_size = setting['train_params']['test_batch_size'])
    # benign_test_acc = np.sum(np.argmax(predictions, axis=1) == y_test, axis=0) / len(y_test)
    benign_test_acc = classifier_predict(classifier, testloader, test_batch_size = setting['train_params']['test_batch_size'])

    setting['results'][epoch_num]['benign_test_acc'] = benign_test_acc


    best_benign_acc_info = process_best_acc(setting = setting, classifier = classifier, acc_key = 'benign_test_acc', current_acc = benign_test_acc, current_epoch_num = epoch_num)




    logger.info(f"\tBenign test acc: {benign_test_acc * 100 :.2f}% \t(Best: {setting['results']['best']['benign_test_acc'][0] * 100 :.2f}% from Epoch #{setting['results']['best']['benign_test_acc'][1]}) {best_benign_acc_info}")

    if setting['management']['task'] not in ['test', 'offline_loop_test']:
        save_most_recent_art_classifier(classifier = classifier, epoch_num = epoch_num, setting = setting)
    save_most_recent_art_classifier(classifier = classifier, epoch_num = epoch_num, setting = setting)

    if attack_flag:
        # if epoch_num >= setting['management']['adv_attack_init_epoch'] or setting['results']['best']['benign_test_acc'][1] == epoch_num or epoch_num == 0:
        if epoch_num >= setting['management']['adv_attack_init_epoch']:
            for attack in setting['attack_params']['attack_methods_list']:

                # x_test_adv, attack_info = generate_attack(attack['attack_method_name'], attack['attack_method_params'], classifier, x_test, logger, target = y_test)
                # predictions = classifier.predict(x_test_adv, batch_size = setting['train_params']['test_batch_size'])
                # adv_test_acc = np.sum(np.argmax(predictions, axis=1) == y_test, axis=0) / len(y_test)
                adv_test_acc, attack_info = classifier_adv_predict(classifier, testloader, logger, test_batch_size = setting['train_params']['test_batch_size'], attack_method_name = attack['attack_method_name'], attack_method_params = attack['attack_method_params'])

                setting['results'][epoch_num][attack_info] = adv_test_acc

                best_adv_acc_info = process_best_acc(setting = setting, classifier = classifier, acc_key = attack_info, current_acc = adv_test_acc, current_epoch_num = epoch_num)

                logger.info(f"\tAdv test acc ({attack_info}): {adv_test_acc * 100 :.2f} \t(Best: {setting['results']['best'][attack_info][0] * 100 :.2f} from Epoch #{setting['results']['best'][attack_info][1]}) {best_adv_acc_info}")


def generate_attack(attack_method_name, attack_method_params, classifier, x_test, logger, target = None):

    if attack_method_name == 'FGSM':
        FGSM_eps = attack_method_params['eps']
        attack = FastGradientMethod(estimator=classifier, eps=FGSM_eps)
        x_test_adv = attack.generate(x=x_test, t=target)
        attack_info = f'<{attack_method_name} with eps={FGSM_eps}>'
    elif attack_method_name == 'PGD':
        eps = attack_method_params['eps']
        eps_step = attack_method_params['eps_step']
        max_iter = attack_method_params['max_iter']
        num_random_init = attack_method_params['num_random_init']
        attack = ProjectedGradientDescent(estimator=classifier, eps=eps, eps_step=eps_step, max_iter=max_iter, num_random_init=num_random_init, verbose = False)
        x_test_adv = attack.generate(x=x_test, y=target)
        attack_info = f'<{attack_method_name} with eps = {eps}, eps_step = {eps_step}, max_iter = {max_iter}, num_random_init = {num_random_init}>'
    elif attack_method_name == 'HSJ':
        max_iter = attack_method_params['max_iter']
        max_eval = attack_method_params['max_eval']
        init_eval = attack_method_params['init_eval']
        norm = attack_method_params['norm']
        if attack_method_params['norm'] == "Inf":
            attack = HopSkipJump(classifier=classifier, max_iter=max_iter, max_eval=max_eval, init_eval=init_eval, norm=np.Inf, verbose=True)
        else:
            attack = HopSkipJump(classifier=classifier, max_iter=max_iter, max_eval=max_eval, init_eval=init_eval, norm=attack_method_params['norm'], verbose=True)
        x_test_adv = attack.generate(x=x_test)
        attack_info = f'<{attack_method_name} with max_iter = {max_iter}, max_eval = {max_eval}, init_eval = {init_eval}, norm = {norm}>'
    elif attack_method_name == 'SMM':
        theta = attack_method_params['theta']
        gamma = attack_method_params['gamma']
        attack = SaliencyMapMethod(classifier=classifier, theta=theta, gamma=gamma, verbose=True)
        x_test_adv = attack.generate(x=x_test)
        attack_info = f'<{attack_method_name} with theta = {theta}, gamma = {gamma}>'

    else:
        logger.info(f'\tSpecified adversarial attack <{attack_method_name}> is invalid.')

    # logger.info(f'\tAdversarial attack {attack_info} generated.')

    return x_test_adv, attack_info

def pprint_best_models(setting, logger):
    logger.info("Reporting best performance...")
    for best_acc_criteria, (best_acc, best_epoch_num) in setting['results']['best'].items():
        logger.info(f"\tBest {best_acc_criteria}: {best_acc * 100 :.2f} from Epoch #{best_epoch_num}.")
        for epoch_acc_criteria, epoch_acc in setting['results'][best_epoch_num].items():
            if epoch_acc_criteria != best_acc_criteria:
                logger.info(f"\t\t{epoch_acc_criteria}: {epoch_acc * 100 :.2f}.")

def save_trained_model(model, filename, setting, logger):
    trained_model_saving_path = setting['management']['output_folder_dir'] + f'{filename}.pt'
    if setting['management']['load_state_dict'] == False:
        # torch.save(model, trained_model_saving_path, pickle_module=dill)
        torch.save(model, trained_model_saving_path)
        logger.info(f"Model {setting['management']['model_dir']} trained and saved to {trained_model_saving_path}.")
    else:
        trained_model_state_dict_saving_path = trained_model_saving_path[:-3] + '.state_dict'
        torch.save(model.state_dict(), trained_model_state_dict_saving_path)
        logger.info(f"Model state dict {setting['management']['model_dir']} trained and saved to {trained_model_state_dict_saving_path}.")


def make_lr_scheduler(optimizer, setting, logger):

    def make_single_lr_scheduler(scheduler_type, scheduler_setting):

        if scheduler_type == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = scheduler_setting['lr_step_size'], gamma = scheduler_setting['gamma'])
        elif scheduler_type == 'MultiStepLR':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = scheduler_setting['milestones'], gamma = scheduler_setting['gamma'])
        elif scheduler_type == 'ConstantLR':
            scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor = scheduler_setting['factor'], total_iters = scheduler_setting['total_iters'])
        else:
            logger.error(f"Invalid scheduler type given: <{scheduler_type}>")

        return scheduler

    if setting['train_params']['scheduler_type'] != 'SequentialLR':
        return make_single_lr_scheduler(setting['train_params']['scheduler_type'], setting['train_params'])

    else:
        schedulers_list = []
        for a_scheduler_setting in setting['train_params']['scheduler_list']:
            schedulers_list.append(make_single_lr_scheduler(a_scheduler_setting['scheduler_type'], a_scheduler_setting))

        return optim.lr_scheduler.SequentialLR(optimizer, schedulers = schedulers_list, milestones = setting['train_params']['milestones'])



def get_eval_worthy_epochs(setting, logger):

    with open(setting['management']['output_setting_dir']) as eval_output_setting_f:
        eval_output_setting = json.load(eval_output_setting_f)
        logger.info(f"Target state dict eval output setting file {setting['management']['output_setting_dir']} loaded.")

    best_acc_LUT = dict()
    online_criteria = eval_output_setting['results']['best'].keys()
    for criterion in online_criteria:
        best_acc_LUT[criterion] = (-1, -1) #acc, epoch

    logger.info(f'Existed eval criteria are: {online_criteria}.')

    for epoch_i in range(setting['management']['state_dict_eval_init_epoch'], setting['management']['state_dict_eval_end_epoch'] + 1, setting['management']['state_dict_eval_inter']):
        for criterion in online_criteria:
            try:
                if eval_output_setting['results'][str(epoch_i)][criterion] >= best_acc_LUT[criterion][0]:
                    best_acc_LUT[criterion] = (eval_output_setting['results'][str(epoch_i)][criterion], epoch_i)
                    # logger.info(f"Best {criterion}: {eval_output_setting['results'][str(epoch_i)][criterion]} found in epoch #{epoch_i} (previously from {best_acc_LUT[criterion][1]}: {best_acc_LUT[criterion][0]* 100 :.2f})")
                    # print(eval_output_setting['results'][str(epoch_i)][criterion])
            except KeyError:
                # print('key error here')
                pass


    eval_worthy_state_dict_base_path = setting['management']['state_dict_dir']
    eval_worthy_state_dict_full_path_list = []

    for criterion, criterion_result in best_acc_LUT.items():
        epoch_index = criterion_result[1]

        state_dict_file_name = setting['management']['state_dict_prefix'] + str(epoch_index) + setting['management']['state_dict_suffix']
        state_dict_full_path = eval_worthy_state_dict_base_path + state_dict_file_name
        eval_worthy_state_dict_full_path_list.append((epoch_index, state_dict_full_path))

    eval_worthy_state_dict_full_path_list = sorted(list(set(eval_worthy_state_dict_full_path_list)), key = lambda x:x[0])

    logger.info(f'Epochs to evaluate: {[i[0] for i in eval_worthy_state_dict_full_path_list]}')
    return eval_worthy_state_dict_full_path_list







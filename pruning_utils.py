import torch
import torch.optim as optim

special_loss_methods = {'GAL', 'ADMM'}

def load_model(method_name, model, setting, logger):
    if method_name == "SFP":
        return SFP.init_net(model, setting, logger)
    elif method_name == 'GAL':
        return GAL.init_net(model, setting, logger)
    elif method_name == 'HRank':
        return HRank.init_net(model, setting, logger)
    elif method_name == "ADMM":
        return ADMM.init_net(model, setting, logger)
    elif method_name == "FPGM":
        return FPGM.init_net(model, setting, logger)
    elif method_name == 'DHP':
        return DHP.DHP.init_net(model, setting, logger)

def load_optimizer(method_name, model, setting):

    if method_name == 'GAL':
        return GAL.load_optimizer(model, setting)
    if method_name == "FPGM":
        return FPGM.load_potimizer(model, setting)
    elif method_name == 'DHP':
        return DHP.DHP.load_optimizer(model, setting)
    else:
        return optim.SGD(model.model.parameters(), lr = setting['train_params']['lr'], momentum = setting['train_params']['momentum'], weight_decay = setting['train_params']['weight_decay'])

def do_prune(method_name, classifier, model, setting, logger, optimizer, scheduler, epoch=1, i=1):
    if method_name == "SFP":
        SFP.do_prune(model, setting, logger, epoch, optimizer)
    elif method_name == 'HRank':
        HRank.do_prune(classifier, model, setting, logger, epoch, i)
    elif method_name == "FPGM":
        FPGM.do_prune(model, setting, logger, epoch, optimizer)
    elif method_name == 'DHP':
        DHP.DHP.do_prune(classifier, model, setting, logger, epoch, i)
    elif method_name == 'GAL':
        GAL.do_prune(model, setting, logger, epoch)
    else:
        return

def pre_prune(method_name, model, setting, logger, inputs, targets):
    if method_name == 'GAL':
        GAL.pre_prune(model, setting, logger, inputs, targets)
    else:
        return

def post_train(method_name, model, setting, logger, epoch, i, lr):
    if method_name == 'GAL':
        GAL.post_train(model, setting, logger, epoch, i)
    elif method_name == 'HRank':
        HRank.post_train(model, setting, logger, epoch, i)
    elif method_name == 'DHP':
        return DHP.DHP.post_train(model, setting, logger, epoch, i, lr)


def special_loss(method_name, model, setting, logger, inputs, targets, epoch, optimizer, batch_index, classifier):
    if method_name == 'GAL':
        return GAL.special_loss(model, setting, logger, inputs, targets)
    elif method_name == "ADMM":
        return ADMM.special_loss(model, setting, logger, inputs, targets, epoch, optimizer, batch_index, classifier)
    else:
        return


def mask_loss(method_name, model, setting, logger, epoch, i):
    if method_name == "ADMM":
        ADMM.mask_loss(model, setting, logger, epoch, i)
    if method_name == "FPGM":
        model.do_grad_mask()
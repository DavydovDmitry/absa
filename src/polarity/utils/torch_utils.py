import torch


def get_optimizer(name, parameters, lr, l2=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=l2)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr, weight_decay=l2)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=l2)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=l2)
    elif name == 'adadelta':
        return torch.optim.Adadelta(parameters, lr=lr, weight_decay=l2)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))

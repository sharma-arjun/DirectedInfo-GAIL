import math
import numpy as np
import torch


def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (
        2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1)


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grad_from(net, grad_grad=False):
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad

def clip_grads(model, clip_val=5.0):
    #average_abs_norm = 0
    #max_abs_norm = 0
    #count = 0.0
    for p in model.parameters():
        if p.grad is not None:
            #count += 1.0
            #average_abs_norm += p.grad.data.abs().mean()
            #if p.grad.data.abs().max() > max_abs_norm:
            #    max_abs_norm = p.grad.data.abs().max()
            p.grad.data = p.grad.data.clamp(-clip_val, clip_val)

    #average_abs_norm /= count

    #return average_abs_norm, max_abs_norm

def get_norm_scalar_for_layer(layer, norm_p):
    return layer.norm(p=norm_p).cpu().data.numpy()[0]

def get_norm_inf_scalar_for_layer(layer):
    data = layer.data.cpu().numpy()
    return np.max(np.abs(data))
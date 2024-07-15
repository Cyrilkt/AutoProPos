# -*- coding: UTF-8 -*-
import torch
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from typing import Union
import os.path as osp
import os
import time
from torchvision.utils import save_image
import torch.distributed as dist
import math
import inspect
#from torch._six import string_classes
import six
string_classes = six.string_types
import six
import collections.abc as container_abcs
import warnings
from utils.ops import load_network




def get_varname(var):
    for fi in reversed(inspect.stack()):
        try:
            names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
            
            if names:
                return names[0]
        except UnicodeDecodeError:
            continue  # Skip frames that cause encoding issues
    return "Unknown Variable"  # Return a default name if none is found



def reduce_tensor(rt):
    rt = rt.clone()
    if dist.is_initialized():
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    else:
        world_size = 1
    rt /= world_size
    return rt


class LoggerX(object):

    def __init__(self, save_root, enable_wandb=False, **kwargs):
        assert dist.is_initialized()
        self.models_save_dir = osp.join(save_root, 'save_models')
        self.images_save_dir = osp.join(save_root, 'save_images')
        self.best_model_save_dir=osp.join(save_root,'save_best_model')
        if kwargs['config'].save_checkpoints:
          os.makedirs(self.models_save_dir, exist_ok=True)
          os.makedirs(self.images_save_dir, exist_ok=True)
        if kwargs['config'].save_best_model  :
          self.best_model_save_dir=osp.join(save_root,'save_best_model')
          os.makedirs(self.best_model_save_dir,exist_ok=True)
        self._modules = []
        self._module_names = []
        self.world_size = dist.get_world_size()
        self.local_rank = dist.get_rank()
        self.enable_wandb = enable_wandb
        if enable_wandb and self.local_rank == 0:
            import wandb
            wandb.init(dir=save_root, settings=wandb.Settings(_disable_stats=True, _disable_meta=True), **kwargs)

    @property
    def modules(self):
        return self._modules

    @property
    def module_names(self):
        return self._module_names

    @modules.setter
    def modules(self, modules):
        #self._module_names=['byol','optimizer']*len(modules)
        for i in range(len(modules)):
            self._modules.append(modules[i])
            self._module_names.append(get_varname(modules[i]))

    def append(self, module, name=None):
        self._modules.append(module)
        if name is None:
            name = get_varname(module)
        self._module_names.append(name)

    def checkpoints(self, epoch,cluster_centers=None,save_best=False):
        if self.local_rank != 0:
            return
        if not save_best:
          torch.save(cluster_centers,osp.join(self.models_save_dir, "cluster_centers.pt"))
          for i in range(len(self.modules)):
              module_name = self.module_names[i]
              module = self.modules[i]
              torch.save(module.state_dict(), osp.join(self.models_save_dir, '{}-{}'.format(module_name, epoch))) 
        else:
          torch.save(cluster_centers,osp.join(self.best_model_save_dir, "cluster_centers.pt"))
          for i in range(len(self.modules)):
              module_name = self.module_names[i]
              module = self.modules[i]
              torch.save(module.state_dict(), osp.join(self.best_model_save_dir, '{}-{}'.format(module_name, "save_best")))

    def load_checkpoints(self, epoch,cluster_centers=None,load_best=False):
        if not load_best:
          for i in range(len(self.modules)):
              module_name = self.module_names[i]
              module = self.modules[i]
              module.load_state_dict(load_network(osp.join(self.models_save_dir, '{}-{}'.format(module_name, epoch))))
          cluster_centers=torch.load(osp.join(self.models_save_dir, "cluster_centers.pt"))
          return(cluster_centers) 
        else:
          for i in range(len(self.modules)):
              module_name = self.module_names[i]
              module = self.modules[i]
              module.load_state_dict(load_network(osp.join(self.best_model_save_dir, '{}-{}'.format(module_name, "save_best")))) 
          cluster_centers=torch.load(osp.join(self.best_model_save_dir, "cluster_centers.pt"))    
          return(cluster_centers)
    
    def msg(self, stats, step):
        pass
    """
    def msg(self, stats, step):
        output_str = '[{}] {:05d}, '.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), step)
        output_dict = {}
        for i in range(len(stats)):
            if isinstance(stats, (list, tuple)):
                var = stats[i]
                var_name = get_varname(stats[i])
            elif isinstance(stats, dict):
                var_name, var = list(stats.items())[i]
            else:
                raise NotImplementedError
            if isinstance(var, torch.Tensor):
                var = var.detach().mean()
                var = reduce_tensor(var)
                var = var.item()
            output_str += '{} {:2.5f}, '.format(var_name, var)
            output_dict[var_name] = var

        if self.enable_wandb and self.local_rank == 0:
            import wandb
            wandb.log(output_dict, step)

        if self.local_rank == 0:
            print(output_str)
    """
    def msg_str(self, output_str):
        if self.local_rank == 0:
            print(str(output_str))

    def save_image(self, grid_img, n_iter, sample_type):
        save_image(grid_img, osp.join(self.images_save_dir,
                                      '{}_{}_{}.jpg'.format(n_iter, self.local_rank, sample_type)),
                   nrow=1)

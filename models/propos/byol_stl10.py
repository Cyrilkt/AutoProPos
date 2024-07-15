# -*- coding: UTF-8 -*-

import argparse
import copy

import torch
import numpy as np
from utils.ops import convert_to_ddp
from utils.ops import convert_to_cuda, is_root_worker, dataset_with_indices
from .byol_wrapper_stl10 import BYOLWrapper
from models.basic_template import TrainTask
from network import backbone_dict
from models import model_dict
from torchvision import transforms
import tqdm


@model_dict.register('propos_stl')
class BYOL(TrainTask):
    __BYOLWrapper__ = BYOLWrapper

    def set_model(self):
        opt = self.opt
        encoder_type, dim_in = backbone_dict[opt.encoder_name]
        encoder = encoder_type()
        byol = self.__BYOLWrapper__(encoder, in_dim=dim_in, num_cluster=self.num_cluster, temperature=opt.temperature,
                                    hidden_size=opt.hidden_size, fea_dim=opt.feat_dim, byol_momentum=opt.momentum_base,
                                    symmetric=opt.symmetric, shuffling_bn=opt.shuffling_bn, latent_std=opt.latent_std,
                                    queue_size=opt.queue_size)
        if opt.syncbn:
            if opt.shuffling_bn:
                byol.encoder_q = torch.nn.SyncBatchNorm.convert_sync_batchnorm(byol.encoder_q)
                byol.projector_q = torch.nn.SyncBatchNorm.convert_sync_batchnorm(byol.projector_q)
                byol.predictor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(byol.predictor)
            else:
                byol = torch.nn.SyncBatchNorm.convert_sync_batchnorm(byol)
        if opt.lars:
            from utils.optimizers import LARS
            optim = LARS
        else:
            optim = torch.optim.SGD
        optimizer = optim(params=self.collect_params(byol, exclude_bias_and_bn=opt.exclude_bias_and_bn),
                          lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)

        self.logger.modules = [byol, optimizer]
        # Initialization
        self.feature_extractor_copy = copy.deepcopy(byol.encoder).cuda()
        byol = byol.cuda()
        self.feature_extractor = byol.encoder
        byol = convert_to_ddp(byol)
        self.byol = byol
        self.optimizer = optimizer

    @staticmethod
    def build_options():
        parser = argparse.ArgumentParser('Private arguments for training of different methods')
        # SSL
        parser.add_argument('--symmetric', help='Symmetric contrastive loss', dest='symmetric', action='store_true')
        parser.add_argument('--hidden_size', help='hidden_size', type=int, default=4096)
        parser.add_argument('--fix_predictor_lr', help='fix the lr of predictor', action='store_true')
        parser.add_argument('--lambda_predictor_lr', help='lambda the lr of predictor', type=float, default=10.)
        parser.add_argument('--shuffling_bn', help='shuffling_bn', action='store_true')

        parser.add_argument('--momentum_base', help='ema momentum min', type=float, default=0.996)
        parser.add_argument('--momentum_max', help='ema momentum max', type=float, default=1.0)
        parser.add_argument('--momentum_increase', help='momentum_increase', action='store_true')

        parser.add_argument('--exclude_bias_and_bn', help='exclude_bias_and_bn', action='store_true')
        parser.add_argument('--lars', help='lars', action='store_true')
        parser.add_argument('--syncbn', help='syncbn', action='store_true')
        parser.add_argument('--byol_transform', help='byol_transform', action='store_true')

        # LOSS
        parser.add_argument('--cluster_loss_weight', type=float, default=1.0, help='weight for cluster loss')
        parser.add_argument('--latent_std', type=float, help='latent_std', default=0.0)
        parser.add_argument('--temperature', type=float, default=0.5, help='temperature')
        parser.add_argument('--queue_size', type=int, help='queue_size', default=0)
        parser.add_argument('--v2', help='v2', action='store_true')

        return parser
    
    def set_loader(self):
        opt = self.opt
        dataset_name = opt.dataset

       
        normalize = self.normalize(opt.dataset)

        train_transform = self.train_transform(normalize)
        self.logger.msg_str('set transforms...')
        self.logger.msg_str(train_transform)

        self.logger.msg_str('set train and unlabeled dataloaders...')
        train_loader, labels, train_sampler = self.build_dataloader(
            dataset_name=opt.dataset,
            transform=train_transform,
            batch_size=opt.batch_size,
            shuffle=True,
            drop_last=True,
            sampler=True,
            train=True)
        unlabeled_loader, _, unlabeled_sampler = self.build_dataloader(
            dataset_name=opt.dataset,
            transform=train_transform,
            batch_size=opt.batch_size,
            shuffle=True,
            drop_last=True,
            sampler=True,
            train=False,
            unlabeled=True)

        test_transform = []
        if 'imagenet' in dataset_name:
            test_transform += [transforms.Resize(256), transforms.CenterCrop(224)]
        
        test_transform += [
            transforms.Resize(opt.img_size),
            transforms.ToTensor(),
            normalize
        ]

        test_transform = transforms.Compose(test_transform)

        
        self.logger.msg_str('set memory dataloaders...')
        memory_loader = self.build_dataloader(opt.dataset,test_transform, train=True, batch_size=opt.batch_size, sampler=True)[0]
        
        if not opt.whole_dataset :
           test_loader = self.build_dataloader(opt.dataset,
                                            test_transform,
                                            train=False,
                                            sampler=True,
                                            batch_size=opt.batch_size)[0]
           self.logger.msg_str(f'set test dataloader with {len(test_loader)} iterations...')
           self.test_loader=test_loader
        
        self.train_loader = train_loader
        self.memory_loader = memory_loader
        self.unlabeled_loader = unlabeled_loader
        self.train_sampler = train_sampler
        self.unlabeled_sampler = unlabeled_sampler
        self.iter_per_epoch = len(train_loader) + len(unlabeled_loader)
        self.num_classes = len(np.unique(labels))
        self.num_samples = len(labels)
        self.indices = torch.zeros(len(self.train_sampler), dtype=torch.long).cuda()
        self.num_cluster = self.num_classes if opt.num_cluster is None else opt.num_cluster
        self.psedo_labels = torch.zeros((self.num_samples,)).long().cuda()

        self.logger.msg_str('load {} images...'.format(self.num_samples))
    
    def fit(self):
        opt = self.opt
        # training routine
        self.progress_bar = tqdm.tqdm(total=self.iter_per_epoch * opt.epochs,disable=not self.verbose)

        n_iter = self.iter_per_epoch * opt.resume_epoch + 1
        self.progress_bar.update(n_iter)
        max_iter = opt.epochs * self.iter_per_epoch
        i=0
        while True:
            epoch = int(n_iter // self.iter_per_epoch + 1)
            self.train_sampler.set_epoch(epoch)
            self.unlabeled_sampler.set_epoch(epoch)

            for inputs in self.unlabeled_loader:
                inputs = convert_to_cuda(inputs)
                self.train_unlabeled(inputs, n_iter)
                self.progress_bar.refresh()
                self.progress_bar.update()
                n_iter += 1

            apply_kmeans = epoch % opt.reassign == 0
            if apply_kmeans:
                self.psedo_labeling(n_iter)

            self.indices.copy_(torch.Tensor(list(iter(self.train_sampler))))
            for inputs in self.train_loader:
                inputs = convert_to_cuda(inputs)
                self.adjust_learning_rate(n_iter)
                self.train(inputs, n_iter)
                self.progress_bar.refresh()
                self.progress_bar.update()
                n_iter += 1
            
            if  i<len(opt.epochs_cluster_analysis) and opt.clusternet:
                if self.cur_epoch==opt.epochs_cluster_analysis[i] :
                    print('self.get_new_k before')
                    self.get_new_k(split=opt.clusternet_training_data,index=i) 
                    print('after')
                    i+=1
            self.cur_epoch += 1
            
            if self.cur_epoch > opt.epochs:
                if torch.distributed.get_rank() == 0 :
                  if opt.whole_dataset:
                    print('Evaluation Best Model WHOLE DATASET :',self.best_results)
                  else:
                    print('Evaluation Best Model TEST SET :',self.best_results_test)
                #print('best acc setup',self.best_acc_setup)
                #print('best_acc_epoch',self.best_acc_epoch)
                #print('best_acc', self.best_acc)
                break
            

    def train(self, inputs, n_iter):
        opt = self.opt

        images, labels = inputs
        self.byol.train()

        im_q, im_k = images

        _start = ((n_iter - 1) % self.iter_per_epoch - len(self.unlabeled_loader)) * opt.batch_size
        indices = self.indices[_start: _start + opt.batch_size]
        
        
        self.byol.module.psedo_labels = self.psedo_labels
        self.byol.module.num_cluster=self.num_cluster
        
        is_warmup = not self.cur_epoch > opt.warmup_epochs

        self.byol.module.latent_std = opt.latent_std * float(not is_warmup)
        # compute loss
        with torch.autocast('cuda', enabled=opt.amp):
            contrastive_loss, cluster_loss_batch, q = self.byol(
                im_q, im_k, indices,True, opt.v2)

        loss = contrastive_loss 

        ####
        
        if ((n_iter - 1) / self.iter_per_epoch) > opt.warmup_epochs:
            loss += cluster_loss_batch * opt.cluster_loss_weight

        self.optimizer.zero_grad()
        # SGD
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            q_std = torch.std(q.detach(), dim=0).mean()

        outputs = [contrastive_loss, cluster_loss_batch, q_std]
        self.logger.msg(outputs, n_iter)

    def train_unlabeled(self, inputs, n_iter):
        opt = self.opt

        images, labels = inputs
        self.byol.train()

        im_q, im_k = images
        _start = ((n_iter - 1) % self.iter_per_epoch - len(self.unlabeled_loader)) * opt.batch_size
        # indices ne sert a rien dans ce cas car nous computons que contrastive_loss
        indices = self.indices[_start: _start + opt.batch_size]

        # compute loss
        self.byol.module.psedo_labels = self.psedo_labels
        self.byol.module.num_cluster=self.num_cluster
        #print('self.predo_labels',self.psedo_labels)
        with torch.autocast('cuda', enabled=opt.amp):
            unlabeled_contrastive_loss, _, q = self.byol(
                    im_q, im_k, indices,True, opt.v2,unlabeled=True)
        self.optimizer.zero_grad()
        # SGD
        unlabeled_contrastive_loss.backward()
        self.optimizer.step()

        outputs=[unlabeled_contrastive_loss]
        self.logger.msg(outputs, n_iter)

    def adjust_learning_rate(self, n_iter):
        opt = self.opt
        lr = self.cosine_annealing_LR(n_iter)
        if opt.fix_predictor_lr:
            predictor_lr = opt.learning_rate
        else:
            predictor_lr = lr * opt.lambda_predictor_lr
        flag = False
        for param_group in self.optimizer.param_groups:
            if 'predictor' in param_group['name']:
                flag = True
                param_group['lr'] = predictor_lr
            else:
                param_group['lr'] = lr
        assert flag

        ema_momentum = opt.momentum_base
        if opt.momentum_increase:
            ema_momentum = opt.momentum_max - (opt.momentum_max - ema_momentum) * (
                    np.cos(np.pi * n_iter / (opt.epochs * self.iter_per_epoch)) + 1) / 2
        self.byol.module.m = ema_momentum

        self.logger.msg([lr, predictor_lr, ema_momentum], n_iter)

    def train_transform(self, normalize):
        opt = self.opt
        if not opt.byol_transform:
            return super().train_transform(normalize)
        from torchvision import transforms
        from utils import TwoCropTransform
        base_transform = [
            transforms.RandomResizedCrop(size=opt.img_size, scale=(opt.resized_crop_scale, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]

        train_transform1 = base_transform + [
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=1.0),
            transforms.RandomSolarize(128, p=0.0)
        ]
        train_transform2 = base_transform + [
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.1),
            transforms.RandomSolarize(128, p=0.2)
        ]
        train_transform1 += [transforms.ToTensor(), normalize]
        train_transform2 += [transforms.ToTensor(), normalize]

        train_transform1 = transforms.Compose(train_transform1)
        train_transform2 = transforms.Compose(train_transform2)
        train_transform = TwoCropTransform(train_transform1, train_transform2)
        return train_transform

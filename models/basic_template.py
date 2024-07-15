from __future__ import print_function

import os
import os.path as osp
import argparse
import warnings

import torch
import torchvision.datasets
from torchvision import transforms
import numpy as np
import torch.distributed as dist
import tqdm

from utils import TwoCropTransform, extract_features,concat_all_gather
from utils.ops import convert_to_cuda, is_root_worker, dataset_with_indices
from utils.grad_scaler import NativeScalerWithGradNormCount
from utils.loggerx import LoggerX
import torch_clustering
from ClusterAnalysis import ClusterAnalysis
from torch.utils.data import DataLoader,TensorDataset,Dataset
from scipy.optimize import linear_sum_assignment as linear_assignment
from PIL import Image



class ExpandChannelsTo3:
    """Transform to convert images with 1 channel to 3 channels by replicating the single channel."""

    def __call__(self, img):
        """
        Apply the transformation.

        Args:
            img (PIL Image or Tensor): The image to transform.

        Returns:
            PIL Image or Tensor: Transformed image with 3 channels.
        """
        #print('img.mode == L',img.mode == 'L')
        if img.mode == 'L':  # Check if it's a grayscale image
            img = img.convert('RGB')  # Convert grayscale to RGB by replicating channels
        return img


class PTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images  # assuming images are stored as PIL images or numpy arrays
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)


def parse_tuples(arg):
  
  if arg=='None':
    return None
  
  try:
      # This will evaluate the string as a Python literal if it's correctly formatted
      data = ast.literal_eval(arg)
      # Validate that it's a list of tuples and each tuple has three float elements
      if not all(isinstance(x, tuple) for x in data):
          raise ValueError
      return data
  except:
      raise argparse.ArgumentTypeError("Input should be a list of float tuples like [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]")        

class TrainTask(object):
    single_view = False
    l2_normalize = True

    def __init__(self, opt):
        self.opt = opt

        self.verbose = is_root_worker()
        total_batch_size = opt.batch_size * opt.acc_grd_step
        if dist.is_initialized():
            total_batch_size *= dist.get_world_size()
        opt.learning_rate = opt.learning_rate * (total_batch_size / 256)
        if opt.resume_epoch > 0 or opt.load_best_model:
            opt.run_name = opt.resume_name
        
        self.cur_epoch = 1
        self.logger = LoggerX(save_root=osp.join(opt.models_dir, opt.run_name),
                              enable_wandb=opt.wandb,
                              config=opt,
                              project=opt.project_name,
                              entity=opt.entity,
                              name=opt.run_name)
        self.feature_extractor = None
        self.feature_extractor_copy = None
        self.is_ptFile=False
        self.set_loader()
        self.set_model()
        self.scaler = NativeScalerWithGradNormCount(amp=opt.amp)
        self.logger.append(self.scaler, name='scaler')
        self.best_acc_setup=None
        self.best_acc_epoch=None
        self.best_acc=None
        self.last_acc=None
        self.last_setup=None
        self.best_acc_test=None
        self.best_acc_setup_test=None
        self.best_results_test=None
        self.last_setup_train_results=None
        self.best_resuls=None

    @staticmethod
    def build_default_options():
        parser = argparse.ArgumentParser('Default arguments for training of different methods')

        parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
        parser.add_argument('--test_freq', type=int, default=50, help='test frequency')
        parser.add_argument('--wandb', help='wandb', action='store_true')
        parser.add_argument('--project_name', help='wandb project_name', type=str, default='Clustering')
        parser.add_argument('--entity', help='wandb project_name', type=str, default='None')
        parser.add_argument('--run_name', type=str, help='each run name')

        parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
        parser.add_argument('--resume_epoch', type=int, default=0, help='number of training epochs')
        parser.add_argument('--resume_name', type=str)
        parser.add_argument('--local_rank', default=0, type=int)
        parser.add_argument('--seed', default=0, type=int)
        parser.add_argument('--eval_metric', nargs='+', type=str,
                            default=['nmi', 'acc', 'ari'], help='evaluation metric NMI ACC ARI')

        # optimization
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--amp', help='amp', action='store_true')
        parser.add_argument('--encoder_name', type=str, help='the type of encoder', default='bigresnet18')
        parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
        parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs')

        # learning rate
        parser.add_argument('--learning_rate', type=float, default=0.05, help='base learning rate')
        parser.add_argument('--learning_eta_min', type=float, default=0.01, help='base learning rate')
        parser.add_argument('--lr_decay_gamma', type=float, default=0.1)
        parser.add_argument('--lr_decay_milestone', nargs='+', type=int, default=[60, 80])
        parser.add_argument('--step_lr', help='step_lr', action='store_true')

        parser.add_argument('--acc_grd_step', help='acc_grd_step', type=int, default=1)
        parser.add_argument('--warmup_epochs', type=int, default=0, help='warmup epochs')
        parser.add_argument('--dist', help='use  for clustering', action='store_true')
        parser.add_argument('--num_devices', type=int, default=-1, help='warmup epochs')

        # dataset
        parser.add_argument('--whole_dataset', action='store_true', help='use whole dataset')
        parser.add_argument('--pin_memory', action='store_true', help='pin_memory for dataloader')
        parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
        parser.add_argument('--data_folder', type=str, default='/home/zzhuang/DATASET/clustering',
                            help='path to custom dataset')
        parser.add_argument('--label_file', type=str, default=None, help='path to label file (numpy format)')
        parser.add_argument('--img_size', type=int, default=32, help='parameter for RandomResizedCrop')
        parser.add_argument('--num_cluster', type=int, help='num_cluster')
        parser.add_argument('--test_resized_crop', action='store_true', help='imagenet test transform')
        parser.add_argument('--resized_crop_scale', type=float, help='randomresizedcrop scale', default=0.08)

        parser.add_argument('--model_name', type=str, help='the type of method', default='supcon')
        parser.add_argument('--use_gaussian_blur', help='use_gaussian_blur', action='store_true')
        parser.add_argument('--save_checkpoints', help='save_checkpoints', action='store_true')
        parser.add_argument('--use_copy', help='use_copy', action='store_true')

        # SSL setting
        parser.add_argument('--feat_dim', type=int, default=2048, help='projection feat_dim')
        parser.add_argument('--data_resample', help='data_resample', action='store_true')
        parser.add_argument('--reassign', type=int, help='reassign kmeans', default=1)
        
        # training automatic
        parser.add_argument('--epochs_cluster_analysis', nargs='+', type=int, default=[300, 300], help='epochs where to run automatic cluster analysis')
      
        parser.add_argument('--resize', nargs='+', type=int, default=None, help='resizing images')
        parser.add_argument('--to_rgb',action='store_true', help='extend to grayscale image to RGB')
        parser.add_argument('--channels',type=int, default=3,help='number of image channels')
        parser.add_argument('--custom_normalize_params', type=parse_tuples, default=None,
                    help='List of tuples where to normalize data. Example: [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]')
        
        parser.add_argument('--permute',action='store_true',help='permute first and second dim of data')
        parser.add_argument('--clusternet',action='store_true',help='activate clusternet')
        parser.add_argument('--clusternet_training_data',type=float,help='number of training data for clusternet')
        parser.add_argument('--cluster_range',nargs='+', type=int, default=[2, 30], help='range of kmeans computation by the clusternet')
        parser.add_argument('--n_candidate_autoencoder_clustering',type=int,default=4,help='number of candidate for jensen  silhouette uniformity metrics based on max avh_silhouette score')       
        parser.add_argument('--n_candidate_k_mean',type=int,default=5,help='number of candidate for mean of best k  based on jensen silhouette uniformity')
        parser.add_argument('--n_candidate_k_max',type=int,default=5,help='number of candidate for selecting max of best k based on jensen silhouette uniformilty')
        parser.add_argument('--autoencoder_latent_dim',type=int,default=3,help='autoencoder latent dimension')
        parser.add_argument('--number_autoencoder',type=int,default=10,help='number of autoencoder')
        parser.add_argument('--save_best_model',action='store_true',help='save best model')
        parser.add_argument('--load_best_model',action='store_true',help='Results on selected model')
        parser.add_argument('--models_dir', type=str, help='path to save/load models', default='/ckpt')
        return parser

    @staticmethod
    def build_options():
        pass

    @staticmethod
    def create_dataset(data_root, dataset_name, train, transform=None, memory=False, label_file=None,unlabeled=False
                       ):
        has_subfolder = False
        if dataset_name in ['cifar10', 'cifar20', 'cifar100','stl10']:
            dataset_type = {'cifar10': torchvision.datasets.CIFAR10,
                            'cifar20': torchvision.datasets.CIFAR100,
                            'cifar100': torchvision.datasets.CIFAR100,
                            'stl10':torchvision.datasets.STL10}[dataset_name]
            has_subfolder = True
            if dataset_name == 'stl10' :
                if unlabeled :
                  train='unlabeled'
                  has_subfolder = False
                elif not train :
                  train='test'
                else :
                  train='train'
                dataset = dataset_type(data_root, train,transform=transform,download=True)
            else:  
                dataset = dataset_type(data_root, train, transform)
            if dataset_name == 'cifar20':
                targets = np.array(dataset.targets)
                super_classes = [
                    [72, 4, 95, 30, 55],
                    [73, 32, 67, 91, 1],
                    [92, 70, 82, 54, 62],
                    [16, 61, 9, 10, 28],
                    [51, 0, 53, 57, 83],
                    [40, 39, 22, 87, 86],
                    [20, 25, 94, 84, 5],
                    [14, 24, 6, 7, 18],
                    [43, 97, 42, 3, 88],
                    [37, 17, 76, 12, 68],
                    [49, 33, 71, 23, 60],
                    [15, 21, 19, 31, 38],
                    [75, 63, 66, 64, 34],
                    [77, 26, 45, 99, 79],
                    [11, 2, 35, 46, 98],
                    [29, 93, 27, 78, 44],
                    [65, 50, 74, 36, 80],
                    [56, 52, 47, 59, 96],
                    [8, 58, 90, 13, 48],
                    [81, 69, 41, 89, 85],
                ]
                import copy
                copy_targets = copy.deepcopy(targets)
                for i in range(len(super_classes)):
                    for j in super_classes[i]:
                        targets[copy_targets == j] = i
                dataset.targets = targets.tolist()
        else:
            data_path = osp.join(data_root, dataset_name)
            dataset_type = torchvision.datasets.ImageFolder
            if 'train' in os.listdir(data_path):
                has_subfolder = True
                dataset = dataset_type(
                    osp.join(data_root, dataset_name, 'train' if train else 'val'), transform=transform)
            else:
                dataset = dataset_type(osp.join(data_root, dataset_name), transform=transform)
        if label_file is not None:
            new_labels = np.load(label_file).flatten()
            assert len(dataset.targets) == len(new_labels)
            noise_ratio = (1 - np.mean(np.array(dataset.targets) == new_labels))
            dataset.targets = new_labels.tolist()
            print(f'load label file from {label_file}, possible noise ratio {noise_ratio}')
        return dataset, has_subfolder
    
    
    def load_tensors(self, data_path, train):
       if train:
          data = torch.load(osp.join(data_path, 'train_data.pt'))
          labels = torch.load(osp.join(data_path, 'train_labels.pt'))  # Added missing parenthesis here
  
       elif not train and ('val_data.pt' in os.listdir(data_path)):
          data = torch.load(osp.join(data_path, 'val_data.pt'))
          labels = torch.load(osp.join(data_path, 'val_labels.pt'))  # Added missing parenthesis here
  
       else:
          raise Exception("No data is found")
       
       if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
       if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

       return data, labels

        
    def create_dataset_from_ptfile(self, data_path, dataset_name, whole_dataset, train, permute, transform=None):
        valset = True
        
        train_data = None
        train_labels = None
        val_data = None
        val_labels = None
        
        if train:
            train_data, train_labels = self.load_tensors(data_path, train)
        elif not train and ('val_data.pt' in os.listdir(data_path)):
            val_data, val_labels = self.load_tensors(data_path, train)
        else:
            train_data, train_labels = self.load_tensors(data_path, train)
            valset = False
        
        if whole_dataset and valset:
            if val_data is None:
                ano_data, ano_labels = self.load_tensors(data_path, not train)
                concat_data = torch.cat((train_data, ano_data), axis=0)
                concat_labels = torch.cat((train_labels, ano_labels), axis=0)
            else:
                ano_data, ano_labels = self.load_tensors(data_path, not train)
                concat_data = torch.cat((val_data, ano_data), axis=0)
                concat_labels = torch.cat((val_labels, ano_labels), axis=0)
    
            concat_data = concat_data.unsqueeze(1).float()
    
            # Create the custom dataset using the transformation
            return PTDataset(concat_data, concat_labels, transform=transform), concat_labels
    
        elif val_data is None:
            train_data = train_data.unsqueeze(1).float()
            
            # Create the custom dataset using the transformation
            return PTDataset(train_data, train_labels, transform=transform), train_labels
        else :
            val_data = val_data.unsqueeze(1).float()
            
            # Create the custom dataset using the transformation
            return PTDataset(val_data, val_labels, transform=transform), val_labels

         
       
    def build_dataloader(self,
                         dataset_name,
                         transform,
                         batch_size,
                         drop_last=False,
                         shuffle=False,
                         sampler=False,
                         train=False,
                         memory=False,
                         data_resample=False,
                         label_file=None,unlabeled=False):

        opt = self.opt
        data_root = opt.data_folder
        
        
        data_path = osp.join(data_root, dataset_name)
        if self.is_ptFile:
            dataset,labels= self.create_dataset_from_ptfile(data_path,dataset_name,opt.whole_dataset,train,opt.permute,transform)
            has_subfolder=False
        else:
          dataset, has_subfolder = self.create_dataset(data_root, dataset_name,
                                                     train, transform=transform,
                                                     memory=memory,
                                                     label_file=label_file,unlabeled=unlabeled)
        
        if opt.dataset=='stl10':
          labels=dataset.labels
        elif not self.is_ptFile : 
          labels = dataset.targets

        labels = np.array(labels)

        if opt.whole_dataset and has_subfolder and not self.is_ptFile:
            ano_dataset = self.create_dataset(data_root, dataset_name, not train, transform=transform,
                                              memory=memory)[0]
            if opt.dataset=='stl10':
              labels = np.concatenate([labels, ano_dataset.labels], axis=0)
            else:
              labels = np.concatenate([labels, ano_dataset.targets], axis=0)
            dataset = torch.utils.data.ConcatDataset([dataset, ano_dataset])

        with_indices = train and (not memory)
        if with_indices and not opt.dataset =='stl10':
            dataset = dataset_with_indices(dataset)

        if sampler:
            if dist.is_initialized():
                from utils.sampler import RandomSampler
                if shuffle and data_resample:
                    num_iter = len(dataset) // (batch_size * dist.get_world_size())
                    sampler = RandomSampler(dataset=dataset, batch_size=batch_size, num_iter=num_iter, restore_iter=0,
                                            weights=None, replacement=True, seed=0, shuffle=True)
                else:
                    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
            else:
                # memory loader
                sampler = None
        else:
            sampler = None

        # if opt.num_workers > 0:
        #     prefetch_factor = max(2, batch_size // opt.num_workers)
        #     persistent_workers = True
        # else:
        #     prefetch_factor = 2
        #     persistent_workers = False
        prefetch_factor = 2
        persistent_workers = True
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(False if sampler is not None else shuffle),
            num_workers=opt.num_workers,
            pin_memory=opt.pin_memory,
            sampler=sampler,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor
        )
        return dataloader, labels, sampler
    
    

    def train_transform(self, normalize):
        '''
        Apply transformations suitable for the SimCLR model.
        :param normalize: A normalization transform.
        :return: A composed torchvision transform.
        '''
        opt = self.opt
        train_transform = []
        data_root = opt.data_folder
        
        
        data_path = osp.join(data_root, opt.dataset)
        if osp.exists(data_path):
          if 'train_data.pt' in os.listdir(data_path):
            self.is_ptFile=True
        
        if opt.resize != 'None':
            train_transform.append(transforms.Resize((opt.resize, opt.resize)))  
        # Adding specific transformations based on the number of channels
        if opt.channels <3 :
            shear_transform = transforms.RandomAffine(degrees=0, shear=(-10, 10))
            random_shear=transforms.RandomApply([shear_transform],p=0.7)
            train_transform.extend([
                transforms.RandomResizedCrop(size=opt.img_size, scale=(opt.resized_crop_scale, 1.)),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            train_transform.extend([
                transforms.RandomResizedCrop(size=opt.img_size, scale=(opt.resized_crop_scale, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
            ])
    
        # Optionally applying Gaussian Blur
        if opt.use_gaussian_blur:
            train_transform.append(
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], 0.5)
            )
    
        # Converting to tensor and applying normalization
        if self.is_ptFile:
          train_transform.extend([normalize])
        else:
          train_transform.extend([transforms.ToTensor(), normalize])
    
        # Composing all the transformations
        train_transform = transforms.Compose(train_transform)
    
        # If using two-crop transform (typically for contrastive learning models like SimCLR)
        if not self.single_view:
            #print('INTO SINGLE VIEW CONDITION')
            train_transform = TwoCropTransform(train_transform)
    
        return train_transform


    def test_transform(self, normalize):
        opt = self.opt

        def resize(image):
            size = (opt.img_size, opt.img_size)
            if image.size == size:
                return image
            return image.resize(size)

        test_transform = []
        if opt.test_resized_crop:
            test_transform += [transforms.Resize(256), transforms.CenterCrop(224)]
        if  opt.resize != 'None':
            test_transform +=[transforms.Resize((opt.resize, opt.resize))]
        if self.is_ptFile :
            test_transform +=[transforms.Resize( (opt.img_size, opt.img_size)),normalize]
        else :
            test_transform += [
                resize,
                transforms.ToTensor(),
                normalize
            ]
        
        test_transform = transforms.Compose(test_transform)
        return test_transform
    
    
        
    #@staticmethod
    def normalize(self,dataset_name):
        normalize_params = {
              'cifar10': [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)],
              'cifar20': [(0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)],
              'cifar100': [(0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)],
              'imagenet': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
              'stl10': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
          }
        #print('self.opt.custom_normalize_params',self.opt.custom_normalize_params)
        if self.opt.custom_normalize_params is not None:
          normalize_params[dataset_name]=self.opt.custom_normalize_params
        
        #if normalize_params == None and dataset_name not in normalize_params.keys():
        if dataset_name not in normalize_params.keys():
            mean, std = normalize_params['imagenet']
            print(f'Dataset {dataset_name} does not exist in normalize_params,'
                  f' use default normalizations: mean {str(mean)}, std {str(std)}.')
        else:
            print('DATASET NAME:',dataset_name)
            #print('normalize_params',normalize_params)
            mean, std = normalize_params[dataset_name]

        normalize = transforms.Normalize(mean=mean, std=std, inplace=True)
        return normalize

    def set_loader(self):
        opt = self.opt

        normalize = self.normalize(opt.dataset)

        train_transform = self.train_transform(normalize)
        self.logger.msg_str(f'set train transform... \n {str(train_transform)}')

        train_loader, labels, sampler = self.build_dataloader(
            dataset_name=opt.dataset,
            transform=train_transform,
            batch_size=opt.batch_size,
            shuffle=True,
            drop_last=True,
            sampler=True,
            train=True,
            data_resample=opt.data_resample,
            label_file=opt.label_file)
        if sampler is None:
            sampler = train_loader.sampler
        self.logger.msg_str(f'set train dataloader with {len(train_loader)} iterations...')

        test_transform = self.test_transform(normalize)
        self.logger.msg_str(f'set test transform... \n {str(test_transform)}')

        if 'imagenet' in opt.dataset:
            if not opt.test_resized_crop:
                warnings.warn('ImageNet should center crop during testing...')
        test_loader = self.build_dataloader(opt.dataset,
                                            test_transform,
                                            train=False,
                                            sampler=True,
                                            batch_size=opt.batch_size)[0]
        self.logger.msg_str(f'set test dataloader with {len(test_loader)} iterations...')
        memory_loader = self.build_dataloader(opt.dataset,
                                              test_transform,
                                              train=True,
                                              batch_size=opt.batch_size,
                                              sampler=True,
                                              memory=True,
                                              label_file=opt.label_file)[0]
        self.logger.msg_str(f'set memory dataloader with {len(memory_loader)} iterations...')

        self.test_loader = test_loader
        self.memory_loader = memory_loader
        self.train_loader = train_loader
        self.sampler = sampler
        self.iter_per_epoch = len(train_loader)
        self.num_samples = len(labels)
        self.gt_labels = labels
        if not (opt.resume_epoch>0):
          self.num_cluster=opt.num_cluster
        opt.num_cluster = self.num_cluster
        self.psedo_labels = torch.zeros((self.num_samples,)).long().cuda()

        self.logger.msg_str('load {} images...'.format(self.num_samples))

    
    def predict(self):
        rank = dist.get_rank()
        device = torch.device(f'cuda:{rank}')
        self.cluster_centers=self.logger.load_checkpoints(0,load_best=True).to(device)
        self.num_cluster=self.cluster_centers.size()[0]
        opt = self.opt

        torch.cuda.empty_cache()
        self.logger.msg_str('Generating the psedo-labels')

        if opt.use_copy:
            msg = self.feature_extractor_copy.load_state_dict(self.feature_extractor.state_dict())
            self.logger.msg_str(msg)
            params = self.feature_extractor_copy.state_dict()
            for k in params:
                dist.broadcast(params[k], src=0)
            feature_extractor = self.feature_extractor_copy
        else:
            feature_extractor = self.feature_extractor
        
        
        mem_features, mem_labels = extract_features(feature_extractor, self.memory_loader)
        if self.l2_normalize:
            mem_features.div_(torch.linalg.norm(mem_features, dim=1, ord=2, keepdim=True))
        
        psedo_labels=self.clustering_predict(mem_features,self.cluster_centers)
        dist.barrier()
        global_std = torch.std(mem_features, dim=0).mean()

        result_best_fitted_acc=self.cluster_acc(mem_labels,psedo_labels)

        results = torch_clustering.evaluate_clustering(mem_labels.cpu().numpy(),
                                                       psedo_labels.cpu().numpy(),
                                                       eval_metric=opt.eval_metric,
                                                       phase='ema_train')
        
        results = {'NMI': results['ema_train_nmi'],  'ARI': results['ema_train_ari'],  'ACC':result_best_fitted_acc[0], 'K': self.num_cluster}
           
        if torch.distributed.get_rank() == 0:
           print('Evaluation train set :',results)
           
        if not opt.whole_dataset:
           val_features, val_labels = extract_features(feature_extractor, self.test_loader)
           val_psedo_labels=self.clustering_predict(val_features,self.cluster_centers) 
           result_best_fitted_acc_test=self.cluster_acc(val_labels,val_psedo_labels)
           #print('ACC best_fitted cluster test:',result_best_fitted_acc_test)
           results_concat_test = torch_clustering.evaluate_clustering(val_labels.cpu().numpy(),
                                                       val_psedo_labels.cpu().numpy(),
                                                       eval_metric=opt.eval_metric,
                                                       phase='ema_train')  
           results_test = {'NMI': results_concat_test['ema_train_nmi'],  'ARI': results_concat_test['ema_train_ari'],  'ACC':result_best_fitted_acc_test[0], 'K': self.num_cluster}
           
        if torch.distributed.get_rank() == 0:
           print('Evaluation test set :',results_test)
        
        
    def fit(self):
        opt = self.opt
        if opt.resume_epoch > 0:
            self.cluster_centers=self.logger.load_checkpoints(opt.resume_epoch)
            self.num_clusters=self.cluster_centers.size()[0]

        n_iter = self.iter_per_epoch * opt.resume_epoch + 1
        self.cur_epoch = int(opt.resume_epoch + 1)
        # training routine
        self.progress_bar = tqdm.tqdm(total=self.iter_per_epoch * opt.epochs, disable=not self.verbose, initial=n_iter)

        self.psedo_labeling(n_iter)
        i=0
        while True:
            self.sampler.set_epoch(self.cur_epoch)
            for inputs in self.train_loader:
                inputs, indices = convert_to_cuda(inputs)
                self.adjust_learning_rate(n_iter)
                self.train(inputs, indices, n_iter)
                self.progress_bar.refresh()
                self.progress_bar.update()
                n_iter += 1

            cur_epoch = self.cur_epoch
            self.logger.msg([cur_epoch, ], n_iter)
            
            if  i<len(opt.epochs_cluster_analysis) and opt.clusternet:
                if self.cur_epoch==opt.epochs_cluster_analysis[i] :
                    self.get_new_k(split=opt.clusternet_training_data,index=i)  
                    i+=1

            apply_kmeans = (self.cur_epoch % opt.reassign) == 0
            if (self.cur_epoch % opt.test_freq == 0) or (self.cur_epoch % opt.save_freq == 0) or apply_kmeans:
                if opt.save_checkpoints:
                    self.logger.checkpoints(int(self.cur_epoch))
            
            if apply_kmeans:
                self.psedo_labeling(n_iter)
            
               
            if (self.cur_epoch % opt.test_freq == 0) or apply_kmeans:
                self.test(n_iter)
                torch.cuda.empty_cache()
            
            
            self.cur_epoch += 1
            
            if self.cur_epoch > opt.epochs :
              if torch.distributed.get_rank() == 0 :
                if opt.whole_dataset:
                  print('Evaluation Best Model WHOLE DATASET :',self.best_results)
                else:
                  print('Evaluation Best Model TEST SET :',self.best_results_test)
              break

    def set_model(opt):
        pass
    
    
        
    def get_new_k(self, split=0.9,index=0):
        opt = self.opt
        if opt.use_copy:
            feature_extractor = self.feature_extractor_copy
        else:
            feature_extractor = self.feature_extractor
    
        mem_features, mem_labels = extract_features(feature_extractor, self.memory_loader)
        if self.l2_normalize:
            mem_features = torch.nn.functional.normalize(mem_features, p=2, dim=1)
    
        gathered_features = concat_all_gather(mem_features)
        new_k_tensor = torch.tensor([0]).cuda()
        if torch.distributed.get_rank() == 0:
            print('*** CLUSTERING SUPERVISOR MODULE ****')
            total_samples = gathered_features.shape[0]
            num_samples = int(total_samples * split) if 0 < split < 1 else int(split)
    
            sampled_indices = torch.randint(0, total_samples, (num_samples,))
            sampled_features = gathered_features[sampled_indices]
    
            cluster_analysis_instance = ClusterAnalysis(
                num_models=opt.number_autoencoder,
                input_dim=sampled_features.size(1),
                latent_dim=opt.autoencoder_latent_dim,
                max_lr=0.01,
                kappa=130,
                features=sampled_features,
                num_epochs=opt.autoencoder_training_epoch, 
                index=index,
                cluster_range=opt.cluster_range,
                rank=dist.get_rank(),
                n_candidate_autoencoder_clustering=opt.n_candidate_autoencoder_clustering,
                n_candidate_k_mean=opt.n_candidate_k_mean,
                n_candidate_k_max=opt.n_candidate_k_max
            ) 
            new_k,list_results,models_tracking_clusterization = cluster_analysis_instance.get_k()
            new_k_tensor+= new_k
            
        
        
        dist.barrier()
           
        # Broadcast the new K from GPU 0 to all others
        torch.distributed.broadcast(new_k_tensor, src=0)
    
        # Update self.num_cluster with the new K for all processes
        self.num_cluster = new_k_tensor.item()
        
                
    @torch.no_grad()
    def test(self, n_iter):
        opt = self.opt
        if opt.whole_dataset:
            return

        if opt.use_copy:
            feature_extractor = self.feature_extractor_copy
        else:
            feature_extractor = self.feature_extractor

        test_features, test_labels = extract_features(feature_extractor, self.test_loader)
        if hasattr(self, 'mem_data') and self.mem_data['epoch'] == self.cur_epoch:
            mem_features, mem_labels = self.mem_data['features'], self.mem_data['labels']
        else:
            mem_features, mem_labels = extract_features(feature_extractor, self.memory_loader)
            if self.l2_normalize:
                mem_features.div_(torch.linalg.norm(mem_features, dim=1, ord=2, keepdim=True))
        if self.l2_normalize:
            test_features.div_(torch.linalg.norm(test_features, dim=1, ord=2, keepdim=True))

        from utils.knn_monitor import knn_monitor
        knn_acc = knn_monitor(
            mem_features,
            mem_labels,
            test_features,
            test_labels,
            knn_k=20,
            knn_t=0.07)
        
        self.logger.msg([knn_acc, ], n_iter)
    


    def best_cluster_fit(self,y_true, y_pred):
        # Convert PyTorch tensors to numpy arrays for compatibility with linear_assignment
        y_true_np = y_true.cpu().numpy().astype(np.int64)
        y_pred_np = y_pred.cpu().numpy().astype(np.int64)
        
        D = max(y_pred_np.max(), y_true_np.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        
        for i in range(y_pred_np.size):
            w[y_pred_np[i], y_true_np[i]] += 1
    
        row_ind, col_ind = linear_assignment(w.max() - w)
        best_fit = []
        for i in range(y_pred_np.size):
            for j in range(len(row_ind)):
                if row_ind[j] == y_pred_np[i]:
                    best_fit.append(col_ind[j])
        return best_fit, row_ind, col_ind, w
    
    def cluster_acc(self,y_true, y_pred):
        best_fit, row_ind, col_ind, w = self.best_cluster_fit(y_true, y_pred)
        return w[row_ind, col_ind].sum() * 1.0 / y_pred.size()


    def train(self, inputs, indices, n_iter):
        pass

    def cosine_annealing_LR(self, n_iter):
        opt = self.opt

        epoch = n_iter / self.iter_per_epoch
        max_lr = opt.learning_rate
        min_lr = max_lr * opt.learning_eta_min
        # warmup
        if epoch < opt.warmup_epochs:
            # lr = (max_lr - min_lr) * epoch / opt.warmup_epochs + min_lr
            lr = opt.learning_rate * epoch / opt.warmup_epochs
        else:
            lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos((epoch - opt.warmup_epochs) * np.pi / opt.epochs))
        return lr

    def step_LR(self, n_iter):
        opt = self.opt
        lr = opt.learning_rate
        epoch = n_iter / self.iter_per_epoch
        if epoch < opt.warmup_epochs:
            # lr = (max_lr - min_lr) * epoch / opt.warmup_epochs + min_lr
            lr = opt.learning_rate * epoch / opt.warmup_epochs
        else:
            for milestone in opt.lr_decay_milestone:
                lr *= opt.lr_decay_gamma if epoch >= milestone else 1.
        return lr

    def adjust_learning_rate(self, n_iter):
        opt = self.opt
        if opt.step_lr:
            lr = self.step_LR(n_iter)
        else:
            lr = self.cosine_annealing_LR(n_iter)

        for i in range(len(self.optimizer.param_groups)):
            self.optimizer.param_groups[i]['lr'] = lr
        self.logger.msg([lr, ], n_iter)

    def clustering(self, features, n_clusters):
        opt = self.opt

        kwargs = {
            'metric': 'cosine' if self.l2_normalize else 'euclidean',
            'distributed': True,
            'random_state': 0,
            'n_clusters': n_clusters,
            'verbose': True
        }
        clustering_model = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
        psedo_labels = clustering_model.fit_predict(features)
        cluster_centers = clustering_model.cluster_centers_
        return psedo_labels, cluster_centers
    
    def clustering_predict(self,features,cluster_centers):
        kwargs = {
            'metric': 'cosine' if self.l2_normalize else 'euclidean',
            'distributed': True,
            'random_state': 0,
            'n_clusters': self.num_cluster,
            'verbose': True
        }
        clustering_model = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
        psedo_labels,_ = clustering_model.predict(features,cluster_centers)
        return(psedo_labels)
           
           
    
    @torch.no_grad()
    def psedo_labeling(self, n_iter):
        opt = self.opt

        torch.cuda.empty_cache()
        self.logger.msg_str('Generating the psedo-labels')

        if opt.use_copy:
            msg = self.feature_extractor_copy.load_state_dict(self.feature_extractor.state_dict())
            self.logger.msg_str(msg)
            params = self.feature_extractor_copy.state_dict()
            for k in params:
                dist.broadcast(params[k], src=0)
            feature_extractor = self.feature_extractor_copy
        else:
            feature_extractor = self.feature_extractor

        mem_features, mem_labels = extract_features(feature_extractor, self.memory_loader)
        if self.l2_normalize:
            # mem_features = F.normalize(mem_features, dim=1)
            mem_features.div_(torch.linalg.norm(mem_features, dim=1, ord=2, keepdim=True))
        
        psedo_labels, cluster_centers = self.clustering(mem_features, self.num_cluster)
        dist.barrier()
        global_std = torch.std(mem_features, dim=0).mean()

        self.logger.msg_str(torch.unique(psedo_labels.cpu(), return_counts=True))
        self.logger.msg_str(torch.unique(mem_labels.long().cpu(), return_counts=True))
        result_best_fitted_acc=self.cluster_acc(mem_labels,psedo_labels)

        results = torch_clustering.evaluate_clustering(mem_labels.cpu().numpy(),
                                                       psedo_labels.cpu().numpy(),
                                                       eval_metric=opt.eval_metric,
                                                       phase='ema_train')
        
        results_train = {'NMI': results['ema_train_nmi'],  'ARI': results['ema_train_ari'],  'ACC':result_best_fitted_acc[0], 'K': self.num_cluster}
        
        if torch.distributed.get_rank() == 0:
          print('TRAIN :',results_train)
        

        dist.broadcast(psedo_labels, src=0)
        dist.broadcast(cluster_centers, src=0)

        self.psedo_labels.copy_(psedo_labels)
        self.cluster_centers = cluster_centers
        self.mem_data = {
            'features': mem_features,
            'labels': mem_labels,
            'epoch': self.cur_epoch
        }
        
        
        #self.last_acc=result_best_fitted_acc
        #self.last_setup=results
        #self.last_setup_train_results=results_train
           
        if not opt.whole_dataset:
           val_features, val_labels = extract_features(feature_extractor, self.test_loader)
           val_psedo_labels=self.clustering_predict(val_features,self.cluster_centers) 
           result_best_fitted_acc_test=self.cluster_acc(val_labels,val_psedo_labels)
           results_concat_test = torch_clustering.evaluate_clustering(val_labels.cpu().numpy(),
                                                       val_psedo_labels.cpu().numpy(),
                                                       eval_metric=opt.eval_metric,
                                                       phase='ema_train')                                            
           self.last_acc=result_best_fitted_acc_test
           self.last_setup=results_concat_test
           results_test = {'NMI': results_concat_test['ema_train_nmi'],  'ARI': results_concat_test['ema_train_ari'],  'ACC':result_best_fitted_acc_test[0], 'K': self.num_cluster}
           
           if torch.distributed.get_rank() == 0:
             print('TEST :',results_test)
           if self.best_acc_test is None:
             self.best_acc_epoch=self.cur_epoch
             self.best_acc_test=result_best_fitted_acc_test
             self.best_acc_setup_test=results_concat_test
             self.best_results_test =results_test
             
           elif self.best_acc_test<result_best_fitted_acc_test:
             self.best_acc_epoch=self.cur_epoch
             self.best_acc_test=result_best_fitted_acc_test
             self.best_acc_setup_test=results_concat_test
             self.best_results_test =results_test
             if opt.save_best_model:
               self.logger.checkpoints(int(self.cur_epoch),self.cluster_centers,save_best=True)
        else :
           if self.best_acc_setup is None:
             self.best_acc=result_best_fitted_acc
             self.best_acc_setup=results
             self.best_acc_epoch=self.cur_epoch
           elif self.best_acc<result_best_fitted_acc:
             self.best_acc=result_best_fitted_acc
             self.best_acc_setup=results
             self.best_results=results_train
             self.best_acc_epoch=self.cur_epoch
             if opt.save_best_model:
               self.logger.checkpoints(int(self.cur_epoch),self.cluster_centers,save_best=True)
        if opt.data_resample:
            counts = torch.unique(psedo_labels.cpu(), return_counts=True)[1]
            weights = torch.zeros(psedo_labels.size()).float()
            for l in range(counts.size(0)):
                weights[psedo_labels == l] = psedo_labels.size(0) / counts[l]
            self.sampler.set_weights(weights)
            self.logger.msg_str(f'set the weights of train dataloader as {weights.cpu().numpy()}')
        
        torch.cuda.empty_cache()

    def collect_params(self, *models, exclude_bias_and_bn=True):
        param_list = []
        for model in models:
            for name, param in model.named_parameters():
                param_dict = {
                    'name': name,
                    'params': param,
                }
                if exclude_bias_and_bn and any(s in name for s in ['bn', 'bias']):
                    param_dict.update({'weight_decay': 0., 'lars_exclude': True})
                param_list.append(param_dict)
        return param_list

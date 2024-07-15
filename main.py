# -*- coding: UTF-8 -*-
'''
@Project : ProPos 
@File    : main.py
@Author  : Zhizhong Huang from Fudan University
@Homepage: https://hzzone.github.io/
@Email   : zzhuang19@fudan.edu.cn
@Date    : 2022/10/19 9:24 PM 
'''
import time # A suppr
import torch
import torch.distributed as dist
import numpy as np
import random
from PIL import ImageFile
import sys
import yaml
import os.path as osp
from datetime import datetime

ImageFile.LOAD_TRUNCATED_IMAGES = True

from models.basic_template import TrainTask
from models import model_dict
import os
from datetime import timedelta


#os.environ['NCCL_ALLREDUCE_TIMEOUT_MS'] = '3600000'



if __name__ == '__main__':
    
    #config_path = sys.argv[1]
    #print("CONFIG PATH",config_path)
    main_dir = os.path.dirname(os.path.abspath(__file__))
    #config_path= os.path.join(main_dir, sys.argv[1])
    config_path=main_dir+sys.argv[1]
    with open(config_path) as f:
        if hasattr(yaml, 'FullLoader'):
            configs = yaml.load(f.read(), Loader=yaml.FullLoader)
        else:
            configs = yaml.load(f.read())
    
    
    
    # Start the timer
    start_time = time.time()
    #with open(PATH, 'r') as file:
    #    configs = yaml.safe_load(file)
    #import sys
    #print('SYS :',sys.path)

    MODEL = model_dict[configs['model_name']]
    default_parser = TrainTask.build_default_options()
    default_opt, unknown_opt = default_parser.parse_known_args('')
    private_parser = MODEL.build_options()
    opt = private_parser.parse_args(unknown_opt, namespace=default_opt)

    if opt.run_name is None:
        opt.run_name = osp.basename(config_path)[:-4]
        #opt.run_name = osp.basename(PATH)[:-4]
    opt.run_name = '{}-{}'.format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), opt.run_name)
    for k in configs:
        setattr(opt, k, configs[k])
    if opt.dist:
        custom_timeout = timedelta(minutes=60)
        dist.init_process_group(backend='nccl', init_method='env://',timeout=custom_timeout)
        torch.cuda.set_device(dist.get_rank())
    if opt.num_devices > 0:
        print('NB DEVICE USED :',torch.cuda.device_count())
        assert opt.num_devices == torch.cuda.device_count()  # total batch size


    seed = opt.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    model = MODEL(opt)
    if not opt.load_best_model:
      model.fit()
    else :
      model.predict()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")
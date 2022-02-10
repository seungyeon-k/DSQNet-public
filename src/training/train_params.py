import numpy as np

import os
import random
import torch
from tensorboardX import SummaryWriter

import argparse
from omegaconf import OmegaConf
from itertools import cycle
from datetime import datetime
# from loader import CustomDataset, DataSplit
from loader import get_dataloader
from models import get_model
from trainers import get_trainer, get_logger
from utils import save_yaml
from optimizers import get_optimizer

def run(cfg, writer):
    # Setup seeds
    seed = cfg.get('seed', 1)
    print(f'running with random seed : {seed}')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Setup device
    device = cfg.device

    # Setup Dataloader
    datainfo = cfg.data
    dataloaders = {}
    for key, data_cfg in datainfo.items():
        dataloaders[key] = get_dataloader(data_cfg)

    # Setup Model
    model = get_model(cfg).to(device)
    trainer = get_trainer(cfg)
    logger = get_logger(cfg, writer)

    # Setup optimizer, lr_scheduler and loss function
    if hasattr(model, 'own_optimizer') and model.own_optimizer:
        optimizer = model.get_optimizer(cfg['training']['optimizer'])
    else:
        optimizer = get_optimizer(cfg["training"]["optimizer"], model.parameters())

    model, train_result = trainer.train(model, optimizer, dataloaders, logger=logger,
                                   logdir=writer.file_writer.get_logdir())


    # for iteration, batch in enumerate(dataloaders[key]):
    #     x_train, y_train = batch
    #     if iteration == 25:
    #         writer.add_mesh('pc', x_train.permute([0,2,1]))
    #     print('{} : {} and {}'.format(iteration, str(x_train.size()), str(y_train.size())))

    # d_dataloaders = {}
    # for key, dataloader_cfg in cfg['data'].items():
    #     d_dataloaders[key] = get_dataloader(dataloader_cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--device', default=0)
    parser.add_argument('--logdir', default='train_results/')
    parser.add_argument('--run', default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.device == 'cpu':
        cfg['device'] = f'cpu'
    else:
        cfg['device'] = f'cuda:{args.device}'
    
    if args.run is None:
        run_id = datetime.now().strftime('%Y%m%d-%H%M')
    else:
        run_id = args.run 

    config_basename = os.path.basename(args.config).split('.')[0]
    # logdir = os.path.join('training', args.logdir, config_basename, str(run_id))
    logdir = os.path.join(args.logdir, config_basename, str(run_id))
    writer = SummaryWriter(logdir=logdir)
    print("Result directory: {}".format(logdir))

    # copy config file
    copied_yml = os.path.join(logdir, os.path.basename(args.config))
    save_yaml(copied_yml, OmegaConf.to_yaml(cfg))
    print(f'config saved as {copied_yml}')

    run(cfg, writer)

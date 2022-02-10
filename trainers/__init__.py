from trainers.logger import BaseLogger
from trainers.base import BaseTrainer
from trainers.base_sq import BaseSQTrainer

def get_trainer(cfg):
    trainer_type = cfg.get('trainer', None)
    arch = cfg['model']['arch']
    device = cfg['device']
    if trainer_type == 'base':
        trainer = BaseTrainer(cfg['training'], device=device)
    if trainer_type == 'base_sq':
        trainer = BaseSQTrainer(cfg['training'], device=device)
    return trainer

def get_logger(cfg, writer):
    logger_cfg = cfg['logger']
    logger = BaseLogger(writer, **logger_cfg)
    return logger
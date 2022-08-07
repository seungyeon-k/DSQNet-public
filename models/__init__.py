import torch
import os 
from omegaconf import OmegaConf

from .dgcnn import DGCNN
from .sqnet import SuperquadricNetwork
from .dsqnet import DeformableSuperquadricNetwork
from .segmenation_network import SegmentationNetwork

def get_model(cfg, *args, **kwargs):
    cfg_model = cfg["model"]
    name = cfg_model["arch"]

    cfg_backbone = cfg_model.pop("backbone")
    backbone = get_backbone_instance(cfg_backbone["arch"])(**cfg_backbone)

    model = get_model_instance(name)
    return model(backbone, **cfg_model)

def get_model_instance(name):
    try:
        return {
            "sqnet": SuperquadricNetwork,
            "dsqnet": DeformableSuperquadricNetwork,
            "segnet": SegmentationNetwork,
        }[name]
    except:
        raise ("Model {} not available".format(name))

def get_backbone_instance(name):
    try:
        return {
            "dgcnn": DGCNN,
        }[name]
    except:
        raise ("backbone {} not available".format(name))

def load_pretrained(identifier, config_file, ckpt_file, root='pretrained', **kwargs):
    """
    load pre-trained model.
    identifier: '<model name>/<run name>'. e.g. 'ae_mnist/z16'
    config_file: name of a config file. e.g. 'ae.yml'
    ckpt_file: name of a model checkpoint file. e.g. 'model_best.pth'
    root: path to pretrained directory
    """
    config_path = os.path.join(root, identifier, config_file)
    ckpt_path = os.path.join(root, identifier, ckpt_file)
    cfg = OmegaConf.load(config_path)
    if "model" in cfg:
        model_name = cfg["model"]['arch']
    else:
        model_name = cfg['arch']
    
    model = get_model(cfg)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'model_state' in ckpt:
        ckpt = ckpt['model_state']
    model.load_state_dict(ckpt)

    return model, cfg
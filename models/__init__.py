from training.models.dgcnn import DGCNN
from training.models.pointnet import PointNet
from training.models.modules import Pointwise_MLP, Estimator_pos_nori_con, \
                                    Estimator_total, MLP, Estimator_sq, \
                                    Estimator_esq, \
                                    Estimator_st, Estimator_est, \
                                    Individual_MLP, \
                                    Estimator_membership
from training.models.param_learning import ParamLearning
from training.models.base_arch import BaseArch
from models.base_arch_segmentation import BaseArchSeg

def get_model(cfg, *args, version=None, **kwargs):
    if 'model' in cfg:
        model_dict = cfg['model']
    elif 'arch' in cfg:
        model_dict = cfg
    else:
        raise ValueError(f'Invalid model configuration dictionary: {cfg}')

    name = model_dict["arch"]
    model = _get_model_instance(name)
    return model(**model_dict)

def _get_model_instance(name):
    try:
        return {
            "dgcnn": get_dgcnn,
            "pointnet": get_pointnet,
            "param_learning": get_param_learning,
            "base_arch": get_base_arch,
            "base_arch_segmentation": get_base_arch_segmentation,
        }[name]
    except:
        raise ("Model {} not available".format(name))

def get_param_learning(**cfg):
    model = ParamLearning(**cfg)
    return model

########################## backbone + ###############################

def get_base_arch(**cfg):
    backbone_model = get_backbone(**cfg['backbone'])
    head_model = get_head(**cfg['head'])
    if 'use_label' in cfg:
        use_label = cfg['use_label']
    else:
        use_label = False
    if 'use_superquadric_label' in cfg:
        use_superquadric_label = cfg['use_superquadric_label']
    else:
        use_superquadric_label = False

    model = BaseArch(backbone_model, head_model, use_label, use_superquadric_label)
    
    return model

def get_base_arch_segmentation(**cfg): # segmentation
    backbone_model = get_backbone(**cfg['backbone'])
    head_model = get_head(**cfg['head'])
    model = BaseArchSeg(backbone_model, head_model)
    return model

def get_backbone(**kwargs):
    if kwargs['arch'] == 'pointnet':
        net = get_pointnet(**kwargs)
    elif kwargs['arch'] == 'dgcnn':
        net = get_dgcnn(**kwargs)
    return net

def get_head(**kwargs):
    net_list = []
    for key in kwargs.keys():
        module_dict = kwargs[key]
        name = module_dict['arch']
        module_nn = _get_head_instance(name)
        module = module_nn(**module_dict)
        net_list.append(module)
    return net_list

def _get_head_instance(name):
    try:
        return {
            'pointwise_mlp': get_pointwise_mlp,
            'mlp': get_mlp,
            'estimator_pos_nori_con': get_estimator_pos_nori_con,
            'estimator_total': get_estimator_total,
            'estimator_sq': get_estimator_sq,
            # 'estimator_sq_local_global': get_estimator_sq_local_global,
            'estimator_esq': get_estimator_esq,
            'estimator_st': get_estimator_st,
            'estimator_est': get_estimator_est,
            'individual_mlp': get_individual_mlp,
            'estimator_membership': get_estimator_membership,
        }[name]
    except:
        raise ("Model {} not available".format(name))

# backbones
def get_pointnet(**cfg):
    model = PointNet(**cfg)
    return model

def get_dgcnn(**cfg):
    model = DGCNN(**cfg)
    return model

# heads
def get_pointwise_mlp(**cfg):
    model = Pointwise_MLP(**cfg)
    return model

def get_mlp(**cfg):
    model = MLP(**cfg)
    return model

def get_estimator_pos_nori_con(**cfg):
    model = Estimator_pos_nori_con(**cfg)
    return model

def get_estimator_total(**cfg):
    model = Estimator_total(**cfg)
    return model

def get_estimator_sq(**cfg):
    model = Estimator_sq(**cfg)
    return model
    
# def get_estimator_sq_local_global(**cfg):
#     model = Estimator_sq_local_global(**cfg)
#     return model

def get_estimator_esq(**cfg):
    model = Estimator_esq(**cfg)
    return model

def get_estimator_st(**cfg):
    model = Estimator_st(**cfg)
    return model

def get_estimator_est(**cfg):
    model = Estimator_est(**cfg)
    return model

def get_individual_mlp(**cfg):
    model = Individual_MLP(**cfg)
    return model

def get_estimator_membership(**cfg):
    model = Estimator_membership(**cfg)
    return model
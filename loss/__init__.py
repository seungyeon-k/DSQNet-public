from loss.dsq_loss import DeformableSuperquadricLoss
from loss.sq_loss import SuperquadricLoss

def get_loss(cfg, *args, device=None, version=None, **kwargs):
    loss_dict = cfg["loss"]
    name = loss_dict['type']
    loss = _get_loss_instance(name)
    return loss(device, **loss_dict)

def _get_loss_instance(name):
    try:
        return {
            "sq_loss": get_superquadric_loss,
            "dsq_loss": get_deformed_superquadric_loss
        }[name]
    except:
        raise ("Loss {} not available".format(name))

def get_superquadric_loss(device, **cfg):
    loss = SuperquadricLoss(device=device, **cfg)
    return loss

def get_deformed_superquadric_loss(device, **cfg):
    loss = DeformableSuperquadricLoss(device=device, **cfg)
    return loss
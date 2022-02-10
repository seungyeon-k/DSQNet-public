from training.loss.mse import MSE
from training.loss.perpoint_position_mse import PerpointPoisitionMSE
from training.loss.perpoint_position_orientation_mse import PerpointPoisitionOrientationMSE
from training.loss.perpoint_total_mse import PerpointTotalMSE, PerpointTotalConeMSE
from training.loss.sq_functional_loss import SuperquadricFunctionalLoss
from training.loss.sq_distance_loss import SuperquadricDistanceLoss
from training.loss.esq_functional_loss import ExtendedSuperquadricFunctionalLoss
from training.loss.esq_distance_loss import ExtendedSuperquadricDistanceLoss
from training.loss.esq_distance_with_normal_loss import ExtendedSuperquadricDistanceWithNormalLoss
from training.loss.esq_distance_with_boundary_loss import ExtendedSuperquadricDistanceWithBoundaryLoss
from training.loss.st_distance_loss import SupertoroidDistanceLoss
from training.loss.st_distance_with_normal_loss import SupertoroidDistanceWithNormalLoss
from training.loss.est_distance_loss import ExtendedSupertoroidDistanceLoss
from training.loss.superquadric_loss import SuperquadricLoss
from training.loss.deformed_superquadric_loss import DeformedSuperquadricLoss
from training.loss.deformed_superquadric_loss2 import DeformedSuperquadricLoss2

def get_loss(cfg, *args, device=None, version=None, **kwargs):
    loss_dict = cfg["loss"]
    name = loss_dict['type']
    loss = _get_loss_instance(name)
    return loss(device, **loss_dict)

def _get_loss_instance(name):
    try:
        return {
            "mse": get_mse,
            "perpoint_position_mse": get_perpoint_position_mse,
            "perpoint_position_orientation_mse": get_perpoint_position_orientation_mse,
            "perpoint_total_mse": get_perpoint_total_mse,
            "perpoint_total_cone_mse": get_perpoint_total_cone_mse,
            "sq_functional_loss": get_sq_functional_loss,
            "sq_distance_loss": get_sq_distance_loss,
            "esq_functional_loss": get_esq_functional_loss,
            "esq_distance_loss": get_esq_distance_loss,
            "esq_distance_with_normal_loss": get_esq_distance_with_normal_loss,
            "esq_distance_with_boundary_loss": get_esq_distance_with_boundary_loss,
            "st_distance_loss": get_st_distance_loss,
            "st_distance_with_normal_loss": get_st_distance_with_normal_loss,
            "est_distance_loss": get_est_distance_loss,
            "superquadric_loss": get_superquadric_loss,
            "deformed_superquadric_loss": get_deformed_superquadric_loss,
            "deformed_superquadric_loss2": get_deformed_superquadric_loss2
        }[name]
    except:
        raise ("Loss {} not available".format(name))

def get_mse(device, **cfg):
    loss = MSE(device=device, **cfg)
    return loss

def get_perpoint_position_mse(device, **cfg):
    loss = PerpointPoisitionMSE(device=device, **cfg)
    return loss

def get_perpoint_position_orientation_mse(device, **cfg):
    loss = PerpointPoisitionOrientationMSE(device=device, **cfg)
    return loss

def get_perpoint_total_mse(device, **cfg):
    loss = PerpointTotalMSE(device=device, **cfg)
    return loss

def get_perpoint_total_cone_mse(device, **cfg):
    loss = PerpointTotalConeMSE(device=device, **cfg)
    return loss

def get_sq_functional_loss(device, **cfg):
    loss = SuperquadricFunctionalLoss(device=device, **cfg)
    return loss

def get_sq_distance_loss(device, **cfg):
    loss = SuperquadricDistanceLoss(device=device, **cfg)
    return loss

def get_esq_functional_loss(device, **cfg):
    loss = ExtendedSuperquadricFunctionalLoss(device=device, **cfg)
    return loss

def get_esq_distance_loss(device, **cfg):
    loss = ExtendedSuperquadricDistanceLoss(device=device, **cfg)
    return loss

def get_esq_distance_with_normal_loss(device, **cfg):
    loss = ExtendedSuperquadricDistanceWithNormalLoss(device=device, **cfg)
    return loss

def get_esq_distance_with_boundary_loss(device, **cfg):
    loss = ExtendedSuperquadricDistanceWithBoundaryLoss(device=device, **cfg)
    return loss

def get_st_distance_loss(device, **cfg):
    loss = SupertoroidDistanceLoss(device=device, **cfg)
    return loss

def get_st_distance_with_normal_loss(device, **cfg):
    loss = SupertoroidDistanceWithNormalLoss(device=device, **cfg)
    return loss

def get_est_distance_loss(device, **cfg):
    loss = ExtendedSupertoroidDistanceLoss(device=device, **cfg)
    return loss

def get_superquadric_loss(device, **cfg):
    loss = SuperquadricLoss(device=device, **cfg)
    return loss

def get_deformed_superquadric_loss(device, **cfg):
    loss = DeformedSuperquadricLoss(device=device, **cfg)
    return loss

def get_deformed_superquadric_loss2(device, **cfg):
    loss = DeformedSuperquadricLoss2(device=device, **cfg)
    return loss
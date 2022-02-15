from .dgcnn import DGCNN
from models.segmenation_network import SegmentationNetwork


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
            "segmentation": SegmentationNetwork,
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

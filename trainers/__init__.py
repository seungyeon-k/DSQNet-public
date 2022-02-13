from .segmentation_trainer import SegmentationTrainer
from .base_sq import BaseSQTrainer


def get_trainer(cfg):
    trainer_type = cfg.get("trainer", None)
    device = cfg["device"]
    if trainer_type == "segmentation":
        trainer = SegmentationTrainer(cfg["training"], device=device)
    elif trainer_type == "base_sq":
        trainer = BaseSQTrainer(cfg["training"], device=device)
    else:
        raise NotImplementedError(f"trainer {trainer_type} not implemented")
    return trainer


import time
import os
import torch
import numpy as np
from metrics import averageMeter
from loss import get_loss


class BaseSQTrainer:
    """Trainer for a conventional iterative training of model"""
    def __init__(self, training_cfg, device):
        self.cfg = training_cfg
        self.device = device
        self.d_val_result = {}
        self.loss_class = get_loss(training_cfg, device=device)
        self.loss = self.loss_class.loss

    def train(self, model, opt, d_dataloaders, logger=None, logdir=''):
        cfg = self.cfg
        best_val_loss = np.inf
        time_meter = averageMeter()
        i = 0
        train_loader, val_loader = d_dataloaders['training'], d_dataloaders['validation']

        for i_epoch in range(cfg.n_epoch):

            for x, x_gt, y, l_gt in train_loader:
                i += 1

                model.train()
                x = x.to(self.device)
                x_gt = x_gt.to(self.device)
                y = y.to(self.device)
                l_gt = l_gt.to(self.device)

                start_ts = time.time()
                d_train = model.train_step(x, x_gt = x_gt, y=y, l_gt=l_gt, optimizer=opt, loss=self.loss)
                
                time_meter.update(time.time() - start_ts)
                logger.process_iter_train(d_train)

                if i % cfg.print_interval == 0:
                    d_train = logger.summary_train(i)
                    print(f"Iter [{i:d}] Avg Loss: {d_train['loss/train_loss_']:.4f} Elapsed time: {time_meter.sum:.4f}")
                    time_meter.reset()

                if i % cfg.val_interval == 0:
                    model.eval()
                    j = 0
                    for val_x, val_x_gt, val_y, val_l_gt in val_loader:
                        j += 1
                        val_x = val_x.to(self.device)
                        val_x_gt = val_x_gt.to(self.device)
                        val_y = val_y.to(self.device)
                        val_l_gt = val_l_gt.to(self.device)

                        d_val = model.validation_step(val_x, x_gt = val_x_gt, y=val_y, l_gt = val_l_gt, loss=self.loss)
                        logger.process_iter_val(d_val)

                        if j == 3:
                            break
                    d_val = logger.summary_val(i)
                    val_loss = d_val['loss/val_loss_']
                    print(d_val['print_str'])
                    best_model = val_loss < best_val_loss

                    if i % cfg.save_interval == 0 or best_model:
                        self.save_model(model, logdir, best=best_model, i_iter=i)
                    if best_model:
                        print(f'Iter [{i:d}] best model saved {val_loss} <= {best_val_loss}')
                        best_val_loss = val_loss
            if i_epoch % cfg.save_interval == 0:
                self.save_model(model, logdir, best=False, i_epoch=i_epoch)

        return model, best_val_loss

    def save_model(self, model, logdir, best=False, i_iter=None, i_epoch=None):
        if best:
            pkl_name = "model_best.pkl"
        else:
            if i_iter is not None:
                pkl_name = "model_iter_{}.pkl".format(i_iter)
            else:
                pkl_name = "model_epoch_{}.pkl".format(i_epoch)
        state = {"epoch": i_epoch, "model_state": model.state_dict(), 'iter': i_iter}
        save_path = os.path.join(logdir, pkl_name)
        torch.save(state, save_path)
        print(f'Model saved: {pkl_name}')
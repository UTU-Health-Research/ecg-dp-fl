#!/usr/bin/env python

import os
import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
import opacus
from opacus.utils.batch_memory_manager import BatchMemoryManager
from openfl.utilities.optimizers.torch import FedProxAdam
import collections

from .metrics import cal_multilabel_metrics

RDP_ALPHAS = [1 + x / 100.0 for x in range(1, 10)] + [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 100))
LEARNING_RATE = 0.001
MU = 0.1
WEIGHT_DECAY=0.00001

# WARNING if this larger than a training set Opacus explodes
PHYS_BATCH_SIZE=256

FC_PREFIX = "fc."

def get_params(model):
    if isinstance(model, nn.Module):
        return model.parameters()
    else:
        return model

def adamopt(model, lr=LEARNING_RATE):
    optimizer = torch.optim.Adam(get_params(model), lr=lr,
            weight_decay=WEIGHT_DECAY)
    return optimizer

def sgdopt(model, lr=LEARNING_RATE):
    optimizer = torch.optim.SGD(get_params(model), lr=lr)
    return optimizer

def fedproxopt(model, lr=LEARNING_RATE, mu=MU):
    optimizer = FedProxAdam(get_params(model), mu=mu, lr=lr,
            weight_decay=WEIGHT_DECAY)
    return optimizer

class ModelBase:
    def __init__(self, model, labels, device):
        self.setup_model(model, labels, device)

    def setup_model(self, model, labels, device):
        self.device = device
        self.labels = labels
        self.model = model
        self.model.to(self.device)

    def get_weights(self):
        """model state dict on GPU -> list of numpy arrays on CPU"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_weights(self, parameters):
        """list of numpy arrays -> model state dict"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = collections.OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def set_classifier(self, parameters, prefix=FC_PREFIX):
        """Update classifier only, parameters should represent a fully connected layer"""
        state_dict = self.model.state_dict()
        state_dict[prefix + "weight"] = torch.tensor(parameters[0])
        state_dict[prefix + "bias"] = torch.tensor(parameters[1])
        self.model.load_state_dict(state_dict, strict=True)

    def update_weights(self, parameters):
        """Update only those weights that are compatible with the model"""
        curr_params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        new_params = []
        for p_curr, p_update in zip(reversed(curr_params), reversed(parameters)):
            if p_curr.shape == p_update.shape:
                new_params.insert(0, p_update)
            else:
                new_params.insert(0, p_curr)  # happens when the classifier dimensions do not match
        self.set_weights(new_params)

class Predict(ModelBase):
    def __init__(self, model, test_dl, labels, device):
        self.setup_model(model, labels, device)
        self.setup_output()
        self.test_dl = test_dl

    def setup_output(self):
        self.sigmoid = nn.Sigmoid()
        self.sigmoid.to(self.device)

    def _loss(self, logits, labels):
        return None

    def _setup_step(self):
        self.labels_all = torch.tensor((), device=self.device)
        self.logits_all = torch.tensor((), device=self.device)
        self.logits_prob_all = torch.tensor((), device=self.device)

    def _cleanup_step(self):
        del self.labels_all
        del self.logits_all
        del self.logits_prob_all

    def _eval_step_fwd(self, batch):
        logits = self.model(*batch[:-1])
        labels = batch[-1]
        loss = self._loss(logits, labels)
        return logits, labels, loss

    def _eval_step(self, batch):
        logits, labels, loss = self._eval_step_fwd(batch)
        logits_prob = self.sigmoid(logits)
        self.labels_all = torch.cat((self.labels_all, labels), 0)
        self.logits_all = torch.cat((self.logits_all, logits), 0)
        self.logits_prob_all = torch.cat((self.logits_prob_all, logits_prob), 0)
        return loss

    def evaluate(self, plot_func=None):
        history = {}
        self.model.eval()
        self._setup_step()

        batch_loss = 0.0
        batch_count = 0

        for i, batch in enumerate(self.test_dl):
            batch = [v.to(self.device) for v in batch]

            with torch.set_grad_enabled(False):
                _ = self._eval_step(batch)

            if i % 10 == 0:
                print('{:<4}/{:>4} predictions made'.format(i+1, len(self.test_dl)))

        # Predicting metrics
        test_macro_avg_prec, test_micro_avg_prec, test_macro_auroc, test_micro_auroc = cal_multilabel_metrics(self.labels_all, self.logits_prob_all, self.labels, 0.5)

        print('macro avg prec: {:<6.3f} micro avg prec: {:<6.3f} macro auroc: {:<6.3f} micro auroc: {:<6.3f} '.format(
            test_macro_avg_prec,
            test_micro_avg_prec,
            test_macro_auroc,
            test_micro_auroc))

        # Draw ROC curve for predictions
        if plot_func:
            # use eg src.modeling.metrics.roc_curves_notebook
            plot_func(self.labels_all, self.logits_prob_all, self.labels)
        test_loss = F.binary_cross_entropy(self.logits_prob_all, self.labels_all).detach().cpu().numpy()

        # Add information to testing history
        history['test_micro_auroc'] = test_micro_auroc
        history['test_micro_avg_prec'] = test_micro_avg_prec
        history['test_macro_auroc'] = test_macro_auroc
        history['test_macro_avg_prec'] = test_macro_avg_prec
        history['test_loss'] = test_loss.mean()

        self._cleanup_step()
        torch.cuda.empty_cache()
        return history

    # XXX: logits parameter is ignored, always compute both
    def predict_proba(self, logits=False):
        history = {}
        self.model.eval()
        self._setup_step()

        for i, batch in enumerate(self.test_dl):
            batch = [v.to(self.device) for v in batch]

            with torch.set_grad_enabled(False):
                _ = self._eval_step(batch)

            if i % 10 == 0:
                print('{:<4}/{:>4} predictions made'.format(i+1, len(self.test_dl)))

        # Computed logits
        history['labels_all'] = self.labels_all.clone()
        history['logits_all'] = self.logits_all.clone()
        history['logits_prob_all'] = self.logits_prob_all.clone()

        self._cleanup_step()
        torch.cuda.empty_cache()
        return history

class Extract(Predict):
    def predict_proba(self, logits=False):
        history = {}
        self.model.eval()
        self._setup_step()
        self.ag_all = torch.tensor((), device=self.device)

        for i, batch in enumerate(self.test_dl):
            batch = [v.to(self.device) for v in batch]
            self.ag_all = torch.cat((self.ag_all, batch[1]), 0)

            with torch.set_grad_enabled(False):
                _ = self._eval_step(batch)

            if i % 10 == 0:
                print('{:<4}/{:>4} predictions made'.format(i+1, len(self.test_dl)))

        # Computed logits
        history['labels_all'] = self.labels_all.clone()
        history['logits_all'] = self.logits_all.clone()
        history['logits_prob_all'] = self.logits_prob_all.clone()
        history['ag_all'] = self.ag_all.clone()

        del self.ag_all
        self._cleanup_step()
        torch.cuda.empty_cache()
        return history

class Training(Predict):
    def __init__(self, model, optfunc, train_dl, labels, device, cp_path, name=""):
        self.setup_model(model, labels, device)
        self.setup_output()
        self.train_dl = train_dl
        self.name = name
        self.cp_path = cp_path
        self.cp_freq = None

        self.optfunc = optfunc
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = self.optfunc(self.model)

    def install_scheduler(self, scheduler):
        self.scheduler = scheduler

    def train(self, epochs, log=True):
        #self.optimizer = self.optfunc(self.model)
        return self._train(self.train_dl, epochs, log)

    def _loss(self, logits, labels):
        return self.criterion(logits, labels)

    def _train_step(self, batch):
        logits, labels, loss = self._eval_step_fwd(batch)
        logits_prob = self.sigmoid(logits)
        self.labels_all = torch.cat((self.labels_all, labels), 0)
        self.logits_prob_all = torch.cat((self.logits_prob_all, logits_prob), 0)
        return loss

    def _epoch_summary(self, epoch, epochs, history, train_loss, log=True):
        train_macro_avg_prec, train_micro_avg_prec, train_macro_auroc, train_micro_auroc = cal_multilabel_metrics(self.labels_all, self.logits_prob_all, self.labels, 0.5)

        if "train_loss" not in history:
            history['train_loss'] = []
            history['train_micro_auroc'] = []
            history['train_micro_avg_prec'] = []
            history['train_macro_auroc'] = []
            history['train_macro_avg_prec'] = []

        history['train_loss'].append(train_loss)
        history['train_micro_auroc'].append(train_micro_auroc)
        history['train_micro_avg_prec'].append(train_micro_avg_prec)
        history['train_macro_auroc'].append(train_macro_auroc)
        history['train_macro_avg_prec'].append(train_macro_avg_prec)

        if log:
            print('{} epoch {:^4}/{:^4} train loss: {:<6.3f}  train macro auroc: {:<6.3f}'.format(
                self.name,
                epoch,
                epochs,
                train_loss,
                train_macro_auroc))

    def finish_epoch(self, epoch_num, history):
        """Can override for DP accounting needs"""
        if hasattr(self, "scheduler"):
            self.scheduler.step()
        if self.cp_freq is not None and epoch_num % self.cp_freq == 0:
            local_weights = self.get_weights()
            # remember the saved model path
            self.model_path = os.path.join(self.cp_path, "{}_epoch_cp_{}.npz".format(self.name, epoch_num))
            np.savez(self.model_path, *local_weights)
        return True

    def _train(self, train_dl, epochs, log=True):
        history = {}

        for epoch in range(1, epochs+1):
            self._setup_step()
            self.model.train()
            train_loss = 0.0

            batch_loss = 0.0
            batch_count = 0
            sample_count = 0
            step = 1

            for batch_idx, batch in enumerate(train_dl):
                batch = [v.to(self.device) for v in batch]
                batch_sz = batch[0].size(0)
                with torch.set_grad_enabled(True):
                    loss = self._train_step(batch)
                    loss_tmp = loss.item() * batch_sz
                    train_loss += loss_tmp

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Printing training information
                    sample_count += batch_sz
                    if step % 5 == 0:
                        batch_loss += loss_tmp
                        batch_count += batch_sz
                        batch_loss = batch_loss / batch_count
                        if log:
                            print('epoch {:^3} [{}/{}] train loss: {:>5.2f}'.format(
                                epoch,
                                sample_count,
                                len(train_dl.dataset),
                                batch_loss
                            ))

                        batch_loss = 0.0
                        batch_count = 0
                    step += 1

            train_loss = train_loss / sample_count
            self._epoch_summary(epoch, epochs, history, train_loss, log)

            self._cleanup_step()
            torch.cuda.empty_cache()
            if not self.finish_epoch(epoch, history):
                # finished training early
                break
        return history

    def summary_stats(self):
        return {}

class TrainAndSelect(Training):
    def __init__(self, model, optfunc, train_dl, val_dl, labels, device, cp_path, name="", patience=-1):
        super().__init__(model, optfunc, train_dl, labels, device, cp_path, name)
        self.val_dl = val_dl
        self.patience = patience
        self.pat_eps = 0.0001  # minimum improvement in validation loss

    def _train_step(self, batch):
        _, _, loss = self._eval_step_fwd(batch)
        return loss

    def _epoch_summary(self, epoch, epochs, history, train_loss, val_loss, log=True):
        _, _, val_macro_auroc, val_micro_auroc = cal_multilabel_metrics(self.labels_all, self.logits_prob_all, self.labels, 0.5)

        if "train_loss" not in history:
            history['train_loss'] = []
            history['val_loss'] = []
            history['val_micro_auroc'] = []
            history['val_macro_auroc'] = []

        history['val_loss'].append(val_loss)
        history['train_loss'].append(train_loss)
        history['val_micro_auroc'].append(val_micro_auroc)
        history['val_macro_auroc'].append(val_macro_auroc)

        if log:
            print('{} epoch {:^4}/{:^4} val loss: {:<6.3f}  val macro auroc: {:<6.3f}'.format(
                self.name,
                epoch,
                epochs,
                val_loss,
                val_macro_auroc))

    def finish_epoch(self, epoch_num, history):
        if super().finish_epoch(epoch_num, history):
            if self.patience > 0 and epoch_num > self.patience:
                recent_best = min(history["val_loss"][-self.patience:])
                earlier_best = min(history["val_loss"][:-self.patience])
                if earlier_best - self.pat_eps >= recent_best:
                    return True
            else:
                return True
        return False

    def train(self, epochs, log=True):
        #self.optimizer = self.optfunc(self.model)
        return self._train(self.train_dl, self.val_dl, epochs, log)

    def _train(self, train_dl, val_dl, epochs, log=True):
        history = {}
        best_loss = None
        self.best_model = None

        for epoch in range(1, epochs+1):
            self.model.train()
            train_loss = 0.0
            batch_loss = 0.0
            batch_count = 0
            sample_count = 0
            step = 1

            for batch_idx, batch in enumerate(train_dl):
                batch = [v.to(self.device) for v in batch]
                batch_sz = batch[0].size(0)
                with torch.set_grad_enabled(True):
                    loss = self._train_step(batch)
                    loss_tmp = loss.item() * batch_sz
                    train_loss += loss_tmp

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Printing training information
                    sample_count += batch_sz
                    if step % 5 == 0:
                        batch_loss += loss_tmp
                        batch_count += batch_sz
                        batch_loss = batch_loss / batch_count
                        if log:
                            print('epoch {:^3} [{}/{}] train loss: {:>5.2f}'.format(
                                epoch,
                                sample_count,
                                len(train_dl.dataset),
                                batch_loss
                            ))

                        batch_loss = 0.0
                        batch_count = 0
                    step += 1

            train_loss = train_loss / sample_count

            self.model.eval()
            val_loss = 0.0
            self._setup_step()

            for batch_idx, batch in enumerate(val_dl):
                batch = [v.to(self.device) for v in batch]
                batch_sz = batch[0].size(0)

                with torch.set_grad_enabled(False):
                    loss = self._eval_step(batch)
                    loss_tmp = loss.item() * batch_sz
                    val_loss += loss_tmp

            val_loss = val_loss / len(val_dl.dataset)
            self._epoch_summary(epoch, epochs, history, train_loss, val_loss, log)

            if best_loss is None or best_loss > val_loss:
                best_loss = val_loss
                self.best_model = self.get_weights()

            self._cleanup_step()
            torch.cuda.empty_cache()
            if not self.finish_epoch(epoch, history):
                # finished training early
                break
        return history

    def summary_stats(self):
        return {}

class DPMixin:
    def calc_delta(self, n_samples):
        return 1 / (10 * n_samples)

    def make_private(self, epsilon, max_grad_norm, e_epochs, delta=None, sigma=None, phys_batch_size=PHYS_BATCH_SIZE):
        self.accountant = "rdp"
        self.epsilon = epsilon
        self.n_samples = len(self.train_dl.dataset)
        self.sample_rate = self.train_dl.batch_size / self.n_samples
        self.max_grad_norm = max_grad_norm
        self.e_epochs = e_epochs # expected epochs
        self.eps_history = []
        self.phys_batch_size = phys_batch_size

        if delta is None:
            self.delta = self.calc_delta(self.n_samples)
        else:
            self.delta = delta

        if sigma is None:
            sigma = opacus.accountants.utils.get_noise_multiplier(
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                sample_rate=self.sample_rate,
                epochs=self.e_epochs,
                accountant=self.accountant,
                alphas=RDP_ALPHAS
            )
        return self._make_private(sigma)

    def _make_private(self, sigma):
        self.privacy_engine = opacus.PrivacyEngine(accountant=self.accountant)

        self.model, self.optimizer, self.train_dl = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_dl,
            noise_multiplier=sigma,
            max_grad_norm=self.max_grad_norm
        )
        self.sigma = sigma

    def _train_batchmm(self, optimizer, *args, **kwargs):
        if not hasattr(self, "privacy_engine"):
            raise RuntimeError("must call make_private() to properly initialize the DPTraining class")

        with BatchMemoryManager(data_loader=self.train_dl,
            max_physical_batch_size=self.phys_batch_size,
            optimizer=optimizer
        ) as new_data_loader:
            hist = self._train(new_data_loader, *args, **kwargs)
        return hist

    def summary_stats(self):
        return {"epsilon": self.eps_history,
                "delta": self.delta,
                "sigma": self.sigma,
        }

class DPTraining(DPMixin, Training):
    def train(self, epochs, log=True):
        return self._train_batchmm(self.optimizer, epochs, log=log)

    def finish_epoch(self, epoch_num, history):
        if super().finish_epoch(epoch_num, history):
            if self.sigma > 0:
                e = self.privacy_engine.get_epsilon(delta=self.delta)
                self.eps_history.append(e)
                if e < self.epsilon:
                    return True # keep training
            else:
                return True
        return False

class PartialDPOptimizer(opacus.optimizers.DPOptimizer):
    """Apply noise mechanism to part of model
       (currently the entire model still receives gradient clipping)
    """
    def add_noise(self):
        for p in self.params:
            opacus.optimizers.optimizer._check_processed_flag(p.summed_grad)

            if hasattr(p, "dp_flag_no_noise"):
                # this wastes some resources by generating the noise tensor,
                # but just assigning p.summed_grad to p.grad may cause subtle issues
                # because the code manipulating these tensors comes from two
                # separate codebases, neither assumes data sharing with another tensor
                z = 0.0
            else:
                z = self.noise_multiplier
            noise = opacus.optimizers.optimizer._generate_noise(
                std=(z * self.max_grad_norm),
                reference=p.summed_grad,
                generator=self.generator,
                secure_mode=self.secure_mode,
            )
            p.grad = (p.summed_grad + noise).view_as(p)

            opacus.optimizers.optimizer._mark_as_processed(p.summed_grad)

class PartialPrivacyEngine(opacus.PrivacyEngine):
    """Privacy engine with the DP optimizer applied to part of model only
       set dp_flag_no_noise attribute to parameters to disable noise
    """
    def _prepare_optimizer(
        self,
        *,
        optimizer,
        noise_multiplier,
        max_grad_norm,
        expected_batch_size,
        loss_reduction="mean",
        distributed=False,
        clipping="flat",
        noise_generator=None,
        grad_sample_mode="hooks",
        **kwargs,
    ):
        generator = None
        if self.secure_mode:
            generator = self.secure_rng
        elif noise_generator is not None:
            generator = noise_generator

        return PartialDPOptimizer(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=self.secure_mode,
            **kwargs,
        )

class SingleLinear(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.fc(x)

#
# Dataset- and model agnostic setup
# (need to pick model arch, initialize model and dataloaders in advance)
#
def trainer_setup(model, train_dl, labels, device, optfunc=adamopt, val_dl=None, cp_params={}, privacy_params={},
        name="", trainer_class=None):
    cp_path = cp_params["path"]

    if trainer_class is None:
        if val_dl:
            if privacy_params:
                # model selection undermines DP anyway, although in principle non-private validation set could be used
                raise NotImplementedError("model selection is not implemented for differentially private training")
            else:
                trainer_class = TrainAndSelect
        else:
            if privacy_params:
                trainer_class = DPTraining
            else:
                trainer_class = Training

    if val_dl:
        trainer = trainer_class(model, optfunc, train_dl, val_dl, labels, device, cp_path, name=name)
    else:
        trainer = trainer_class(model, optfunc, train_dl, labels, device, cp_path, name=name)

    if privacy_params:
        trainer.make_private(
            privacy_params["epsilon"],
            privacy_params["max_grad_norm"],
            privacy_params["e_epochs"],
            delta=privacy_params.get("delta"),  # calculated if not given
            sigma=privacy_params.get("sigma"),  # noise magnitude, calculated unless given
            phys_batch_size=privacy_params.get("phys_batch_size", PHYS_BATCH_SIZE)
        )

    if "freq" in cp_params:
        trainer.cp_freq = cp_params["freq"]

    return trainer



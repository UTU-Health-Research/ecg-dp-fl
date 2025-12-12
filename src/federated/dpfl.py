#!/usr/bin/env python
# coding: utf-8

import os
import os.path
import torch
import numpy as np
import collections

import opacus
from openfl.experimental.workflow.interface import FLSpec
from openfl.experimental.workflow.placement import aggregator, collaborator
from openfl.experimental.workflow.interface import Collaborator, Aggregator
from openfl.experimental.workflow.runtime import LocalRuntime

from src.modeling.train_utils import DPTraining, PartialPrivacyEngine

# define the classifier layer(s) in original and Opacus-wrapped models
FC_PREFIX = "fc."
FC_PRIVATE_PREFIX = "_module." + FC_PREFIX

# default path
CHECKPOINT_PATH = "checkpoints"

def params_tensor(params, device):
    c = []
    for p in params:
        c.append(torch.zeros(p.shape, device=device, dtype=p.dtype))
    return c

class DPTrainEmbed(DPTraining):
    def get_weights(self):
        # for easier finetuning compatibility, return classifier with 0 weights
        return [np.zeros(val.shape) if k.startswith(FC_PRIVATE_PREFIX) else val.cpu().numpy()
                for k, val in self.model.state_dict().items()]

    def set_weights(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = collections.OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        old_state_dict = self.model.state_dict()
        state_dict[FC_PRIVATE_PREFIX + "weight"] = old_state_dict[FC_PRIVATE_PREFIX + "weight"]
        state_dict[FC_PRIVATE_PREFIX + "bias"] = old_state_dict[FC_PRIVATE_PREFIX + "bias"]
        self.model.load_state_dict(state_dict, strict=True)

    def _make_private(self, sigma):
        for p in self.model.fc.parameters():
        #    p.dp_flag_no_clip = True
            p.dp_flag_no_noise = True

        self.privacy_engine = PartialPrivacyEngine(accountant=self.accountant)
        self.model, self.optimizer, self.train_dl = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_dl,
            noise_multiplier=sigma,
            max_grad_norm=self.max_grad_norm
        )
        self.sigma = sigma

class DPTrainScaffold(DPTraining):
    def _make_private(self, sigma):
        super()._make_private(sigma)
        self.optimizer_accounting_hook = self.optimizer.step_hook
        self.optimizer.attach_step_hook(self.drift_correction)

    def drift_correction(self, opt):
        """
        DP-SCAFFOLD drift correction (https://arxiv.org/abs/2111.09278)

        This runs after the noise mechanism, so p.grad is differentially private
        """
        for i, p in enumerate(opt.params):
            if self.K == 0:
                self.local_variate_c_next[i] += p.grad
            if self.global_variate_c:  # start doing this after the first FL step
                                       # when the variates have accumulated
                p.grad -= self.local_variate_c[i]
                p.grad += self.global_variate_c[i]
        if self.optimizer_accounting_hook is not None:
            self.optimizer_accounting_hook(opt)
        self.K += 1
    
    def train(self, epochs, log=True):
        self.K = 0
        self.local_variate_c_next = params_tensor(self.optimizer.params, self.device)
        return super().train(epochs, log)

    def update_local_variate(self):
        if self.K:
            #print(f"updating local variate, {self.K}")
            #new_c = []
            #for i, p in enumerate(self.model.parameters()):
            #    new_c.append(self.local_variate_c_next[i] / self.K)
            #self.local_variate_c = new_c
            self.local_variate_c = self.local_variate_c_next
            self.K = 0

    def get_local_variate(self):
        self.update_local_variate()
        return [t.cpu().numpy() for t in self.local_variate_c]

    def set_global_variate(self, gv):
        self.global_variate_c = [torch.from_numpy(arr).to(self.device)
            for arr in gv]

class DPTrainStep(DPTraining):
    def save_gradient(self, lr=1.0):
        """model state dict on GPU -> list of numpy arrays on CPU"""
        self.last_gradient = [p.grad.cpu().numpy() for p in self.model.parameters()]

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

            step_done = False
            for batch_idx, batch in enumerate(train_dl):
                batch = [v.to(self.device) for v in batch]
                batch_sz = batch[0].size(0)
                with torch.set_grad_enabled(True):
                    loss = self._train_step(batch)
                    loss_tmp = loss.item() * batch_sz
                    train_loss += loss_tmp

                    self.optimizer.zero_grad()
                    loss.backward()
                    # must be Opacus DPOptimizer
                    if self.optimizer.pre_step():
                        step_done = True

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
                if step_done:
                    self.save_gradient()
                    break

            train_loss = train_loss / sample_count
            self._epoch_summary(epoch, epochs, history, train_loss, log)

            self._cleanup_step()
            torch.cuda.empty_cache()
            if not self.finish_epoch(epoch, history):
                # finished training early
                break
        return history

#
# XXX: NO validation/eval data because we are using 4 entire silos to train
#      then test on separate hold out set
def fl_setup(train_silos):
    train_names = list(train_silos.keys())
    collaborators = []
    global_stats = {}
    for k in train_names:
        c = Collaborator(name=k)
        c.private_attributes = {
            "collaborator_name": k
        }
        global_stats[k] = {}
        collaborators.append(c)

    agg = Aggregator()
    agg.private_attributes = {"global_stats": global_stats}

    local_runtime = LocalRuntime(aggregator=agg, collaborators=collaborators, backend='single_process')
    #print(f'Local runtime collaborators = {local_runtime.collaborators}')
    return local_runtime

def fedavg_arrays(all_params, weights=None):
    """all_params: list of numpy arrays"""
    n = len(all_params[0])
    new_params = []
    for i in range(n):
        new_params.append(np.average([params[i] for params in all_params],
                                                      axis=0,
                                                      weights=weights))
    return new_params

def stats_append(append_from, append_to, round_num):
    def append_key(k):
        if k + "_x" not in append_to:
            append_to[k + "_x"] = []
            append_to[k + "_y"] = []
        x = np.linspace(round_num - 1, round_num, len(append_from[k + ""]) + 1)
        append_to[k + "_x"] += x.tolist()[1:]
        append_to[k + "_y"] += append_from[k + ""]
    append_key("train_loss")
    append_key("train_macro_auroc")

#
# No validation/eval in this flow
# Will save checkpoints so that progression of test performance can be understood
# BUT selecting the best model based on checkpoint history CANNOT be used because this will leak information
class FederatedFlow(FLSpec):
    def __init__(self, num_rounds=3, local_epoch=1, n_collab=2, cp_path=CHECKPOINT_PATH, trainer_log=False):
        self.global_weights = []
        self.num_rounds = num_rounds
        self.local_epoch = local_epoch
        self.n_selected_collaborators = n_collab
        self.get_trainer = lambda x: None    # hide trainers from OpenFL behind a callable
        self._checkpoint = False
        self.cp_path = cp_path
        self.trainer_log = trainer_log

    @aggregator
    def start(self):
        self.current_round = 0
        self.collaborators = self.runtime.collaborators  # Fetch the collaborators dynamically
        self.next(self.internal_loop)

    @aggregator
    def select_collaborators(self):
        self.selected_collaborator_indices = np.random.choice(range(len(self.collaborators)), \
            self.n_selected_collaborators, replace=False)
        self.selected_collaborators = [self.collaborators[idx] for idx in self.selected_collaborator_indices]
        self.next(self.train, foreach="selected_collaborators")

    @collaborator
    def train(self):
        trainer = self.get_trainer(self.collaborator_name)
        if self.global_weights:
            trainer.set_weights(self.global_weights)
        history = trainer.train(self.local_epoch, log=self.trainer_log)

        self.local_stats = {"train_loss": history["train_loss"],
                "train_macro_auroc": history["train_macro_auroc"],
                "name": self.collaborator_name}
        self.local_weights = trainer.get_weights()
        self.train_dataset_length = len(trainer.train_dl.dataset)
        self.next(self.join)

    @aggregator
    def join(self, inputs):
        train_datasize = sum([input_.train_dataset_length for input_ in inputs])
        collab_weights = []
        for input_ in inputs:
            local_name = input_.local_stats["name"]
            stats_append(input_.local_stats, self.global_stats[local_name], self.current_round)
            collab_weights.append(input_.train_dataset_length / train_datasize)

        self.global_weights = fedavg_arrays([input_.local_weights
                                             for input_ in inputs], collab_weights)
        self.model_path = os.path.join(self.cp_path, "fl_global_cp_{}.npz".format(self.current_round))
        np.savez(self.model_path, *self.global_weights)
        self.next(self.internal_loop)

    @aggregator
    def internal_loop(self):
        if self.current_round == self.num_rounds:
            self.next(self.final_report, foreach="collaborators")
        else:
            self.current_round += 1
            print(f"current round : {self.current_round}")
            self.next(self.select_collaborators)

    @collaborator
    def final_report(self):
        self.local_summary = {"stats": self.get_trainer(self.collaborator_name).summary_stats(),
                              "name": self.collaborator_name}
        self.next(self.end)

    @aggregator
    def end(self, inputs):
        self.summary = {}
        for input_ in inputs:
            local_name = input_.local_summary["name"]
            self.summary[local_name] = input_.local_summary["stats"]
        self.stats = self.global_stats
        print(f"Federated learning complete after {self.num_rounds} rounds.")

class ScaffoldFlow(FederatedFlow):
    def __init__(self, num_rounds=3, local_epoch=1, n_collab=2, cp_path=CHECKPOINT_PATH, trainer_log=False):
        super().__init__(num_rounds, local_epoch, n_collab, cp_path, trainer_log)
        self.global_variate_c = []

    @collaborator
    def train(self):
        trainer = self.get_trainer(self.collaborator_name)
        if self.global_weights:
            trainer.set_weights(self.global_weights)
        trainer.set_global_variate(self.global_variate_c)
        history = trainer.train(self.local_epoch, log=self.trainer_log)

        self.local_stats = {"train_loss": history["train_loss"],
                "train_macro_auroc": history["train_macro_auroc"],
                "name": self.collaborator_name}
        self.local_weights = trainer.get_weights()
        self.local_variate_c = trainer.get_local_variate()
        self.train_dataset_length = len(trainer.train_dl.dataset)
        self.next(self.join)

    @aggregator
    def join(self, inputs):
        train_datasize = sum([input_.train_dataset_length for input_ in inputs])
        for input_ in inputs:
            local_name = input_.local_stats["name"]
            stats_append(input_.local_stats, self.global_stats[local_name], self.current_round)

        l = len(inputs)
        collab_weights = [1/l for i in range(l)]
        self.global_weights = fedavg_arrays([input_.local_weights for input_ in inputs],
                                             collab_weights)
        self.global_variate_c = fedavg_arrays([input_.local_variate_c for input_ in inputs],
                                              collab_weights)
        self.model_path = os.path.join(self.cp_path, "fl_global_cp_{}.npz".format(self.current_round))
        np.savez(self.model_path, *self.global_weights)
        self.next(self.internal_loop)

class FedProxFlow(FederatedFlow):
    @collaborator
    def train(self):
        trainer = self.get_trainer(self.collaborator_name)
        if self.global_weights:
            trainer.set_weights(self.global_weights)

        # trainer must have FedProx optimizer wrapped in a Opacus optimizer
        trainer.optimizer.original_optimizer.set_old_weights(
            [p.clone().detach() for p in trainer.model.parameters()])
        history = trainer.train(self.local_epoch, log=self.trainer_log)

        self.local_stats = {"train_loss": history["train_loss"],
                "train_macro_auroc": history["train_macro_auroc"],
                "name": self.collaborator_name}
        self.local_weights = trainer.get_weights()
        self.train_dataset_length = len(trainer.train_dl.dataset)
        self.next(self.join)

class MRMTLFlow(FederatedFlow):
    @collaborator
    def train(self):
        trainer = self.get_trainer(self.collaborator_name)
        if self.current_round == 1 and self.global_weights:
            trainer.set_weights(self.global_weights)

        # trainer must have FedProx optimizer wrapped in a Opacus optimizer
        if self.global_weights:
            trainer.optimizer.original_optimizer.set_old_weights(
                [torch.tensor(v).to(trainer.device) for v in self.global_weights])
        history = trainer.train(self.local_epoch, log=self.trainer_log)

        self.local_stats = {"train_loss": history["train_loss"],
                "train_macro_auroc": history["train_macro_auroc"],
                "name": self.collaborator_name}
        self.local_weights = trainer.get_weights()
        self.train_dataset_length = len(trainer.train_dl.dataset)
        self.next(self.join)

    @aggregator
    def join(self, inputs):
        train_datasize = sum([input_.train_dataset_length for input_ in inputs])
        collab_weights = []
        model_paths = {}
        for input_ in inputs:
            local_name = input_.local_stats["name"]
            stats_append(input_.local_stats, self.global_stats[local_name], self.current_round)
            collab_weights.append(input_.train_dataset_length / train_datasize)

            mp = os.path.join(self.cp_path, "fl_{}_cp_{}.npz".format(local_name, self.current_round))
            np.savez(mp, *input_.local_weights)
            model_paths[local_name] = mp

        self.global_weights = fedavg_arrays([input_.local_weights
                                             for input_ in inputs], collab_weights)
        self.model_path = model_paths
        self.next(self.internal_loop)

def add_noise(weights_array, sigma):
    res = []
    for w in weights_array:
        res.append(w + np.random.normal(loc=0, scale=sigma, size=w.shape))
    return res

# responsible for
# --- noise
# --- accounting
# collaborator trainers must be set up with sigma=0, constant q
class DPFlow(FederatedFlow):
    def make_private(self, n_samples, privacy_params={}):
        self.epsilon = privacy_params["epsilon"]
        self.sample_rate = privacy_params["sample_rate"]
        self.max_grad_norm = privacy_params["max_grad_norm"]
        self.e_epochs = privacy_params["e_epochs"]
        self.n_samples = n_samples
        self.eps_history = []

        if "delta" in privacy_params:
            self.delta = privacy_params["delta"]
        else:
            self.delta = 1 / (10 * self.n_samples)

        # precompute noise
        if self.epsilon > 0.00001:
            self.z = opacus.accountants.utils.get_noise_multiplier(
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                sample_rate=self.sample_rate,
                steps=self.e_epochs,
                accountant="rdp"
            )
        else:
            self.z = 0
        print(f"z = {self.z:.2f}")

        # epsilon tracking
        self.accountant = opacus.accountants.rdp.RDPAccountant()

    @aggregator
    def join(self, inputs):
        train_datasize = sum([input_.train_dataset_length for input_ in inputs])
        collab_weights = []
        for input_ in inputs:
            local_name = input_.local_stats["name"]
            stats_append(input_.local_stats, self.global_stats[local_name], self.current_round)
            collab_weights.append(input_.train_dataset_length / train_datasize)

        clipped_weights = fedavg_arrays([input_.local_weights
                                             for input_ in inputs], collab_weights)
        self.global_weights = add_noise(clipped_weights,
                                        ((self.z * self.max_grad_norm) / (self.sample_rate * self.n_samples)))

        self.accountant.step(noise_multiplier=self.z, sample_rate=self.sample_rate)
        eps, _ = self.accountant.get_privacy_spent(delta=self.delta)
        self.eps_history.append(eps)

        if self.current_round % 20 == 0:
            self.model_path = os.path.join(self.cp_path, "fl_global_cp_{}.npz".format(self.current_round))
            np.savez(self.model_path, *self.global_weights)
        self.next(self.internal_loop)

    @collaborator
    def final_report(self):
        self.local_summary = {"stats": self.get_trainer(self.collaborator_name).summary_stats(),
                              "name": self.collaborator_name}
        self.next(self.end)

    @aggregator
    def end(self, inputs):
        self.summary = {}
        for input_ in inputs:
            local_name = input_.local_summary["name"]
            self.summary[local_name] = input_.local_summary["stats"]
            self.summary[local_name]["epsilon"] = self.eps_history
        self.stats = self.global_stats
        print(f"Federated learning complete after {self.num_rounds} rounds.")

GRAD_SCALE = -1.0     # for tweaking
#GRAD_SCALE = -0.1
class DPAdamFlow(DPFlow):
    def __init__(self, num_rounds=3, local_epoch=1, n_collab=2, cp_path=CHECKPOINT_PATH, trainer_log=False, lr=0.003):
        super().__init__(num_rounds, local_epoch, n_collab, cp_path, trainer_log)
        self.eta = lr
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.tau = 0.000000001
        self.m_t = []
        self.v_t = []

    # https://flower.ai/docs/framework/_modules/flwr/server/strategy/fedadam.html#FedAdam
    def fedadam(self, fedavg_grad, current_weights):
        # get delta from aggregate weights
        #delta_t = [x - y for x, y in zip(fedavg_weights, current_weights)]
        if not current_weights:
                        current_weights = params_array(fedavg_grad)
        delta_t = [GRAD_SCALE * arr for arr in fedavg_grad]

        # m_t
        if not self.m_t:
            self.m_t = [np.zeros_like(x) for x in delta_t]
        self.m_t = [
            np.multiply(self.beta_1, x) + (1 - self.beta_1) * y
            for x, y in zip(self.m_t, delta_t)
        ]

        # v_t
        if not self.v_t:
            self.v_t = [np.zeros_like(x) for x in delta_t]
        self.v_t = [
            self.beta_2 * x + (1 - self.beta_2) * np.multiply(y, y)
            for x, y in zip(self.v_t, delta_t)
        ]

        eta_norm = (self.eta * np.sqrt(1 - np.power(self.beta_2, self.current_round + 1.0)) / (1 - np.power(self.beta_1, self.current_round + 1.0)))

        new_weights = [
            x + eta_norm * y / (np.sqrt(z) + self.tau)
            for x, y, z in zip(current_weights, self.m_t, self.v_t)
        ]
        return new_weights

    @collaborator
    def train(self):
        trainer = self.get_trainer(self.collaborator_name)
        if self.global_weights:
            trainer.set_weights(self.global_weights)
        history = trainer.train(self.local_epoch, log=self.trainer_log)

        self.local_stats = {"train_loss": history["train_loss"],
                            "train_macro_auroc": history["train_macro_auroc"],
                            "name": self.collaborator_name}
        self.local_grad = trainer.last_gradient
        self.train_dataset_length = len(trainer.train_dl.dataset)
        self.next(self.join)

    @aggregator
    def join(self, inputs):
        train_datasize = sum([input_.train_dataset_length for input_ in inputs])
        collab_weights = []
        for input_ in inputs:
            local_name = input_.local_stats["name"]
            stats_append(input_.local_stats, self.global_stats[local_name], self.current_round)
            collab_weights.append(input_.train_dataset_length / train_datasize)

        clipped_grad = fedavg_arrays([input_.local_grad
                                     for input_ in inputs], collab_weights)
        noisy_grad = add_noise(clipped_grad,
                                  ((self.z * self.max_grad_norm) / (self.sample_rate * self.n_samples)))

        #self.global_weights = self.fedadam(clipped_grad, self.global_weights)
        self.global_weights = self.fedadam(noisy_grad, self.global_weights)

        self.accountant.step(noise_multiplier=self.z, sample_rate=self.sample_rate)
        eps, _ = self.accountant.get_privacy_spent(delta=self.delta)
        self.eps_history.append(eps)

        if self.current_round % 20 == 0:
            self.model_path = os.path.join(self.cp_path, "fl_global_cp_{}.npz".format(self.current_round))
            np.savez(self.model_path, *self.global_weights)
        self.next(self.internal_loop)


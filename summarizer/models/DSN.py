import os
import sys
import time
import datetime
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from summarizer.models import Model
#from __future__ import print_function

import vsum_tools

"""
Deep Reinforcement Learning for Unsupervised Video Summarization with Diversity-Representativeness Reward
https://arxiv.org/pdf/1801.00054v3.pdf
https://github.com/KaiyangZhou/pytorch-vsumm-reinforce
"""

__all__ = ['DSN']

class DSN(nn.Module):
    """Deep Summarization Network"""
    def __init__(self, in_dim=1024, hid_dim=256, num_layers=1, cell='lstm'):
        super(DSN, self).__init__()
        assert cell in ['lstm', 'gru'], "cell must be either 'lstm' or 'gru'"
        if cell == 'lstm':
            self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.GRU(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hid_dim*2, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        p = F.sigmoid(self.fc(h))
        return p


class DSNModel(Model):
    def _init_model(self):
        model = DSN()
        self.beta = 0.01
        self.num_episode = 5
        return model

    def train(self, fold):
        start_time = time.time()
       
        self.model.train()
        
        train_keys, _ = self._get_train_test_keys(fold)
        baselines = {key: 0. for key in train_keys} # baseline rewards for videos
        reward_writers = {key: [] for key in train_keys} # record reward changes for each video

        criterion = nn.MSELoss()
        if self.hps.use_cuda:
            criterion = criterion.cuda()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hps.lr,
            weight_decay=self.hps.l2_req
        )

        # To record performances of the best epoch
        best_f_score = 0.0
        
        # For each epoch
        for epoch in range(self.hps.epochs_max):
            idxs = np.arange(len(train_keys))
            np.random.shuffle(idxs) # shuffle indices

            for idx in idxs:
                key = train_keys[idx]
                dataset = self.dataset[key]
                seq = dataset['features'][...] # sequence of features, (seq_len, dim)
                seq = torch.from_numpy(seq).unsqueeze(0) # input shape (1, seq_len, dim)
                if self.hps.use_cuda: 
                    seq = seq.cuda()
                probs = self.model(seq) # output shape (1, seq_len, 1)

                cost = self.beta * (probs.mean() - 0.5)**2 # minimize summary length penalty term [Eq.11]
                m = Bernoulli(probs)
                epis_rewards = []
                for _ in range(self.num_episode):
                    actions = m.sample()
                    log_probs = m.log_prob(actions)
                    reward = compute_reward(seq, actions, use_gpu=self.hps.use_cuda)
                    expected_reward = log_probs.mean() * (reward - baselines[key])
                    cost -= expected_reward # minimize negative expected reward
                    epis_rewards.append(reward.item())

                self.optimizer.zero_grad()
                cost.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                baselines[key] = 0.9 * baselines[key] + 0.1 * np.mean(epis_rewards) # update baseline reward via moving average
                reward_writers[key].append(np.mean(epis_rewards))

            epoch_reward = np.mean([reward_writers[key][epoch] for key in train_keys])
            self.hps.writer.add_scalar('{}/Fold_{}/Train/Reward'.format(self.dataset_name, fold+1), epoch_reward, epoch)
            self.log.info("epoch {}/{}\t reward {}\t".format(epoch+1, self.hps.epochs_max, epoch_reward))

            # Evaluate performances on test keys
            if epoch % self.hps.test_every_epochs == 0:
                f_score = self.test(fold)
                self.model.train()
                self.hps.writer.add_scalar('{}/Fold_{}/Test/F-score'.format(self.dataset_name, fold+1), f_score, epoch)
                if f_score > best_f_score:
                    best_f_score = f_score
                    self.best_weights = self.model.state_dict()

        return best_f_score

    def test(self, fold):
        self.model.eval()
        _, test_keys = self._get_train_test_keys(fold)
        summary = {}
        with torch.no_grad():
            for key in test_keys:
                seq = self.dataset[key]['features'][...]
                seq = torch.from_numpy(seq).unsqueeze(0)

                if self.hps.use_cuda:
                    seq = seq.float().cuda()

                y = self.model(seq)
                summary[key] = y[0].detach().cpu().numpy()

        f_score = self._eval_summary(summary, test_keys)
        return f_score
    
def compute_reward(seq, actions, ignore_far_sim=True, temp_dist_thre=20, use_gpu=False):
    """
    Compute diversity reward and representativeness reward

    Args:
        seq: sequence of features, shape (1, seq_len, dim)
        actions: binary action sequence, shape (1, seq_len, 1)
        ignore_far_sim (bool): whether to ignore temporally distant similarity (default: True)
        temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
        use_gpu (bool): whether to use GPU
    """
    _seq = seq.detach()
    _actions = actions.detach()
    pick_idxs = _actions.squeeze().nonzero().squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1
    
    if num_picks == 0:
        # give zero reward is no frames are selected
        reward = torch.tensor(0.)
        if use_gpu: reward = reward.cuda()
        return reward

    _seq = _seq.squeeze()
    n = _seq.size(0)

    # compute diversity reward
    if num_picks == 1:
        reward_div = torch.tensor(0.)
        if use_gpu: reward_div = reward_div.cuda()
    else:
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
        dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t()) # dissimilarity matrix [Eq.4]
        dissim_submat = dissim_mat[pick_idxs,:][:,pick_idxs]
        if ignore_far_sim:
            # ignore temporally distant similarity
            pick_mat = pick_idxs.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
        reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.)) # diversity reward [Eq.3]

    # compute representativeness reward
    dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_mat = dist_mat + dist_mat.t()
    dist_mat.addmm_(1, -2, _seq, _seq.t())
    dist_mat = dist_mat[:,pick_idxs]
    dist_mat = dist_mat.min(1, keepdim=True)[0]
    #reward_rep = torch.exp(torch.FloatTensor([-dist_mat.mean()]))[0] # representativeness reward [Eq.5]
    reward_rep = torch.exp(-dist_mat.mean())

    # combine the two rewards
    reward = (reward_div + reward_rep) * 0.5

    return reward



if __name__ == "__main__":
    pass
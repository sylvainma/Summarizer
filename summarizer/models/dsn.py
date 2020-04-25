import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from summarizer.models import Model

"""
Deep Reinforcement Learning for Unsupervised Video Summarization with Diversity-Representativeness Reward
https://arxiv.org/pdf/1801.00054v3.pdf
https://github.com/KaiyangZhou/pytorch-vsumm-reinforce
"""

class DSN(nn.Module):
    """Deep Summarization Network"""
    def __init__(self, in_dim=1024, hid_dim=256, num_layers=1, cell='lstm'):
        super(DSN, self).__init__()
        assert cell in ['lstm', 'gru'], "cell must be either 'lstm' or 'gru'"
        if cell == 'lstm':
            self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.GRU(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(hid_dim * 2, 1),
            nn.Sigmoid())

    def forward(self, x):
        h, _ = self.rnn(x)
        probs = self.out(h)
        return probs


class DSNModel(Model):
    def _init_model(self):
        self.beta = int(self.hps.extra_params.get("beta", 0.01))
        self.num_episodes = int(self.hps.extra_params.get("num_episodes", 5))
        self.eps = float(self.hps.extra_params.get("eps", 0.5))
        self.far_sim = bool(self.hps.extra_params.get("far_sim", False))
        self.temp_dist_thre = int(self.hps.extra_params.get("temp_dist_thre", 20))
        self.sup = bool(self.hps.extra_params.get("sup", False))
        model = DSN()
        return model

    def train(self, fold):
        self.model.train()
        train_keys, _ = self._get_train_test_keys(fold)

        # Model parameters
        self.log.debug("Parameters: {}".format(sum([_.numel() for _ in self.model.parameters()])))
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hps.lr,
            weight_decay=self.hps.l2_req)
        
        # BCE loss for supervised extension
        loss_BCE = nn.BCELoss()
        if self.hps.use_cuda:
            loss_BCE.cuda()

        # Baseline rewards for videos
        baselines = {key: 0. for key in train_keys} 

        # Record reward changes for each video across epochs
        reward_writers = {key: [] for key in train_keys}

        # To record performances of the best epoch
        best_corr, best_f_score = 0.0, 0.0
        
        # For each epoch
        for epoch in range(self.hps.epochs):
            epoch_avg_loss = []
            random.shuffle(train_keys)

            # For each training video
            for batch_i, key in enumerate(train_keys):
                dataset = self.dataset[key]
                seq = dataset['features'][...]
                seq = torch.from_numpy(seq).unsqueeze(1) # (seq_len, 1, dim)
                y = dataset["gtscore"][...]
                y = torch.from_numpy(y).view(-1, 1, 1) # (seq_len, 1, 1)

                # Normalize frame scores
                y -= y.min()
                y /= y.max()

                if self.hps.use_cuda: 
                    seq, y = seq.cuda(), y.cuda()
                
                # Score probabilities from the RNN
                probs = self.model(seq)
                dist = Bernoulli(probs)

                # Regularization: summary length penalty term [Eq.11]
                loss = self.beta * (probs.mean() - self.eps) ** 2

                # Extension to supervised learning (neg-MLE <=> BCE)
                if self.sup:
                    loss += loss_BCE(probs, y)
                
                # Run episodes
                epis_rewards = []
                for _ in range(self.num_episodes):
                    # Sample actions using the distribution
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions)

                    # Compute reward of current episode
                    reward = self.compute_reward(seq, actions, 
                        far_sim=self.far_sim,
                        temp_dist_thre=self.temp_dist_thre)
                    
                    # Negative policy gradient [Eq.10] of current episode
                    loss -= log_probs.mean() * (reward - baselines[key])

                    # Record rewards over the episodes
                    epis_rewards.append(reward.item())

                # Average the loss over the episodes
                loss /= float(self.num_episodes)

                # Update model's parameters
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

                # Update baseline reward via moving average
                baselines[key] = 0.9 * baselines[key] + 0.1 * np.mean(epis_rewards) 
                
                # Record average reward of the current video of the current epoch
                reward_writers[key].append(np.mean(epis_rewards))

                # Record loss
                epoch_avg_loss.append(float(loss)) 

            # Log average reward and loss by the end of the epoch
            epoch_avg_reward = np.mean([reward_writers[key][epoch] for key in train_keys])
            epoch_avg_loss = np.mean(epoch_avg_loss)
            self.hps.writer.add_scalar('{}/Fold_{}/Train/Reward'.format(self.dataset_name, fold+1), epoch_avg_reward, epoch)
            self.hps.writer.add_scalar('{}/Fold_{}/Train/Loss'.format(self.dataset_name, fold+1), epoch_avg_loss, epoch)
            self.log.info("Epoch: {:6}   Reward: {:.05f}   Loss: {:.05f}".format(
                str(epoch+1)+"/"+str(self.hps.epochs), epoch_avg_reward, epoch_avg_loss))

            # Evaluate performances on test keys
            if epoch % self.hps.test_every_epochs == 0:
                corr, f_score = self.test(fold)
                self.model.train()
                self.hps.writer.add_scalar('{}/Fold_{}/Test/Correlation'.format(self.dataset_name, fold+1), corr, epoch)
                self.hps.writer.add_scalar('{}/Fold_{}/Test/F-score'.format(self.dataset_name, fold+1), f_score, epoch)
                if f_score > best_f_score:
                    best_f_score = f_score
                if corr > best_corr:
                    best_corr = corr
                    self.best_weights = self.model.state_dict()

        return best_corr, best_f_score
    
    def compute_reward(self, seq, actions, far_sim=False, temp_dist_thre=20):
        """Compute diversity reward and representativeness reward
        Args:
            seq: sequence of features, shape (seq_len, 1, dim)
            actions: binary action sequence, shape (seq_len, 1, 1)
            far_sim (bool): whether to use temporally distant similarity (default: False)
            temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
        """
        _seq = seq.detach()
        _actions = actions.detach()
        pick_idxs = _actions.squeeze().nonzero().squeeze()
        num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1
        
        # Give zero reward if no frames are selected
        if num_picks == 0:
            reward = torch.tensor(0.)
            if self.hps.use_cuda:
                reward = reward.cuda()
            return reward

        _seq = _seq.squeeze()
        T = _seq.size(0)

        # Compute diversity reward [Eq.3]
        if num_picks == 1:
            reward_div = torch.tensor(0.)
            if self.hps.use_cuda:
                reward_div = reward_div.cuda()
        else:
            # Dissimilarity function (cosine dissimilarity) [Eq.4]
            normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
            dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t())
            dissim_submat = dissim_mat[pick_idxs,:][:,pick_idxs]
            if not far_sim:
                # Ignore temporally distant similarity
                pick_mat = pick_idxs.expand(num_picks, num_picks)
                temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
                dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
            reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.))

        # Compute representativeness reward [Eq.5]
        dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(T, T) # (T, T)
        dist_mat = dist_mat + dist_mat.t()
        dist_mat.addmm_(1, -2, _seq, _seq.t())
        dist_mat = dist_mat[:,pick_idxs]
        dist_mat = dist_mat.min(1, keepdim=True)[0]
        reward_rep = torch.exp(-dist_mat.mean())

        # Combine the two rewards
        reward = (reward_div + reward_rep) * 0.5

        return reward


if __name__ == "__main__":
    pass
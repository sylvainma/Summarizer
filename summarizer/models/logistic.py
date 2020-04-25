import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from summarizer.models import Model

"""
Simple Logistic Regression.
"""

class LogisticRegression(nn.Module):
    def __init__(self, input_dim=1024):
        super(LogisticRegression, self).__init__()
        self.input_dim = input_dim
        self.perceptron = nn.Linear(input_dim, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, self.input_dim) # (1,D,T) => (T,D)
        x = self.perceptron(x)
        y = self.sig(x)
        y = y.view(1, -1) # (T,1) => (1,T)
        return y


class LogisticRegressionModel(Model):
    def _init_model(self):
        model = LogisticRegression()
        return model

    def train(self, fold):
        self.model.train()
        train_keys, _ = self._get_train_test_keys(fold)

        criterion = nn.MSELoss()
        if self.hps.use_cuda:
            criterion = criterion.cuda()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hps.lr,
            weight_decay=self.hps.l2_req)

        # To record performances of the best epoch
        best_corr, best_f_score = 0.0, 0.0

        # For each epoch
        for epoch in range(self.hps.epochs):
            train_avg_loss = []
            random.shuffle(train_keys)

            # For each training video
            for key in train_keys:
                dataset = self.dataset[key]
                seq = dataset['features'][...]
                seq = torch.from_numpy(seq).unsqueeze(0)
                target = dataset['gtscore'][...]
                target = torch.from_numpy(target).unsqueeze(0)

                # Normalize frame scores
                target -= target.min()
                target /= target.max()

                if self.hps.use_cuda:
                    seq, target = seq.cuda(), target.cuda()

                y = self.model(seq)

                loss = criterion(y, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_avg_loss.append(float(loss))

            # Average training loss value of epoch
            train_avg_loss = np.mean(np.array(train_avg_loss))
            self.log.info("Epoch: {0:6}    Train loss: {1:.05f}".format(
                str(epoch+1)+"/"+str(self.hps.epochs), train_avg_loss))
            self.hps.writer.add_scalar('{}/Fold_{}/Train/Loss'.format(self.dataset_name, fold+1), train_avg_loss, epoch)

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


if __name__ == "__main__":
    D, T = 1024, 300
    seq = torch.rand(1, D, T)
    target = torch.rand(T)
    model = LogisticRegression()
    y = model(seq)
    assert y.shape[1] == T, f"{y.shape} wrong shape"
    print((y - target).mean().item())

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import Model

"""
Logistic Regression as a baseline.
"""

class LogisticRegression(nn.Module):
    def __init__(self, input_dim=1024):
        """Baseline model for Video Summarization"""
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
    
    def train(self):
        self.model.train()
        train_keys = self.split["train_keys"][:]

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

            print("Epoch: {0:6}".format(str(epoch+1)+"/"+str(self.hps.epochs_max)), end='')
            train_avg_loss = []
            random.shuffle(train_keys)

            # For each training video
            for i, key in enumerate(train_keys):
                dataset = self.dataset[key]
                seq = dataset['features'][...]
                seq = torch.from_numpy(seq).unsqueeze(0)
                target = dataset['gtscore'][...]
                target = torch.from_numpy(target).unsqueeze(0)

                # Normalize frame scores
                target -= target.min()
                target /= target.max()

                if self.hps.use_cuda:
                    seq, target = seq.float().cuda(), target.float().cuda()

                y = self.model(seq)

                loss = criterion(y, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_avg_loss.append(float(loss))

            # Average training loss value of epoch
            train_avg_loss = np.mean(np.array(train_avg_loss))
            print("   Train loss: {0:.05f}".format(train_avg_loss, end=''))

            # Evaluate performances on test keys
            if epoch % self.hps.test_every_epochs == 0 or epoch == 0:
                f_score = self.test()
                self.model.train()
                if f_score > best_f_score:
                    best_f_score = f_score
                    self.best_weights = self.model.state_dict()

        return best_f_score

    def test(self):
        self.model.eval()
        test_keys = self.split["test_keys"][:]
        summary = {}
        with torch.no_grad():
            for i, key in enumerate(test_keys):
                seq = self.dataset[key]['features'][...]
                seq = torch.from_numpy(seq).unsqueeze(0)

                if self.hps.use_cuda:
                    seq = seq.float().cuda()

                y = self.model(seq)
                summary[key] = y[0].detach().cpu().numpy()

        f_score = self._eval_summary(summary, test_keys)
        return f_score


if __name__ == "__main__":
    D, T = 1024, 300
    seq = torch.rand(1, D, T)
    target = torch.rand(T)
    model = LogisticRegression()
    y = model(seq)
    assert y.shape[1] == T, f"{y.shape} wrong shape"
    print((y - target).mean().item())

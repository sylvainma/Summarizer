import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from summarizer.models import Model

"""
Unsupervised Video Summarization with Adversarial LSTM Networks
http://web.engr.oregonstate.edu/~sinisa/research/publications/cvpr17_summarization.pdf
"""

class sLSTM(nn.Module):
    def __init__(self, input_size=1024, hidden_size=1024, num_layers=2):
        """Selector LSTM"""
        super(sLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            bidirectional=True
        )
        self.out = nn.Linear(hidden_size * 2, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """
        Input 
          x: (seq_len, batch_size, input_size)
        Output
          scores: (seq_len, batch_size, 1)
        """
        scores, _ = self.lstm(x)
        scores = self.out(scores)
        scores = self.sig(scores)
        return scores

class eLSTM(nn.Module):
    def __init__(self, input_size=1024, hidden_size=2048, num_layers=2):
        """Encoder LSTM"""
        super(eLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False
        )
        self.mu = nn.Linear(hidden_size, hidden_size)
        self.logvar = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        """
        Input 
          x: (seq_len, batch_size, input_size)
        Output
          h_mu, h_logvar: (num_layers, batch_size, hidden_size)
          c_last: (num_layers, batch_size, hidden_size)
        """
        _, (h_last, c_last) = self.lstm(x)
        h_mu = self.mu(h_last)
        h_logvar = self.logvar(h_last)
        return (h_mu, h_logvar), c_last

class dLSTM(nn.Module):
    def __init__(self, input_size=1024, hidden_size=2048, num_layers=2):
        """Decoder LSTM"""
        super(dLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False
        )
        self.recons = nn.Linear(hidden_size, input_size)
        
    def forward_step(self, x_prev, h_prev, c_prev):
        """Decode one sequence step.
        Input
          x_prev: (1, batch_size, hidden_size)
          h_prev, c_prev: (num_layers, batch_size, hidden_size)
        Output
          x_next: (1, batch_size, hidden_size)
          h_next, c_next: (num_layers, batch_size, hidden_size)
        """
        x_next, (h_next, c_next) = self.lstm(x_prev, (h_prev, c_prev))
        return x_next, (h_next, c_next)

    def forward(self, seq_len, h_0, c_0):
        """Decode entire sequence.
        Input 
          seq_len: (1,)
          h_0, c_0: (num_layers, batch_size, hidden_size)
        Output
          x_hat: (1, batch_size, input_size)
        """
        batch_size, hidden_size = h_0.size(1), h_0.size(2)
        x = torch.zeros(1, batch_size, hidden_size).cuda()
        h, c = h_0, c_0
        x_hat = []
        for i in range(seq_len):
            x, (h, c) = self.forward_step(x, h, c)
            x_hat.append(self.recons(x))
        x_hat = torch.cat(x_hat, dim=0) # TODO: reverse according to paper
        return x_hat

class VAE(nn.Module):
    def __init__(self, input_size=1024, hidden_size=2048, num_layers=2):
        """Variational Auto Encoder LSTM"""
        super(VAE, self).__init__()
        
        self.e_lstm = eLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers)
        
        self.d_lstm = dLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Input 
          x: (seq_len, batch_size, input_size)
        Output
          x_hat: (seq_len, batch_size, input_size)
          h_mu, h_logvar: (num_layers, batch_size, hidden_size)
        """
        (h_mu, h_logvar), c = self.e_lstm(x)
        h = self.reparameterize(h_mu, h_logvar)
        x_hat = self.d_lstm(x.size(0), h, c)
        return x_hat, (h_mu, h_logvar)

class Summarizer(nn.Module):
    def __init__(self, input_size=1024, hidden_size=2048, num_layers=2):
        """Summarizer: Selector + VAE"""
        super(Summarizer, self).__init__()
        self.s_lstm = sLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.vae = VAE(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, x):
        """
        Input 
          x: (seq_len, batch_size, input_size)
        Output
          x_hat: (seq_len, batch_size, input_size)
          h_mu, h_logvar: (num_layers, batch_size, hidden_size)
          scores: (seq_len, batch_size, 1)
        """
        scores = self.s_lstm(x)
        x_weighted = x * scores
        x_hat, (h_mu, h_logvar) = self.vae(x_weighted)
        return x_hat, (h_mu, h_logvar), scores

class SumGANModel(Model):
    def _init_model(self):
        model = Summarizer()
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
            for key in train_keys:
                dataset = self.dataset[key]
                seq = dataset['features'][...]
                seq = torch.from_numpy(seq).unsqueeze(1) # (seq_len, 1, n_features)
                target = dataset['gtscore'][...]
                target = torch.from_numpy(target).unsqueeze(1) # (seq_len, 1, 1)

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
            for key in test_keys:
                seq = self.dataset[key]['features'][...]
                seq = torch.from_numpy(seq).unsqueeze(0)

                if self.hps.use_cuda:
                    seq = seq.float().cuda()

                y = self.model(seq)
                summary[key] = y[0].detach().cpu().numpy()

        f_score = self._eval_summary(summary, test_keys)
        return f_score


if __name__ == "__main__":
    model = Summarizer().cuda()
    x = torch.randn(10, 3, 1024).cuda()
    x_hat, (mu, logvar), scores = model(x)
    print(x.shape, x_hat.shape, mu.shape, logvar.shape, scores.shape)
    assert x.shape[0] == scores.shape[0]
    assert x.shape[1] == scores.shape[1]
    assert scores.shape[2] == 1
    assert mu.shape[0] == logvar.shape[0]
    assert mu.shape[2] == logvar.shape[2]
    assert x.shape[0] == x_hat.shape[0]
    assert x.shape[1] == x_hat.shape[1]
    assert x.shape[2] == x_hat.shape[2]


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
Unsupervised Video Summarization with Adversarial LSTM Networks
http://web.engr.oregonstate.edu/~sinisa/research/publications/cvpr17_summarization.pdf
https://github.com/j-min/Adversarial_Video_Summary
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
        x = torch.zeros(1, batch_size, hidden_size).to(h_0.device)
        h, c = h_0, c_0
        x_hat = []
        for i in range(seq_len):
            x, (h, c) = self.forward_step(x, h, c)
            x_hat.append(self.recons(x))
        x_hat = torch.cat(x_hat, dim=0)
        x_hat = torch.flip(x_hat, (0,)) # reverse
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
    def __init__(self, input_size=1024, sLSTM_hidden_size=1024, sLSTM_num_layers=2,
                     edLSTM_hidden_size=2048, edLSTM_num_layers=2):
        """Summarizer: Selector (sLSTM) + VAE (eLSTM/dLSTM).
        Args
          input_size: size of the frame feature descriptor
          sLSTM_hidden_size: hidden size of sLSTM
          sLSTM_num_layers: number of layers of sLSTM
          edLSTM_hidden_size: hidden size of eLSTM and dLSTM
          edLSTM_num_layers: number of layers of eLSTM and dLSTM
        """
        super(Summarizer, self).__init__()
        self.s_lstm = sLSTM(input_size=input_size, hidden_size=sLSTM_hidden_size, num_layers=sLSTM_num_layers)
        self.vae = VAE(input_size=input_size, hidden_size=edLSTM_hidden_size, num_layers=edLSTM_num_layers)

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

class cLSTM(nn.Module):
    def __init__(self, input_size=1024, hidden_size=1024, num_layers=2):
        """Discriminator as a classifier LSTM"""
        super(cLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid())

    def forward(self, x):
        """
        Input 
          x: (seq_len, batch_size, input_size)
        Output
          probs: (batch_size, 1)
          h_last_top: (batch_size, hidden_size)
        """
        _, (h_last, _) = self.lstm(x) # (num_layers*2, batch_size, hidden_size)
        h_last_top = h_last[-1]       # (batch_size, hidden_size)
        probs = self.out(h_last_top)  # (batch_size, 1)
        return probs, h_last_top

class GAN(nn.Module):
    def __init__(self, input_size=1024, hidden_size=1024, num_layers=2):
        """GAN: discriminator.
        Args
          input_size: size of the frame feature descriptor
          hidden_size: hidden size of cLSTM
          num_layers: number of layers of cLSTM
        """
        super(GAN, self).__init__()
        self.c_lstm = cLSTM(
          input_size=input_size,
          hidden_size=hidden_size,
          num_layers=num_layers)

    def forward(self, x):
        """
        Input
          x: (seq_len, batch_size, input_size)
        Output
          probs: (batch_size, 1)
          h_last_top: (batch_size, hidden_size)
        """
        probs, h_last_top = self.c_lstm(x)
        return probs, h_last_top


class SumGANModel(Model):
    def _init_model(self):
        # SumGAN hyperparameters
        self.clip = float(self.hps.extra_params.get("clip", 5.0))
        self.sigma = float(self.hps.extra_params.get("sigma", 0.3))
        self.input_size = int(self.hps.extra_params.get("input_size", 1024))
        self.sLSTM_hidden_size = int(self.hps.extra_params.get("sLSTM_hidden_size", 1024))
        self.sLSTM_num_layers = int(self.hps.extra_params.get("sLSTM_num_layers", 2))
        self.edLSTM_hidden_size = int(self.hps.extra_params.get("edLSTM_hidden_size", 2048))
        self.edLSTM_num_layers = int(self.hps.extra_params.get("edLSTM_num_layers", 2))
        self.cLSTM_hidden_size = int(self.hps.extra_params.get("cLSTM_hidden_size", 1024))
        self.cLSTM_num_layers = int(self.hps.extra_params.get("cLSTM_num_layers", 2))

        # Model
        self.summarizer = Summarizer(
            input_size=self.input_size,
            sLSTM_hidden_size=self.sLSTM_hidden_size, sLSTM_num_layers=self.sLSTM_num_layers,
            edLSTM_hidden_size=self.edLSTM_hidden_size, edLSTM_num_layers=self.edLSTM_num_layers)
        self.gan = GAN(
            input_size=self.input_size, 
            hidden_size=self.cLSTM_hidden_size, num_layers=self.cLSTM_num_layers)
        model = nn.ModuleList([self.summarizer, self.gan])
        
        print("SumGAN parameters:", sum([_.numel() for _ in model.parameters()]))
        return model

    def loss_recons(self, h_real, h_fake):
        """minimize E[l2_norm(phi(x) - phi(x_hat))]"""
        return torch.norm(h_real - h_fake, p=2)

    def loss_prior(self, mu, logvar):
        """minimize -D_KL(q(e|x) || p(e))"""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def loss_sparsity(self, scores, sigma):
        """minimize l2_norm(E[s_t] - sigma)"""
        return torch.abs(torch.mean(scores) - sigma)

    def loss_gan_generator(self, probs_fake):
        """maximize E[log(cLSTM(x_hat))]"""
        label_real = torch.full_like(probs_fake, 1.0).to(probs_fake.device)
        return self.loss_BCE(probs_fake, label_real)

    def loss_gan_discriminator(self, probs_real, probs_fake):
        """maximize E[log(cLSTM(x))] + E[log(1 - cLSTM(x_hat))]"""
        label_real = torch.full_like(probs_real, 1.0).to(probs_real.device)
        label_fake = torch.full_like(probs_fake, 0.0).to(probs_fake.device)
        return self.loss_BCE(probs_real, label_real) + self.loss_BCE(probs_fake, label_fake)

    def train(self, fold):
        self.model.train()
        train_keys, _ = self._get_train_test_keys(fold)
        
        # Optimization
        self.s_e_optimizer = torch.optim.Adam(
            list(self.summarizer.s_lstm.parameters())
            + list(self.summarizer.vae.e_lstm.parameters()),
            lr=self.hps.lr,
            weight_decay=self.hps.l2_req)
        self.d_optimizer = torch.optim.Adam(
            self.summarizer.vae.d_lstm.parameters(),
            lr=self.hps.lr,
            weight_decay=self.hps.l2_req)
        self.c_optimizer = torch.optim.Adam(
            self.gan.c_lstm.parameters(),
            lr=self.hps.lr,
            weight_decay=self.hps.l2_req)

        # BCE loss for GAN optimization
        self.loss_BCE = nn.BCELoss()

        # To record performances of the best epoch
        best_f_score = 0.0

        # For each epoch
        for epoch in range(self.hps.epochs_max):

            print("Epoch: {0:6}".format(str(epoch+1)+"/"+str(self.hps.epochs_max)), end='')
            train_avg_D_x = []
            train_avg_D_x_hat = []
            random.shuffle(train_keys)

            # For each training video
            for batch_i, key in enumerate(train_keys):
                dataset = self.dataset[key]
                x = dataset['features'][...]
                x = torch.from_numpy(x).unsqueeze(1) # (seq_len, 1, n_features)
                y = dataset['gtscore'][...]
                y = torch.from_numpy(y).unsqueeze(1) # (seq_len, 1, 1)

                # Normalize frame scores
                y -= y.min()
                y /= y.max()

                if self.hps.use_cuda:
                    x, y = x.float().cuda(), y.float().cuda()

                ###############################
                # Selector and Encoder update
                ###############################
                # Forward
                x_hat, (mu, logvar), scores = self.summarizer(x)
                probs_real, h_real = self.gan(x)
                probs_fake, h_fake = self.gan(x_hat)

                # Losses
                loss_recons = self.loss_recons(h_real, h_fake)
                loss_prior = self.loss_prior(mu, logvar)
                loss_sparsity = self.loss_sparsity(scores, self.sigma)
                loss_s_e = loss_recons + loss_prior + loss_sparsity

                # Update
                loss_s_e.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.s_e_optimizer.step()
                self.s_e_optimizer.zero_grad()

                ###############################
                # Decoder update
                ###############################
                # Forward
                x_hat, (mu, logvar), scores = self.summarizer(x)
                _, h_real = self.gan(x)
                probs_fake, h_fake = self.gan(x_hat)

                # Losses
                loss_recons = self.loss_recons(h_real, h_fake)
                loss_gan = self.loss_gan_generator(probs_fake)
                loss_d = loss_recons + loss_gan

                # Update
                loss_d.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.d_optimizer.step()
                self.d_optimizer.zero_grad()

                ###############################
                # Discriminator update
                ###############################
                # Forward
                x_hat, (mu, logvar), scores = self.summarizer(x)
                probs_real, h_real = self.gan(x)
                probs_fake, h_fake = self.gan(x_hat)

                # Losses
                loss_c = self.loss_gan_discriminator(probs_real, probs_fake)

                # Update
                loss_c.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.c_optimizer.step()
                self.c_optimizer.zero_grad()

                ###############################
                # Record losses
                ###############################
                train_avg_D_x.append(torch.mean(probs_real).detach().cpu().numpy())
                train_avg_D_x_hat.append(torch.mean(probs_fake).detach().cpu().numpy())

            # Average probs for real and fake data
            print("   D(x): {0:.05f}   D(x_hat): {1:.05f}".format(
              np.mean(train_avg_D_x), np.mean(train_avg_D_x_hat),
            end=''))

            # Evaluate performances on test keys
            if epoch % self.hps.test_every_epochs == 0 or epoch == 0:
                f_score = self.test(fold)
                self.model.train()
                if f_score > best_f_score:
                    best_f_score = f_score
                    self.best_weights = self.model.state_dict()

            # Free unused memory from GPU
            torch.cuda.empty_cache()

        return best_f_score

    def test(self, fold):
        self.model.eval()
        _, test_keys = self._get_train_test_keys(fold)
        summary = {}
        with torch.no_grad():
            for key in test_keys:
                x = self.dataset[key]['features'][...]
                x = torch.from_numpy(x).unsqueeze(0)

                if self.hps.use_cuda:
                    x = x.float().cuda()

                _, _, scores = self.summarizer(x)
                summary[key] = scores[0].detach().cpu().numpy()

        f_score = self._eval_summary(summary, test_keys)
        return f_score


if __name__ == "__main__":
    model = nn.ModuleList([Summarizer(), GAN()])
    print("Parameters:", sum([_.numel() for _ in model.parameters()]))

    model = Summarizer()
    x = torch.randn(10, 3, 1024)
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

    model = GAN()
    x = torch.randn(10, 3, 1024)
    probs, h = model(x)
    print(x.shape, probs.shape)
    assert x.shape[1] == probs.shape[0]
    assert probs.shape[1] == 1


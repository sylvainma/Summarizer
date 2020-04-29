import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from summarizer.models import Model
from summarizer.models.sumgan import GAN

"""
Upgraded version of SumGAN with Attention.
* Transformer instead of sLSTM
* Transformer-based autoencoder instead of lstm VAE
* Add noise to inputs of the discriminator (cLSTM)
* Use Wasserstein loss (WGAN)
"""

class Transformer(nn.Module):
    def __init__(self, input_size=1024, encoder_layers=4, attention_heads=8, epsilon=1e-5):
        super(Transformer, self).__init__()
        self.input_size = input_size
        self.layer_norm = torch.nn.LayerNorm(input_size, epsilon)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=attention_heads,
            dim_feedforward=input_size)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.transformer_encoder_layer, 
            num_layers=encoder_layers, 
            norm=self.layer_norm)
        self.out = nn.Sequential(
            nn.Linear(input_size, 1),
            nn.Sigmoid())

    def forward(self, x):
        """Pass through tranformer and a final sigmoid layer.
        Input
          x: (seq_len, batch_size, input_size)
        Output
          scores: (seq_len, batch_size, 1)
        """
        encoder_out = self.transformer_encoder.forward(x)
        scores = self.out(encoder_out)
        return scores

class AutoencoderTransformer(nn.Module):
    def __init__(self, input_size=1024, encoder_layers=4, attention_heads=8, epsilon=1e-5):
        super(AutoencoderTransformer, self).__init__()
        self.input_size = input_size
        
        # Encoder
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=attention_heads,
            dim_feedforward=input_size)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.transformer_encoder_layer, 
            num_layers=encoder_layers)

        # Decoder
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_size,
            nhead=attention_heads,
            dim_feedforward=input_size)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=self.transformer_decoder_layer, 
            num_layers=encoder_layers)
        
    def forward(self, x):
        """Pass through encoder-decoder transformer.
        Input
          x: (seq_len, batch_size, input_size)
        Output
          x_hat: (seq_len, batch_size, input_size)
        """
        encoder_out = self.transformer_encoder(x)
        x_hat = self.transformer_decoder(x, encoder_out)
        return x_hat

class Summarizer(nn.Module):
    def __init__(self, input_size=1024, s_encoder_layers=2, s_attention_heads=4,
                     ae_encoder_layers=2, ae_attention_heads=4):
        """Summarizer: Selector (Transformer) + Autoencoder Transformer.
        Args
          input_size: size of the frame feature descriptor
          s_encoder_layers: selector number of layers
          s_attention_heads: selector number of heads
          ae_encoder_layers: autoencoder number of layers
          ae_attention_heads: autoencoder number of heads
        """
        super(Summarizer, self).__init__()
        self.selector = Transformer(
            input_size=input_size,
            encoder_layers=s_encoder_layers,
            attention_heads=s_attention_heads)
        self.ae = AutoencoderTransformer(
            input_size=input_size,
            encoder_layers=ae_encoder_layers,
            attention_heads=ae_attention_heads)

    def forward(self, x, uniform=False, p=0.3):
        """
        Input 
          x: (seq_len, batch_size, input_size)
        Output
          x_hat: (seq_len, batch_size, input_size)
          scores: (seq_len, batch_size, 1)
        """
        if uniform:
            seq_len, batch_size, _ = x.size()
            dist = Bernoulli(torch.full((seq_len, batch_size, 1), p).to(x.device))
            scores = dist.sample()
        else:
            scores = self.selector(x)
        
        x_weighted = x * scores
        x_hat = self.ae(x_weighted)
        return x_hat, scores

class SumGANAtt(nn.Module):
    def __init__(self, input_size=1024, s_encoder_layers=2, s_attention_heads=4, 
                        ae_encoder_layers=2, ae_attention_heads=4,
                        cLSTM_hidden_size=1024, cLSTM_num_layers=2):
        """SumGAN: Summarizer + GAN"""
        super(SumGANAtt, self).__init__()
        self.summarizer = Summarizer(
            input_size=input_size,
            s_encoder_layers=s_encoder_layers, s_attention_heads=s_attention_heads,
            ae_encoder_layers=ae_encoder_layers, ae_attention_heads=ae_attention_heads)
        self.gan = GAN(
            input_size=input_size,
            hidden_size=cLSTM_hidden_size, num_layers=cLSTM_num_layers)
    
    def forward(self, x):
        """
        Input
          x: (seq_len, batch_size, input_size)
        Output
          scores: (seq_len, batch_size, 1)
        """
        return self.summarizer.selector(x)


class SumGANAttModel(Model):
    def _init_model(self):
        # SumGAN hyperparameters
        self.sigma = float(self.hps.extra_params.get("sigma", 0.3))
        self.input_size = int(self.hps.extra_params.get("input_size", 1024))
        self.s_encoder_layers = int(self.hps.extra_params.get("s_encoder_layers", 2))
        self.s_attention_heads = int(self.hps.extra_params.get("s_attention_heads", 4))
        self.ae_encoder_layers = int(self.hps.extra_params.get("ae_encoder_layers", 2))
        self.ae_attention_heads = int(self.hps.extra_params.get("ae_attention_heads", 4))
        self.cLSTM_hidden_size = int(self.hps.extra_params.get("cLSTM_hidden_size", 1024))
        self.cLSTM_num_layers = int(self.hps.extra_params.get("cLSTM_num_layers", 2))
        self.sup = bool(self.hps.extra_params.get("sup", False))
        self.pretrain_ae = int(self.hps.extra_params.get("pretrain_ae", 100))
        self.epoch_noise = int(self.hps.extra_params.get("epoch_noise", 0.2*self.hps.epochs))

        # Model
        model = SumGANAtt(input_size=self.input_size,
            s_encoder_layers=self.s_encoder_layers, s_attention_heads=self.s_attention_heads,
            ae_encoder_layers=self.ae_encoder_layers, ae_attention_heads=self.ae_attention_heads,
            cLSTM_hidden_size=self.cLSTM_hidden_size, cLSTM_num_layers=self.cLSTM_num_layers)
        
        return model

    def loss_ae(self, x, x_hat):
        """minimize E[l2_norm(x - x_hat)]"""
        return torch.norm(x - x_hat, p=2)
    
    def loss_recons(self, h_real, h_fake):
        """minimize E[l2_norm(phi(x) - phi(x_hat))]"""
        return torch.norm(h_real - h_fake, p=2)

    def loss_sparsity(self, scores, sigma):
        """minimize l2_norm(E[s_t] - sigma)"""
        return torch.abs(torch.mean(scores) - sigma)

    def loss_sparsity_sup(self, scores, gtscores):
        """minimize BCE(scores, gtscores)"""
        return self.loss_BCE(scores, gtscores)

    def loss_gan_generator(self, probs_fake, probs_uniform):
        """maximize 0.5 * (cLSTM(x_hat) + cLSTM(x_hat_p))"""
        return torch.mean(-0.5 * (probs_fake + probs_uniform))

    def loss_gan_discriminator(self, probs_real, probs_fake, probs_uniform):
        """maximize cLSTM(x) - 0.5 * (cLSTM(x_hat) + cLSTM(x_hat_p))"""
        return torch.mean(-probs_real + 0.5 * (probs_fake + probs_uniform))

    def pretrain(self, fold):
        """Pretrain VAE before learning the GAN, as recommended in paper"""
        train_keys, _ = self._get_train_test_keys(fold)
        ae_optimizer = torch.optim.Adam(
            self.model.summarizer.ae.parameters(),
            lr=self.hps.lr,
            weight_decay=self.hps.l2_req)

        for epoch in range(self.pretrain_ae):
            train_avg_loss_ae = []
            random.shuffle(train_keys)

            for key in train_keys:
                dataset = self.dataset[key]
                x = dataset["features"][...]
                x = torch.from_numpy(x).unsqueeze(1) # (seq_len, 1, n_features)

                if self.hps.use_cuda:
                    x = x.cuda()
                
                # Pretrain the lstm VAE
                x_hat = self.model.summarizer.ae(x)
                loss_ae = self.loss_ae(x, x_hat)
                ae_optimizer.zero_grad()
                loss_ae.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                ae_optimizer.step()
                train_avg_loss_ae.append(float(loss_ae))

            # Log VAE loss
            if epoch % 10 == 0:
                train_avg_loss_ae = np.mean(train_avg_loss_ae)
                self.log.info(f"Pretrain: {epoch+1:3}/{self.pretrain_ae:3}   Lae: {train_avg_loss_ae:.05f}")

            # Free unused memory from GPU
            torch.cuda.empty_cache()

    def train(self, fold):
        self.model.train()
        train_keys, _ = self._get_train_test_keys(fold)

        # Pretrain VAE
        if self.pretrain_ae > 0:
            self.pretrain(fold)
        
        # Optimization
        self.s_e_optimizer = torch.optim.Adam(
            list(self.model.summarizer.selector.parameters())
            + list(self.model.summarizer.ae.transformer_encoder.parameters())
            + list(self.model.summarizer.ae.transformer_encoder_layer.parameters()),
            lr=self.hps.lr,
            weight_decay=self.hps.l2_req)
        self.d_optimizer = torch.optim.Adam(
            list(self.model.summarizer.ae.transformer_decoder.parameters())
            + list(self.model.summarizer.ae.transformer_decoder_layer.parameters()),
            lr=self.hps.lr,
            weight_decay=self.hps.l2_req)
        self.c_optimizer = torch.optim.SGD(
            self.model.gan.c_lstm.parameters(),
            lr=self.hps.lr,
            weight_decay=self.hps.l2_req)

        # BCE loss for GAN optimization
        self.loss_BCE = nn.BCELoss()
        if self.hps.use_cuda:
            self.loss_BCE.cuda()

        # To record performances of the best epoch
        best_corr, best_avg_f_score, best_max_f_score = -1.0, 0.0, 0.0

        # For each epoch
        for epoch in range(self.hps.epochs):
            train_avg_loss_s_e = []
            train_avg_loss_d = []
            train_avg_loss_c = []
            train_avg_D_x = []
            train_avg_D_x_hat = []
            train_avg_D_x_hat_p = []
            dist_gtscore = []
            dist_scores = []
            dist_scores_uniform = []
            random.shuffle(train_keys)

            # For each training video
            for batch_i, key in enumerate(train_keys):
                dataset = self.dataset[key]
                x = dataset["features"][...]
                x = torch.from_numpy(x).unsqueeze(1) # (seq_len, 1, n_features)
                y = dataset["gtscore"][...]
                y = torch.from_numpy(y).view(-1, 1, 1) # (seq_len, 1, 1)

                # Normalize frame scores
                y -= y.min()
                y /= y.max()

                if self.hps.use_cuda:
                    x, y = x.cuda(), y.cuda()
                
                ###############################
                # Selector and Encoder update
                ###############################
                # Forward
                x_hat, scores = self.model.summarizer(x)
                _, h_real = self.model.gan(x)
                _, h_fake = self.model.gan(x_hat)

                # Losses
                loss_recons = self.loss_recons(h_real, h_fake)
                if self.sup:
                    loss_sparsity = self.loss_sparsity_sup(scores, y)
                else:
                    loss_sparsity = self.loss_sparsity(scores, self.sigma)
                loss_s_e = loss_recons + loss_sparsity

                # Update
                self.s_e_optimizer.zero_grad()
                loss_s_e.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.s_e_optimizer.step()

                ###############################
                # Decoder update
                ###############################
                # Forward
                x_hat, _ = self.model.summarizer(x)
                x_hat_p, _ = self.model.summarizer(x, uniform=True, p=self.sigma)
                _, h_real = self.model.gan(x)
                probs_fake, h_fake = self.model.gan(x_hat)
                probs_uniform, _ = self.model.gan(x_hat_p)

                # Losses
                loss_recons = self.loss_recons(h_real, h_fake)
                loss_gan = self.loss_gan_generator(probs_fake, probs_uniform)
                loss_d = loss_recons + loss_gan

                # Update
                self.d_optimizer.zero_grad()
                loss_d.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.d_optimizer.step()

                ###############################
                # Discriminator update
                ###############################
                # Forward
                x_hat, _ = self.model.summarizer(x)
                x_hat_p, _ = self.model.summarizer(x, uniform=True, p=self.sigma)
                if epoch < self.epoch_noise:
                    x = torch.randn_like(x) * x
                    x_hat = x_hat * torch.randn_like(x_hat)
                    x_hat_p = x_hat_p * torch.randn_like(x_hat_p)
                probs_real, _ = self.model.gan(x)
                probs_fake, _ = self.model.gan(x_hat)
                probs_uniform, _ = self.model.gan(x_hat_p)

                # Losses
                loss_c = self.loss_gan_discriminator(probs_real, probs_fake, probs_uniform)

                # Update
                self.c_optimizer.zero_grad()
                loss_c.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.c_optimizer.step()

                ###############################
                # Record losses
                ###############################
                train_avg_loss_s_e.append(float(loss_s_e))
                train_avg_loss_d.append(float(loss_d))
                train_avg_loss_c.append(float(loss_c))
                train_avg_D_x.append(torch.mean(probs_real).detach().cpu().numpy())
                train_avg_D_x_hat.append(torch.mean(probs_fake).detach().cpu().numpy())
                train_avg_D_x_hat_p.append(torch.mean(probs_uniform).detach().cpu().numpy())
                if batch_i == 0:
                    dist_gtscore.append(y.detach().cpu().numpy())
                    dist_scores.append(scores.detach().cpu().numpy())
                    dist_scores_uniform.append(scores_uniform.detach().cpu().numpy())

            # Log losses and probs for real and fake data by the end of the epoch
            train_avg_loss_s_e = np.mean(train_avg_loss_s_e)
            train_avg_loss_d = np.mean(train_avg_loss_d)
            train_avg_loss_c = np.mean(train_avg_loss_c)
            train_avg_D_x = np.mean(train_avg_D_x)
            train_avg_D_x_hat = np.mean(train_avg_D_x_hat)
            train_avg_D_x_hat_p = np.mean(train_avg_D_x_hat_p)
            self.log.info(f"Epoch: {epoch+1:3}/{self.hps.epochs:3}   "
                            f"Lse: {train_avg_loss_s_e:.05f}  "
                            f"Ld: {train_avg_loss_d:.05f}  "
                            f"Lc: {train_avg_loss_c:.05f}  "
                            f"D(x): {train_avg_D_x:.05f}  "
                            f"D(x_hat): {train_avg_D_x_hat:.05f}  "
                            f"D(x_hat_p): {train_avg_D_x_hat_p:.05f}")
            self.hps.writer.add_scalar(f'{self.dataset_name}/Fold_{fold+1}/Train/Lse', train_avg_loss_s_e, epoch)
            self.hps.writer.add_scalar(f'{self.dataset_name}/Fold_{fold+1}/Train/Ld', train_avg_loss_d, epoch)
            self.hps.writer.add_scalar(f'{self.dataset_name}/Fold_{fold+1}/Train/Lc', train_avg_loss_c, epoch)
            self.hps.writer.add_scalar(f'{self.dataset_name}/Fold_{fold+1}/Train/D_x', train_avg_D_x, epoch)
            self.hps.writer.add_scalar(f'{self.dataset_name}/Fold_{fold+1}/Train/D_x_hat', train_avg_D_x_hat, epoch)
            self.hps.writer.add_scalar(f'{self.dataset_name}/Fold_{fold+1}/Train/D_x_hat_p', train_avg_D_x_hat_p, epoch)

            # Log the distribution of scores predicted by the selector
            self.hps.writer.add_histogram(f'{self.dataset_name}/Fold_{fold+1}/Train/dist_gtscore', np.array(dist_gtscore), epoch)
            self.hps.writer.add_histogram(f'{self.dataset_name}/Fold_{fold+1}/Train/dist_scores', np.array(dist_scores), epoch)
            self.hps.writer.add_histogram(f'{self.dataset_name}/Fold_{fold+1}/Train/dist_scores_uniform', np.array(dist_scores_uniform), epoch)

            # Evaluate performances on test keys
            if epoch % self.hps.test_every_epochs == 0:
                avg_corr, (avg_f_score, max_f_score) = self.test(fold)
                self.model.train()
                self.hps.writer.add_scalar(f'{self.dataset_name}/Fold_{fold+1}/Test/Correlation', avg_corr, epoch)
                self.hps.writer.add_scalar(f'{self.dataset_name}/Fold_{fold+1}/Test/F-score_avg', avg_f_score, epoch)
                self.hps.writer.add_scalar(f'{self.dataset_name}/Fold_{fold+1}/Test/F-score_max', max_f_score, epoch)
                best_avg_f_score = max(best_avg_f_score, avg_f_score)
                best_max_f_score = max(best_max_f_score, max_f_score)
                if avg_corr > best_corr:
                    best_corr = avg_corr
                    self.best_weights = self.model.state_dict()

            # Free unused memory from GPU
            torch.cuda.empty_cache()

        return best_corr, best_avg_f_score, best_max_f_score


if __name__ == "__main__":
    model = SumGAN()
    print("Parameters:", sum([_.numel() for _ in model.parameters()]))

    model = Summarizer()
    x = torch.randn(10, 3, 1024)
    x_hat, scores = model(x)
    print(x.shape, x_hat.shape, scores.shape)
    assert x.shape[0] == scores.shape[0]
    assert x.shape[1] == scores.shape[1]
    assert scores.shape[2] == 1
    assert x.shape[0] == x_hat.shape[0]
    assert x.shape[1] == x_hat.shape[1]
    assert x.shape[2] == x_hat.shape[2]

    model = GAN()
    x = torch.randn(10, 3, 1024)
    probs, h = model(x)
    print(x.shape, probs.shape)
    assert x.shape[1] == probs.shape[0]
    assert probs.shape[1] == 1

import os
import sys
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from summarizer.models import Model

"""
Summarizing Videos with Attention
https://arxiv.org/abs/1812.01969

"""

class VASNet(nn.Module):
    def __init__(self, feature_dim=1024, max_length=None, pos_embed="simple", ignore_self=False, attention_aperture=None, scale=None, epsilon=1e-6, weight_init="xavier"):
        super(VASNet, self).__init__()

        # feature dimension that is the the dimensionality of the key, query and value vectors
        # as well as the hidden dimension for the FF layers
        self.feature_dim = feature_dim

        # Aperture to control the range of attention. If None, all frames are considered (global attn.)
        # If this is an integer w, frames [t-w, t+w] will be considered (local attention)
        self.aperture = attention_aperture
        
        # Whether to include the frame x_t in computing attention weights
        self.ignore_self = ignore_self

        # scaling factor to have more stable gradients. VasNet recommends 0.06,
        # but self-attention defaults to 1/square root of the dimension of the key vectors.
        self.scale = scale if scale is not None else 1 / np.sqrt(self.feature_dim)

        # Optional positional embeddings
        self.max_length = max_length
        if self.max_length:
            self.pos_embed_type = pos_embed

            if self.pos_embed_type == "simple":
                self.pos_embed = torch.nn.Embedding(self.max_length, self.feature_dim)
            elif self.pos_embed_type == "attention":
                self.pos_embed = torch.zeros(self.max_length, self.feature_dim)
                for pos in np.arange(self.max_length):
                    for i in np.arange(0, self.feature_dim, 2):
                        self.pos_embed[pos, i] = np.sin(pos / (10000 ** ((2 * i)/self.feature_dim)))
                        self.pos_embed[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1))/self.feature_dim)))
            else:
                self.max_length = None

        # Common steps
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = torch.nn.LayerNorm(self.feature_dim, epsilon)

        # self-attention layers
        self.K = nn.Linear(in_features=self.feature_dim, out_features=self.feature_dim, bias=False)
        self.Q = nn.Linear(in_features=self.feature_dim, out_features=self.feature_dim, bias=False)
        self.V = nn.Linear(in_features=self.feature_dim, out_features=self.feature_dim, bias=False)
        self.attention_head_projection = nn.Linear(in_features=self.feature_dim, out_features=self.feature_dim, bias=False)
        self.softmax = nn.Softmax(dim=2)

        # FFNN layers
        self.k1 = nn.Linear(in_features=self.feature_dim, out_features=self.feature_dim)
        self.k2 = nn.Linear(in_features=self.feature_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        # Linear layers weights initialization
        # VasNet uses a Xavier uniform initialization with a gain of sqrt(2) by default
        if weight_init.lower() in ["he", "kaiming"]:
            init.kaiming_uniform_(self.K.weight)
            init.kaiming_uniform_(self.Q.weight)
            init.kaiming_uniform_(self.V.weight)
            init.kaiming_uniform_(self.attention_head_projection.weight)

            init.kaiming_uniform_(self.k1.weight)
            init.kaiming_uniform_(self.k2.weight)
        else:
            init.xavier_uniform_(self.K.weight, gain=np.sqrt(2.0))
            init.xavier_uniform_(self.Q.weight, gain=np.sqrt(2.0))
            init.xavier_uniform_(self.V.weight, gain=np.sqrt(2.0))
            init.xavier_uniform_(self.attention_head_projection.weight, gain=np.sqrt(2.0))

            init.xavier_uniform_(self.k1.weight, gain=np.sqrt(2.0))
            init.xavier_uniform_(self.k2.weight, gain=np.sqrt(2.0))

        init.constant_(self.k1.bias, 0.1)
        init.constant_(self.k2.bias, 0.1)


    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape

        negative_inf = float('-inf')

        assert self.feature_dim == feature_dim

        if self.max_length is not None:
            assert self.max_length >= seq_len
            if self.pos_embed_type == "simple":
                pos_tensor = torch.arange(seq_len).repeat(1, batch_size).view([batch_size, seq_len]).to(x.device)
                x += self.pos_embed(pos_tensor)
            elif self.pos_embed_type == "attention":
                x += self.pos_embed[:seq_len, :].repeat(1, batch_size).view(batch_size, seq_len, feature_dim).to(x.device)

        K = self.K(x)
        Q = self.Q(x)
        V = self.V(x)

        e = torch.bmm(Q, torch.transpose(K, 1, 2))
        e = e * self.scale

        if self.ignore_self:
            e[:, torch.eye(seq_len).bool()] = negative_inf

        if self.aperture is not None:
            assert isinstance(self.aperture, int)
            scope_mask = torch.mul(torch.tril(e, diagonal=self.aperture), torch.triu(e, diagonal=-self.aperture)) 
            e[scope_mask == 0] = negative_inf

        alpha = self.softmax(e)
        alpha = self.dropout(alpha)
        c = torch.bmm(alpha, V)
        c = self.attention_head_projection(c)

        # Residual connection
        y = c + x
        y = self.dropout(y)
        y = self.layer_norm(y)

        # Two layer FFNN
        y = self.k1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.layer_norm(y)
        y = self.k2(y)
        y = self.sigmoid(y)
        y = y.view(batch_size, -1)

        return y


class VASNetModel(Model):
    def _init_model(self):
        model = VASNet(
            max_length=int(self.hps.extra_params["max_pos"]) if "max_pos" in self.hps.extra_params else None,
            pos_embed=self.hps.extra_params.get("pos_embed", "simple"),
            ignore_self=bool(self.hps.extra_params.get("ignore_self", False)),
            attention_aperture=int(self.hps.extra_params["local"]) if "local" in self.hps.extra_params else None,
            scale=float(self.hps.extra_params["scale"]) if "scale" in self.hps.extra_params else None,
            epsilon=float(self.hps.extra_params.get("epsilon", 1e-6)), 
            weight_init=self.hps.extra_params.get("weight_init", "xavier")
        )

        cuda_device = self.hps.cuda_device
        if self.hps.use_cuda:
            print("Setting CUDA device: ", cuda_device)
            torch.cuda.set_device(cuda_device)
        if self.hps.use_cuda:
            model.cuda()
        return model

    def train(self, fold):
        self.model.train()
        train_keys, _ = self._get_train_test_keys(fold)

        criterion = nn.MSELoss()
        if self.hps.use_cuda:
            criterion = criterion.cuda()

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=self.hps.lr, weight_decay=self.hps.l2_req)

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
                f_score = self.test(fold)
                self.model.train()
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

if __name__ == "__main__":
    pass

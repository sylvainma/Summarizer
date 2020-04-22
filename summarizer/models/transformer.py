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
Attention Is All You Need
https://arxiv.org/abs/1706.03762
https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoder
"""

class Transformer(nn.Module):
    def __init__(self, feature_dim=1024, encoder_layers=6, attention_heads=8, more_residuals=False, max_length=None, pos_embed="simple", epsilon=1e-5, weight_init=None):
        super(Transformer, self).__init__()

        # feature dimension that is the the dimensionality of the key, query and value vectors
        # as well as the hidden dimension for the FF layers
        self.feature_dim = feature_dim

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

        # Optional: Add a residual connection between before/after the Encoder layers
        self.more_residuals = more_residuals

        # Common steps
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = torch.nn.LayerNorm(self.feature_dim, epsilon)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=attention_heads, dim_feedforward=self.feature_dim, dropout=0.1, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.transformer_encoder_layer, num_layers=encoder_layers, norm=self.layer_norm)

        self.k1 = nn.Linear(in_features=self.feature_dim, out_features=self.feature_dim)
        self.k2 = nn.Linear(in_features=self.feature_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        # Weights initialization
        if weight_init:
            if weight_init.lower() in ["he", "kaiming"]:
                for i in np.arange(self.transformer_encoder.num_layers):
                    init.kaiming_uniform_(self.transformer_encoder.layers[i].linear1.weight)
                    init.kaiming_uniform_(self.transformer_encoder.layers[i].linear2.weight)
                init.kaiming_uniform_(self.k1.weight)
                init.kaiming_uniform_(self.k2.weight)
            elif weight_init.lower() == "xavier":
                for i in np.arange(self.transformer_encoder.num_layers):
                    init.xavier_uniform_(self.transformer_encoder.layers[i].linear1.weight)
                    init.xavier_uniform_(self.transformer_encoder.layers[i].linear2.weight)
                init.xavier_uniform_(self.k1.weight)
                init.xavier_uniform_(self.k2.weight)

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape

        if self.max_length is not None:
            assert self.max_length >= seq_len
            if self.pos_embed_type == "simple":
                pos_tensor = torch.arange(seq_len).repeat(1, batch_size).view([batch_size, seq_len]).to(x.device)
                x += self.pos_embed(pos_tensor)
            elif self.pos_embed_type == "attention":
                x += self.pos_embed[:seq_len, :].repeat(1, batch_size).view(batch_size, seq_len, feature_dim).to(x.device)

        encoder_out = self.transformer_encoder.forward(x)
        
        if self.more_residuals:
            encoder_out += x

        y = self.k1(encoder_out)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.layer_norm(y)
        y = self.k2(y)
        y = self.sigmoid(y)
        y = y.view(batch_size, -1)

        return y


class TransformerModel(Model):
    def _init_model(self):
        model = Transformer(
            encoder_layers=int(self.hps.extra_params.get("encoder_layers", 6)),
            attention_heads=int(self.hps.extra_params.get("attention_heads", 8)),
            more_residuals=self.hps.extra_params.get("more_residuals", False),
            max_length=int(self.hps.extra_params["max_pos"]) if "max_pos" in self.hps.extra_params else None,
            pos_embed=self.hps.extra_params.get("pos_embed", "simple"),
            epsilon=float(self.hps.extra_params.get("epsilon", 1e-5)), 
            weight_init=self.hps.extra_params.get("weight_init", None)
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
            self.hps.writer.add_scalar('{}/Fold_{}/Train/Loss'.format(self.hps.current_dataset, fold+1), train_avg_loss, epoch)

            # Evaluate performances on test keys
            if epoch % self.hps.test_every_epochs == 0 or epoch == 0:
                f_score = self.test(fold)
                self.hps.writer.add_scalar('{}/Fold_{}/Test/F-score'.format(self.hps.current_dataset, fold+1), f_score, epoch)
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
    model = Transformer()
    print("Trainable parameters in model:", sum([_.numel() for _ in model.parameters() if _.requires_grad]))

    print()
    print("Possible flags for Transformer:")
    print("encoder_layers: an integer describing the number of encoder layers to use in the transformer. Default=6")
    print("encoder_layers: an integer describing the number of attention heads to use in the transformer. Default=8")
    print("max_pos: an integer describing the maximum length of a sequence (e.g. the number of frames in each video). Specify to use positional encodings. Default=None")
    print("more_residuals. Specify to add a residual connection between before and after the encoder layers. Default=False")
    print("max_pos: an integer describing the maximum length of a sequence (e.g. the number of frames in each video). Specify to use positional encodings. Default=None")
    print("pos_embed: \"simple\" or \"attention\". Whether to use simple (embedding of the position of the image in sequence) or attention-based cos-sin positional encodings. Specify `max_pos` to use positional encodings. Default=simple")
    print("epsilon: a float added to the denominator for numerical stability when performing layer normalization. Default=1e-5")
    print("weight_init: \"xavier\" or \"he\"/\"kaiming\". Whether to use Xavier weight initialization, Kaiming initialization, or none. Default=None")

    x = torch.randn(10, 3, 1024)
    y = model(x)
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == y.shape[1]

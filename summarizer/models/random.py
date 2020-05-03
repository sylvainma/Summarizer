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
Random scores as a baseline for comparison.
"""

class Random(nn.Module):
    def __init__(self):
        super(Random, self).__init__()

    def forward(self, x):
        seq_len, batch_size, _ = x.shape
        scores = torch.rand((seq_len, batch_size, 1))
        scores = scores.to(x.device)
        return scores

class RandomModel(Model):
    def _init_model(self):
        model = Random()
        return model

    def draw_scores(self, keys):
        for i, key in enumerate(keys):
            d = self.dataset[key]
            video_name = d["video_name"][...]
            gtscore = d["gtscore"][...]
            self.hps.writer.add_histogram(f"{self.dataset_name}/dist_gtscore", gtscore, i)

    def train(self, fold):
        self.model.train()
        train_keys, _ = self._get_train_test_keys(fold)
        self.draw_scores(train_keys)

        criterion = nn.MSELoss()
        if self.hps.use_cuda:
            criterion = criterion.cuda()
        
        # To record performances of the best epoch
        best_corr, best_avg_f_score, best_max_f_score = -1.0, 0.0, 0.0

        # For each epoch
        for epoch in range(self.hps.epochs):
            train_avg_loss = []
            random.shuffle(train_keys)

            # For each training video
            for key in train_keys:
                dataset = self.dataset[key]
                seq = dataset["features"][...]
                seq = torch.from_numpy(seq).unsqueeze(1)
                target = dataset["gtscore"][...]
                target = torch.from_numpy(target).view(-1, 1, 1)

                # Normalize frame scores
                target -= target.min()
                target /= target.max()

                if self.hps.use_cuda:
                    seq, target = seq.cuda(), target.cuda()

                scores = self.model(seq)
                loss = criterion(scores, target)
                train_avg_loss.append(float(loss))

            # Average training loss value of epoch
            train_avg_loss = np.mean(np.array(train_avg_loss))
            self.log.info("Epoch: {0:6}    Train loss: {1:.05f}".format(
                str(epoch+1)+"/"+str(self.hps.epochs), train_avg_loss))
            self.hps.writer.add_scalar('{}/Fold_{}/Train/Loss'.format(self.dataset_name, fold+1), train_avg_loss, epoch)

            # Evaluate performances on test keys
            if epoch % self.hps.test_every_epochs == 0:
                avg_corr, (avg_f_score, max_f_score) = self.test(fold)
                self.model.train()
                self.hps.writer.add_scalar('{}/Fold_{}/Test/Correlation'.format(self.dataset_name, fold+1), avg_corr, epoch)
                self.hps.writer.add_scalar('{}/Fold_{}/Test/F-score_avg'.format(self.dataset_name, fold+1), avg_f_score, epoch)
                self.hps.writer.add_scalar('{}/Fold_{}/Test/F-score_max'.format(self.dataset_name, fold+1), max_f_score, epoch)
                best_avg_f_score = max(best_avg_f_score, avg_f_score)
                best_max_f_score = max(best_max_f_score, max_f_score)
                if avg_corr > best_corr:
                    best_corr = avg_corr
                    self.best_weights = self.model.state_dict()

        return best_corr, best_avg_f_score, best_max_f_score


if __name__ == "__main__":
    pass

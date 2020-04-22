import os
import sys
import numpy as np
import h5py
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from summarizer.vsum_tools import generate_summary, evaluate_summary

class Model:
    """Abstract class handling the training process"""
    def __init__(self, hps, splits_file):
        self.hps = hps
        self.splits_file = splits_file
        self.dataset = h5py.File(hps.dataset_of_file[splits_file], "r")
        self.dataset_name = hps.dataset_name_of_file[splits_file]
        
    def reset(self):
        """Reset between two folds of the cross-validation"""
        self.model = self._init_model()
        torch.cuda.empty_cache()
        if self.hps.use_cuda:
            self.model.cuda()
        return self

    def _get_train_test_keys(self, fold):
        """Train/Test keys from current split file and fold"""
        self.fold = fold
        self.split = self.hps.splits_of_file[self.splits_file][fold]
        return self.split["train_keys"][:], self.split["test_keys"][:]

    def _init_model(self):
        """Initialize here your model"""
        raise Exception("_init_model has not been implemented")

    def train(self, fold):
        """Train model on train_keys"""
        raise Exception("train has not been implemented")

    def test(self):
        """Test model on test_keys"""
        raise Exception("test has not been implemented")

    def predict(self, features):
        """Predict targets given features as input, should return a numpy"""
        seq = torch.from_numpy(features).unsqueeze(0)
        if self.hps.use_cuda:
            seq = seq.float().cuda()
        y = self.model(seq)
        y = y[0].detach().cpu().numpy()
        return y

    def predict_dataset(self, pred_path):
        """Predict on all videos in the dataset and save in hdfs5 file"""
        # Load best weights
        self.model.load_state_dict(self.best_weights)
        self.model.eval()
        # Create or open result hdfs5 file
        with h5py.File(pred_path, "w") as f:
            dataset_file = os.path.basename(self.hps.dataset_of_file[self.splits_file])
            g = f.create_group(dataset_file)
            # Get machine summary for each key
            for key in self.dataset.keys():
                # Get video data
                d = self.dataset[key]
                features = d["features"][...]
                cps = d['change_points'][...]
                num_frames = d['n_frames'][()]
                nfps = d['n_frame_per_seg'][...].tolist()
                positions = d['picks'][...]
                user_summary = d['user_summary'][...]
                # Predict scores and compute machine summary
                scores = self.predict(features)
                machine_summary = generate_summary(scores, cps, num_frames, nfps, positions)
                # Save in hdfs5 file
                k = g.create_group(key)
                k.create_dataset("scores", data=scores)
                k.create_dataset("user_summary", data=user_summary)
                k.create_dataset("machine_summary", data=machine_summary)

    def save_best_weights(self, weights_path):
        """Dump current best weights"""
        if self.best_weights is None:
            raise Exception("best_weights property is empty, can't save model's weights")
        torch.save(self.best_weights, weights_path)

    def load_weights(self, weights_path):
        """Load weights"""
        self.model.load_state_dict(torch.load(weights_path))

    def _eval_summary(self, machine_summary_activations, test_keys):
        eval_metric = 'avg' if self.dataset_name == 'tvsum' else 'max'

        fms = []
        for key in test_keys:
            d = self.dataset[key]
            probs = machine_summary_activations[key]

            if 'change_points' not in d:
                print("ERROR: No change points in dataset/video ", key)

            cps = d['change_points'][...]
            num_frames = d['n_frames'][()]
            nfps = d['n_frame_per_seg'][...].tolist()
            positions = d['picks'][...]
            user_summary = d['user_summary'][...]

            machine_summary = generate_summary(probs, cps, num_frames, nfps, positions)
            fm, _, _ = evaluate_summary(machine_summary, user_summary, eval_metric)
            fms.append(fm)

        f_score = np.mean(fms)
        return f_score

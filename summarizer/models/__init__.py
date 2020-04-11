import numpy as np
import h5py
import torch
from vsum_tools import generate_summary, evaluate_summary

class Model:
    """Abstract class handling the training process"""
    def __init__(self, hps, splits_file, fold):
        self.hps = hps
        self.splits_file = splits_file
        self.fold = fold
        self.split = hps.splits_of_file[splits_file][fold]
        self.dataset = h5py.File(hps.dataset_of_file[splits_file], "r")
        self.metric = hps.metric_of_file[splits_file]
        self.best_weights = None
        self.model = self._init_model()

    def _init_model(self):
        """Initialize here your model"""
        raise Exception("_init_model has not been implemented")
    
    def train(self):
        """Train model on train_keys"""
        raise Exception("train has not been implemented")

    def test(self):
        """Test model on test_keys"""
        raise Exception("test has not been implemented")

    def _eval_summary(self, machine_summary_activations, test_keys):
        eval_metric = 'avg' if self.metric == 'tvsum' else 'max'

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
    
    def save_best_weights(self, weights_path):
        """Dump current best weights"""
        if self.best_weights is None:
            raise Exception("best_weights property is empty, can't save model's weights")
        torch.save(self.best_weights, weights_path)

    def load_weights(self, weights_path):
        """Load weights"""
        self.model.load_state_dict(torch.load(weights_path))

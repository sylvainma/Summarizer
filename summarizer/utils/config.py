from torch.autograd import Variable
from . import parse_splits_filename
from models.vasnet import VASNetModel
from models.baseline import LogisticRegressionModel

class HParameters:
    """Hyperparameters configuration class"""
    def __init__(self):
        self.verbose = False
        self.use_cuda = False
        self.cuda_device = 0
        self.max_summary_length = 0.15

        self.l2_req = 0.00001
        self.lr = 0.00005

        self.epochs_max = 300
        self.train_batch_size = 1

        # Experiment name, used as output directory
        self.output_dir = 'ex-10'

        # Project root directory
        self.root = ''
        self.datasets = ['datasets/eccv16_dataset_summe_google_pool5.h5',
                        'datasets/eccv16_dataset_tvsum_google_pool5.h5',
                        'datasets/eccv16_dataset_ovp_google_pool5.h5',
                        'datasets/eccv16_dataset_youtube_google_pool5.h5']

        # Split files to be trained/tested on
        self.splits_files = ['splits/tvsum_splits.json', 'splits/summe_splits.json']

        # Default model
        self.model_class = VASNetModel

        # For other dynamic properties
        self._init()

    def _init(self):
        # List of splits by filename
        self.dataset_of_file = {}
        self.splits_of_file = {}
        self.metric_of_file = {}
        for splits_file in self.splits_files:
            dataset_name, splits = parse_splits_filename(splits_file)
            self.dataset_of_file[splits_file] = self.get_dataset_by_name(dataset_name).pop()
            self.splits_of_file[splits_file] = splits
            self.metric_of_file[splits_file] = dataset_name

    def load_from_args(self, args):
        for key in args:
            val = args[key]
            if val is not None:
                if hasattr(self, key) and isinstance(getattr(self, key), list):
                    val = val.split()
                setattr(self, key, val)
        if "model" in args:
            self.model_class = {
                "baseline": LogisticRegressionModel,
                "vasnet": VASNetModel
            }.get(args["model"], LogisticRegressionModel)
        self._init()

    def get_dataset_by_name(self, dataset_name):
        for d in self.datasets:
            if dataset_name in d:
                return [d]
        return None

    def __str__(self):
        """Nicely lists hyperparameters when object is printed"""
        vars = ["verbose", "use_cuda", "cuda_device",
                "l2_req", "lr", "epochs_max",
                "output_dir", "splits_files"]
        info_str = ''
        for i, var in enumerate(vars):
            val = getattr(self, var)
            if isinstance(val, Variable):
                val = val.data.cpu().numpy().tolist()[0]
            info_str += "["+str(i)+"] "+var+": "+str(val)
            info_str += "\n" if i < len(vars)-1 else ""

        return info_str

if __name__ == "__main__":
    # Check default values
    hps = HParameters()
    print(hps)
    # Check update with args works well
    args = {
        'root': 'root_dir',
        'datasets': 'set1,set2,set3',
        'splits': 'split1, split2',
        'new_param_float': 1.23456
    }
    hps.load_from_args(args)
    print(hps)
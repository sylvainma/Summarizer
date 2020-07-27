import os
import sys
import shutil
import inspect
import logging
import datetime
import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from summarizer.utils import parse_splits_filename
from summarizer.models.rand import RandomTrainer
from summarizer.models.logistic import LogisticRegressionTrainer
from summarizer.models.vasnet import VASNetTrainer
from summarizer.models.transformer import TransformerTrainer
from summarizer.models.dsn import DSNTrainer
from summarizer.models.sumgan import SumGANTrainer
from summarizer.models.sumgan_att import SumGANAttTrainer


class HParameters:
    """Hyperparameters configuration class"""
    def __init__(self):
        """Place in init the default values"""
        self.use_cuda = False
        self.cuda_device = 0

        self.weight_decay = 0.00001
        self.lr = 0.00005

        self.epochs = 10
        self.test_every_epochs = 2

        # Project root directory
        self.datasets = [
            "datasets/summarizer_dataset_summe_google_pool5.h5",
            "datasets/summarizer_dataset_tvsum_google_pool5.h5",
            "datasets/summarizer_dataset_LOL_google_pool5.h5"]

        # Default split files to be trained/tested on
        self.splits_files = "minimal"

        # Default model
        self.model_class = LogisticRegressionTrainer

        # Dict containing extra parameters, possibly model-specific
        self.extra_params = None

        # Length of summary video as a proportion of the original
        self.summary_proportion = 0.15
        
        # Keyshot selection algorithm to build the summary video (0-1 knapsack/rank)
        self.selection_algorithm = "knapsack"

        # Logger default level is INFO
        self.log_level = logging.INFO

    def load_from_args(self, args):
        # Any key from flags
        for key in args:
            val = args[key]
            if val is not None:
                if hasattr(self, key) and isinstance(getattr(self, key), list):
                    val = val.split(",")
                setattr(self, key, val)

        # Pick model
        self.model_class = {
            "random": RandomTrainer,
            "logistic": LogisticRegressionTrainer,
            "vasnet": VASNetTrainer,
            "transformer": TransformerTrainer,
            "dsn": DSNTrainer,
            "sumgan": SumGANTrainer,
            "sumgan_att": SumGANAttTrainer,
            None: RandomTrainer
        }.get(args["model"], None)
        if self.model_class is None:
            raise KeyError(f"{args['model']} model is not unknown")

        # Other dynamic properties
        self._init()

    def _init(self):
        # Experiment name, used as output directory
        log_dir = str(int(datetime.datetime.now().timestamp()))
        log_dir += "_" + self.model_class.__name__
        self.log_path = os.path.join("logs", log_dir)
        
        # Tensorboard
        self.writer = SummaryWriter(self.log_path)

        # Handle use_cuda flag
        if self.use_cuda == "default":
            self.use_cuda = torch.cuda.is_available()
        elif self.use_cuda == "yes":
            self.use_cuda = True
        else:
            self.use_cuda = False

        # Specify CUDA device (possibly different from GPU 0)
        if self.use_cuda:
            torch.cuda.set_device(self.cuda_device)

        # Handle splits options/shorthands
        if self.splits_files == "minimal":
            # minimal working example
            self.splits_files = ["splits/tvsum_splits_overfit.json"]
        elif self.splits_files == "overfit":
            # POC using overfitting splits (1 fold with train=test) for TVSum and SumMe datasets
            self.splits_files = [
                "splits/tvsum_splits_overfit.json",
                "splits/summe_splits_overfit.json"]
        elif self.splits_files == "tvsum":
            self.splits_files = ["splits/tvsum_splits.json"]
        elif self.splits_files == "summe":
            self.splits_files = ["splits/summe_splits.json"]
        elif self.splits_files == "LOL":
            self.splits_files = ["splits/LOL_splits.json"]
        elif self.splits_files == "all":
            self.splits_files = [
                "splits/tvsum_splits.json",
                "splits/tvsum_splits_overfit.json",
                "splits/summe_splits.json",
                "splits/summe_splits_overfit.json",
                "splits/LOL_splits.json"]
        # ... or a custom split file, or list of split files

        # List of splits by filename
        self.dataset_name_of_file = {}
        self.dataset_of_file = {}
        self.splits_of_file = {}
        for splits_file in self.splits_files:
            dataset_name, splits = parse_splits_filename(splits_file)
            self.dataset_name_of_file[splits_file] = dataset_name
            self.dataset_of_file[splits_file] = self.get_dataset_by_name(dataset_name).pop()
            self.splits_of_file[splits_file] = splits

        # Destination for weights and predictions on dataset
        self.weights_path = {}
        self.pred_path = {}
        for splits_file in self.splits_files:
            weights_file = f"{os.path.basename(splits_file)}.pth"
            self.weights_path[splits_file] = os.path.join(self.log_path, weights_file)
            pred_file = f"{os.path.basename(splits_file)}_preds.h5"
            self.pred_path[splits_file] = os.path.join(self.log_path, pred_file)

        # Create log path if does not exist
        os.makedirs(self.log_path, exist_ok=True)

        # Logger
        self.logger = logging.getLogger("summarizer")
        fmt = logging.Formatter("%(asctime)s::%(levelname)s: %(message)s", "%H:%M:%S")
        ch = logging.StreamHandler()
        fh = logging.FileHandler(os.path.join(self.log_path, "train.log"))
        ch.setFormatter(fmt)
        fh.setFormatter(fmt)
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)
        self.logger.setLevel(getattr(logging, self.log_level.upper()))

        # Save model file into log directory
        src = inspect.getfile(self.model_class)
        dst = os.path.join(self.log_path, os.path.basename(src))
        shutil.copyfile(src, dst)

    def get_dataset_by_name(self, dataset_name):
        for d in self.datasets:
            if dataset_name in d:
                return [d]
        return None

    def __str__(self):
        """Nicely lists hyperparameters when object is printed"""
        vars = ["use_cuda", "cuda_device", "log_level",
                "weight_decay", "lr", "epochs",
                "summary_proportion", "selection_algorithm",
                "log_path", "splits_files", "extra_params"]
        info_str = ""
        for i, var in enumerate(vars):
            val = getattr(self, var)
            if isinstance(val, Variable):
                val = val.data.cpu().numpy().tolist()[0]
            info_str += "["+str(i)+"] "+var+": "+str(val)
            info_str += "\n" if i < len(vars)-1 else ""

        return info_str

    def get_full_hps_dict(self):
        """Returns the list of hyperparameters as a flat dict"""
        vars = ["weight_decay", "lr", "epochs"]

        hps = {}
        for i, var in enumerate(vars):
            val = getattr(self, var)
            if isinstance(val, Variable):
                val = val.data.cpu().numpy().tolist()[0]
            hps[var] = val

        return hps

if __name__ == "__main__":
    # Check default values
    hps = HParameters()
    print(hps)
    # Check update with args works well
    args = {
        "root": "root_dir",
        "datasets": "set1,set2,set3",
        "splits": "split1, split2",
        "new_param_float": 1.23456
    }
    hps.load_from_args(args)
    print(hps)

import argparse
from utils.config import HParameters

def train(hps):
    """Training"""
    # For every split file
    for splits_file in hps.splits_files:
        print("Start training on {}".format(splits_file))
        n_folds = len(hps.splits_of_file[splits_file])
        fscore_cv = 0.0

        # For every fold in current split file
        for fold in range(n_folds):
            model = hps.model_class(hps, splits_file, fold)
            model.train()
            fscore_test = model.test()
            fscore_cv += fscore_test

            # Report F-score of current fold
            print("File: {}   Split: {}/{}   Test F-score: {:0.5f}".format(
                splits_file, fold+1, n_folds, fscore_test))

        # Report cross-validation F-score of current split file
        fscore_cv /= n_folds
        print("File: {0:}   Cross-validation F-score: {1:0.5f}".format(splits_file, fscore_cv))

def evaluate(hps):
    """Evaluation on test set"""
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser("CS7643 Spring 2020 Project : Video Summarization")
    parser.add_argument('-r', '--root', type=str, default='', help="Project root directory")
    parser.add_argument('-d', '--datasets', type=str, help="Path to a comma separated list of h5 datasets")
    parser.add_argument('-s', '--splits-files', type=str, help="Comma separated list of split files")
    parser.add_argument('-o', '--output-dir', type=str, default='data', help="Experiment name")
    parser.add_argument('-v', '--verbose', action='store_true', help="Prints out more messages")
    parser.add_argument('-t', '--test', action='store_true', help="Test")
    parser.add_argument('-e', '--epochs-max', type=int, default=300, help="Number of epochs")
    parser.add_argument('-m', '--model', type=str, help="Model class name")
    args = parser.parse_args()

    hps = HParameters()
    hps.load_from_args(args.__dict__)
    print("Hyperparameters:")
    print("----------------------------------------------------------------------")
    print(hps)
    print("----------------------------------------------------------------------")

    if hps.test:
        evaluate(hps)
    else:
        train(hps)

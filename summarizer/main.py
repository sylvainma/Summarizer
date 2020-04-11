import os
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
        models = []
        for fold in range(n_folds):
            model = hps.model_class(hps, splits_file, fold)
            best_fscore = model.train()
            fscore_cv += best_fscore
            models.append((fold, model, best_fscore))

            # Report F-score of current fold
            print("File: {}   Split: {}/{}   Best F-score: {:0.5f}".format(
                splits_file, fold+1, n_folds, best_fscore))

        # Report cross-validation F-score of current split file
        fscore_cv /= n_folds
        print("File: {0:}   Cross-validation F-score: {1:0.5f}".format(splits_file, fscore_cv))

        # Dump weights of the best model among folds
        best_model = max(models, key=lambda m: m[2])[1]
        weights_file = f"{os.path.basename(splits_file)}.pth"
        weights_path = os.path.join(hps.log_path, weights_file)
        best_model.save_best_weights(weights_path)
        print("File: {0:}   Best weights: {1:}".format(splits_file, weights_path))


def test(hps):
    """Evaluation on test keys"""
    # For every split file
    for splits_file in hps.splits_files:
        print("Start testing on {}".format(splits_file))
        n_folds = len(hps.splits_of_file[splits_file])
        fscore_avg = 0.0

        # For every fold in current split file
        for fold in range(n_folds):
            model = hps.model_class(hps, splits_file, fold)
            model.load_weights(hps.weights_of_file[splits_file])
            fscore = model.test()
            fscore_avg += fscore

            # Report F-score of current fold
            print("File: {}   Split: {}/{}   F-score: {:0.5f}".format(
                splits_file, fold+1, n_folds, fscore))

        # Report cross-validation F-score of current split file
        fscore_avg /= n_folds
        print("File: {0:}   Average F-score: {1:0.5f}".format(splits_file, fscore_avg))


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
    parser.add_argument('-w', '--weights-path', type=str, help="Weights path")
    args = parser.parse_args()
    hps = HParameters()
    hps.load_from_args(args.__dict__)
    print("Hyperparameters:")
    print("----------------------------------------------------------------------")
    print(hps)
    print("----------------------------------------------------------------------")

    if hps.test:
        test(hps)
    else:
        train(hps)

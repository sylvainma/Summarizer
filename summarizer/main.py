import os
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from summarizer.utils.config import HParameters


def train(hps):
    """Training"""
    # For every split file
    for splits_file in hps.splits_files:
        print("Start training on {}".format(splits_file))
        n_folds = len(hps.splits_of_file[splits_file])
        fscore_cv = 0.0
        
        # Destination for weights and predictions on dataset
        weights_path = hps.weights_path[splits_file]
        pred_path = hps.pred_path[splits_file]

        # For every fold in current split file
        fscore_max = 0.0
        model = hps.model_class(hps, splits_file)
        for fold in range(n_folds):
            fold_best_fscore = model.reset().train(fold)
            fscore_cv += fold_best_fscore
            
            # Save weights if it is the current maximum F-score
            if fold_best_fscore > fscore_max:
                fscore_max = fold_best_fscore
                model.save_best_weights(weights_path)

            # Report F-score of current fold
            print("File: {}   Fold: {}/{}   Fold best F-score: {:0.5f}".format(
                splits_file, fold+1, n_folds, fold_best_fscore))

        # Report cross-validation F-score of current split file and location of best weights
        fscore_cv /= n_folds
        print("File: {0:}   Cross-validation F-score: {1:0.5f}".format(splits_file, fscore_cv))
        print("File: {0:}   Best weights: {1:}".format(splits_file, weights_path))

        # Predict on all videos of the dataset using the best weights
        model.reset().load_weights(weights_path)
        model.predict_dataset(pred_path)
        print("File: {0:}   Machine summaries: {1:}".format(splits_file, pred_path))


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
            print("File: {}   Fold: {}/{}   F-score: {:0.5f}".format(
                splits_file, fold+1, n_folds, fscore))

        # Report cross-validation F-score of current split file
        fscore_avg /= n_folds
        print("File: {0:}   Average F-score: {1:0.5f}".format(splits_file, fscore_avg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CS7643 Spring 2020 Project : Video Summarization")
    parser.add_argument('-v', '--verbose', action='store_true', help="Prints out more messages")
    parser.add_argument('-c', '--use-cuda', choices=['yes', 'no', 'default'], default='default', help="Use cuda for pytorch models")
    parser.add_argument('-d', '--datasets', type=str, help="Path to a comma separated list of h5 datasets")
    parser.add_argument('-s', '--splits-files', type=str, help="Comma separated list of split files")
    parser.add_argument('-m', '--model', type=str, help="Model class name")
    parser.add_argument('-e', '--epochs-max', type=int, default=300, help="Number of epochs for train mode")
    parser.add_argument('-w', '--weights-path', type=str, help="Weights path")
    parser.add_argument('-t', '--test', action='store_true', help="Test mode")
    args, unknown_args = parser.parse_known_args()

    hps_init = args.__dict__
    extra_params = {unknown_args[i].lstrip('-'): u.lstrip('-') if u[0] != '-' else True for i, u in enumerate(unknown_args[1:] + ['-']) if unknown_args[i][0] == '-'} if len(unknown_args) > 0 else {}
    hps_init["extra_params"] = extra_params

    hps = HParameters()
    hps.load_from_args(hps_init)
    print("Hyperparameters:")
    print("----------------------------------------------------------------------")
    print(hps)
    print("----------------------------------------------------------------------")

    if hps.test:
        test(hps)
    else:
        train(hps)

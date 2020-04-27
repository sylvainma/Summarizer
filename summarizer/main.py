import os
import argparse
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from summarizer.utils.config import HParameters


def train(hps):
    """Training"""
    # For every split file
    results = []
    for splits_file in hps.splits_files:
        hps.logger.info(f"Start training on {splits_file}")
        n_folds = len(hps.splits_of_file[splits_file])
        corrs_cv, avg_fscores_cv, max_fscores_cv = [], [], []
        
        # Destination for weights and predictions on dataset
        weights_path = hps.weights_path[splits_file]
        pred_path = hps.pred_path[splits_file]

        # For every fold in current split file
        corr_max = -1.0
        model = hps.model_class(hps, splits_file)
        for fold in range(n_folds):
            fold_best_corr, fold_best_avg_f_score, fold_best_max_f_score = model.reset().train(fold)
            corrs_cv.append(fold_best_corr)
            avg_fscores_cv.append(fold_best_avg_f_score)
            max_fscores_cv.append(fold_best_max_f_score)
            
            # Save weights if it is the current maximum correlation
            if fold_best_corr > corr_max:
                corr_max = fold_best_corr
                model.save_best_weights(weights_path)

            # Report F-score of current fold
            hps.logger.info(
                f"File: {splits_file}   "
                f"Fold: {fold+1}/{n_folds}   "
                f"Corr: {fold_best_corr: 0.5f}  "
                f"Avg F-score: {fold_best_avg_f_score:0.5f}  "
                f"Max F-score: {fold_best_max_f_score:0.5f}")

        # Report cross-validation F-score of current split file and location of best weights
        hps.logger.info(
            f"File: {splits_file}   "
            f"Cross-validation Corr: {np.mean(corrs_cv): 0.5f}  "
            f"Avg F-score: {np.mean(avg_fscores_cv):0.5f}  "
            f"Max F-score: {np.mean(max_fscores_cv):0.5f}")
        hps.logger.info(f"File: {splits_file}   Best weights: {weights_path}")

        # Log it for Tensorboard
        hparam_dict = hps.get_full_hps_dict()
        hparam_dict["dataset"] = hps.dataset_name_of_file[splits_file]
        metric_dict = {f"Correlation/Fold_{f+1}": corr for f, corr in enumerate(corrs_cv)}
        metric_dict = {f"F-score_avg/Fold_{f+1}": score for f, score in enumerate(avg_fscores_cv)}
        metric_dict = {f"F-score_max/Fold_{f+1}": score for f, score in enumerate(max_fscores_cv)}
        metric_dict["Correlation/CV_Average"] = np.mean(corrs_cv)
        metric_dict["F-score_avg/CV_Average"] = np.mean(avg_fscores_cv)
        metric_dict["F-score_max/CV_Average"] = np.mean(max_fscores_cv)
        hps.writer.add_hparams(hparam_dict, metric_dict)

        # Predict on all videos of the dataset using the best weights
        model.reset().load_weights(weights_path)
        model.predict_dataset(pred_path)
        hps.logger.info(f"File: {splits_file}   Machine predictions: {pred_path}")

        # Save results of current splits file
        results.append((splits_file, np.mean(corrs_cv), np.mean(avg_fscores_cv), np.mean(max_fscores_cv)))
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Summarizer : Model Training")
    parser.add_argument('-c', '--use-cuda', choices=['yes', 'no', 'default'], default='default', help="Use cuda for pytorch models")
    parser.add_argument('-s', '--splits-files', type=str, help="Comma separated list of split files")
    parser.add_argument('-m', '--model', type=str, help="Model class name")
    parser.add_argument('-e', '--epochs', type=int, help="Number of epochs for train mode")
    parser.add_argument('-l', '--log-level', choices=['critical', 'error', 'warning', 'info', 'debug'], default='info', help="Set logger to custom level")
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

    train(hps)

    # Close the Tensorboard writer
    hps.writer.close()

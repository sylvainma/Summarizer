import os
import sys
import datetime
import logging
import argparse
import torch
import pandas as pd
from tabulate import tabulate
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from summarizer.utils.config import HParameters
from summarizer.main import train

"""
Benchmark table. Models with their best known hyperparamaters should be 
integrated here. Once models are trained, a benchmark table is shown to 
termial but it is also saved into logs.
"""

def benchmark(args, log_path):
    """Successively train models"""
    table_results = []
    base = {
        "splits_files": args.splits_files,
        "log_level": "error",
    }

    ###############################
    # Random
    ###############################
    table_results += benchmark_model("Random", dict({
        "model": "random",
        "epochs": 1,
        "extra_params": {}
    }, **base))

    ###############################
    # Logistic Regression
    ###############################
    table_results += benchmark_model("Logistic Regression", dict({
        "model": "logistic",
        "epochs": min(30, args.max_epochs),
        "extra_params": {}
    }, **base))

    # Finally show results and save them in logs
    table = pd.DataFrame(table_results, columns=[
        "Model", "File", "Correlation", "Avg F-score", "Max F-score", "Logs"])
    show_save_results(table, log_path)


def benchmark_model(name, args):
    """Routine to train one model"""
    logging.info(f"Train {name} model...")
    hps = HParameters()
    hps.load_from_args(args)
    model_results = train(hps)
    results = []
    for splits_file, corr, avg_fscore, max_fscore in model_results:
        results.append([name, splits_file, corr, avg_fscore, max_fscore, hps.log_path])
        logging.info(
            f"File: {splits_file}  "
            f"Corr: {corr: 0.5f}  "
            f"Avg F-score: {avg_fscore:0.5f}  "
            f"Max F-score: {max_fscore:0.5f}")
    logging.info(f"Logs saved in {hps.log_path}")
    return results


def show_save_results(table, log_path):
    """Display to terminal and save to logs the Pandas table"""
    # Display in terminal
    table_str = tabulate(table, headers="keys", tablefmt="psql", showindex=False)
    print(table_str)

    # Save to logs
    os.makedirs(log_path, exist_ok=True)
    table_file = os.path.join(log_path, "table.txt")
    with open(table_file, "w") as f:
        f.write(table_str)
    logging.info(f"Table saved in {table_file}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s::%(levelname)s: %(message)s")
    
    # Where to save benchmark results
    log_dir = str(int(datetime.datetime.now().timestamp()))
    log_path = os.path.join("logs", f"{log_dir}_benchmark") 

    # Split files for benchmark
    splits_files = ",".join([
        "splits/tvsum_splits.json",
        "splits/summe_splits.json"])

    # Maximum number of epochs per model
    max_epochs = 300

    # Args
    parser = argparse.ArgumentParser("Summarizer : Benchmark")
    parser.add_argument('-e', '--max-epochs', type=int, default=max_epochs, help="Maximum number of epochs per model")
    parser.add_argument('-s', '--splits-files', type=str, default=splits_files, help="Comma separated list of split files")
    args, _ = parser.parse_known_args()

    # Start benchmark
    print(args)
    benchmark(args, log_path)
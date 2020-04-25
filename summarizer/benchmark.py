import os
import sys
import datetime
import logging
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

def benchmark(splits_files, max_epochs, log_path):
    """Successively train models"""
    table_results = []

    ###############################
    # Random
    ###############################
    table_results += benchmark_model("Random", {
            "model": "random",
            "splits_files": splits_files,
            "epochs": 1,
            "log_level": "error",
            "extra_params": {}
        })

    ###############################
    # Logistic Regression
    ###############################
    table_results += benchmark_model("Logistic Regression", {
            "model": "logistic",
            "splits_files": splits_files,
            "epochs": min(30, max_epochs),
            "log_level": "error",
            "extra_params": {}
        })

    # Finally show results and save them in logs
    table = pd.DataFrame(table_results, columns=["Model", "File", "Correlation", "F-score", "Logs"])
    show_save_results(table, log_path)


def benchmark_model(name, args):
    """Routine to train one model"""
    logging.info(f"Train {name} model...")
    hps = HParameters()
    hps.load_from_args(args)
    model_results = train(hps)
    results = []
    for splits_file, corr, fscore in model_results:
        logging.info(f"File: {splits_file}   Corr: {corr}   F-score: {fscore}")
        results.append([name, splits_file, corr, fscore, hps.log_path])
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
    max_epochs = 1

    # Start benchmark
    benchmark(splits_files, max_epochs, log_path)
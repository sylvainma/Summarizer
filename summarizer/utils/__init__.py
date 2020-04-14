import os
import json

def parse_splits_filename(splits_filename):
    # Parse split file and count number of k_folds
    _, sfname = os.path.split(splits_filename)
    sfname, _ = os.path.splitext(sfname)
    dataset_name = sfname.split('_')[0]  # Get dataset name e.g. tvsum

    # Get number of discrete splits within each split json file
    with open(splits_filename, 'r') as sf:
        splits = json.load(sf)

    # dataset_name: summe, tvsum, etc
    # splits: [{"train_keys": [...], "test_keys": [...]}, ...]
    # with len(splits) := number of k_folds
    return dataset_name, splits

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

class Proportion(object):
    """ Restricted float class describing a proportion in ]0, 1]."""
    def __eq__(self, value):
        return 0 < value <= 1

    def __contains__(self, item):
        return self.__eq__(item)

    def __iter__(self):
        yield 'a proportion value in ]0, 1]'

    def __str__(self):
        return 'a proportion value in ]0, 1]'

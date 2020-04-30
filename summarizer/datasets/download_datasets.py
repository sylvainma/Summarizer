import os
import os.path as osp
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from summarizer.utils.io import download_file_from_google_drive

"""
Download `summarizer_dataset_*.h5` files automatically.
"""

def download_datasets():
    datasets_path = osp.dirname(osp.abspath(__file__))
    datasets = [
        ("summarizer_dataset_summe_google_pool5.h5", "1LUcnvGpGzt5X59-x72N02k-zXm5dt9Hn"),
        ("summarizer_dataset_tvsum_google_pool5.h5", "1Ur-q0O9gi-VgBLNM7X8bdhSkcoI-CrnC"),
        ("summarizer_dataset_LOL_google_pool5.h5", "1suaESy2yxuCshcLFN-7IjmtvEXOL4nrV")]
    
    for dst, fid in datasets:
        path = osp.join(datasets_path, dst)
        if osp.exists(path): 
            os.remove(path)
        print(f"Downloading {dst}...")
        download_file_from_google_drive(fid, path)


if __name__ == "__main__":
    download_datasets()
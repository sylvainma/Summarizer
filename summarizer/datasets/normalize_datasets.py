import os
import sys
import h5py
import numpy as np
import pandas as pd
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from summarizer.utils.io import load_tvsum_mat
from summarizer.utils.eval import upsample

"""
Script for passing from eccv16_dataset_*.h5 to summarizer_dataset_*.h5.
"""

###############################
# TVSum
###############################
# Retrieve the information TSV containing the list of video names for TvSum.
# In the h5, they're called video_(x)y. The TSV gives us the mapping from the original video names. 
tvsum_info = pd.read_table("videos/tvsum/ydata-tvsum50-info.tsv").reset_index()
tvsum_info["index"] = tvsum_info["index"].apply(lambda x: "video_"+str(x+1))

# Retrieve the original annotations scores for each frame, as [1,5] grades, which we normalize in [0,1]
tvsum_data = load_tvsum_mat("videos/tvsum/ydata-tvsum50.mat")
user_scores = {"video_"+str(x+1): (video["user_anno"].T-1.0)/(5.0-1.0) for x, video in enumerate(tvsum_data)}

# Load the TvSum h5
shutil.move('eccv16_dataset_tvsum_google_pool5.h5', 'summarizer_dataset_tvsum_google_pool5.h5')
tvsum = h5py.File('summarizer_dataset_tvsum_google_pool5.h5', 'r+')

for key in tvsum.keys():
    # Add the "video_name" key for each of the videos, just like in SumMe.
    tvsum[key].create_dataset("video_name",
        dtype=h5py.string_dtype(encoding='utf-8'), 
        data=tvsum_info[tvsum_info['index'] == key]['video_id'].get(0))
    
    # Add original normalized importance scores annotations
    tvsum[key].create_dataset("user_scores",
        data=user_scores[key])

tvsum.close()
print("TVSum done.")

###############################
# SumMe
###############################
# Rename the SumMe h5 file for consistency. We didn't change anything there.
shutil.move('eccv16_dataset_summe_google_pool5.h5', 'summarizer_dataset_summe_google_pool5.h5')
summe = h5py.File('summarizer_dataset_summe_google_pool5.h5', 'r+')

for key in summe.keys():
    # Add /user_scores for consistency with TVSum
    user_scores = np.expand_dims(upsample(
        summe[key]["gtscore"][...], 
        summe[key]["n_frames"][...], 
        summe[key]["picks"][...])
    , axis=0)
    summe[key].create_dataset("user_scores",
        data=user_scores)

summe.close()
print("SumMe done.")
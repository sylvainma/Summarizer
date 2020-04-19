import sys
import h5py
import numpy as np
import pandas as pd
import shutil

# Retrieve the information TSV containing the list of video names for TvSum.
# In the h5, they're called video_(x)y. The TSV gives us the mapping from the original video names. 
tvsum_info = pd.read_table("videos/tvsum/ydata-tvsum50-info.tsv").reset_index()
tvsum_info["index"] = tvsum_info["index"].apply(lambda x: "video_"+str(x+1))

# Load the TvSum h5 and add the "video_name" key for each of the videos, just like in SumMe.
shutil.move('eccv16_dataset_tvsum_google_pool5.h5', 'summarizer_dataset_tvsum_google_pool5.h5')
tvsum = h5py.File('summarizer_dataset_tvsum_google_pool5.h5', 'r+')

for key in tvsum.keys():
    video_name = tvsum[key].create_dataset("video_name", dtype=h5py.string_dtype(encoding='utf-8'), data = tvsum_info[tvsum_info['index'] == key]['video_id'].get(0))

tvsum.close()

# Rename the SumMe h5 file for consistency. We didn't change anything there.
shutil.move('eccv16_dataset_summe_google_pool5.h5', 'summarizer_dataset_summe_google_pool5.h5')
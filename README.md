# VideoSummarization

## Splits
To generate dataset splits:
```
python create_split.py -d datasets/eccv16_dataset_summe_google_pool5.h5 --save-dir splits --save-name summe_splits --num-splits 5
python create_split.py -d datasets/eccv16_dataset_tvsum_google_pool5.h5 --save-dir splits --save-name tvsum_splits --num-splits 5
```
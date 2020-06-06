# Summarizer

## Train
```
python main.py --model <model_name>
```
See `python main.py --help` for more parameters. Weights and logs are saved in `logs/<timestamp>_<model_name>`.

## How to implement a new model
Go to `/models` and create a new file of the name of the model. Create Pytorch model inherited from `nn.Module` to design the model architecture. In the same file create a class inherited from `Model` (in `/models/__init__.py`) to handle the epoch loop. A simple example is the baseline model in `/models/baseline.py`. In `utils/config.py` add in `load_from_args` method your model.

## Generate video summaries
When training, by the end of the classification, scores are computed on every video of the dataset using the best found weights. The generated summaries are saved in `logs/<timestamp>_<model_name>/<dataset_name>_preds.h5`. Video summaries can be generated using `summary2video.py`:
```
python summary2video.py -p logs/1586668539_LogisticRegressionModel/summe_splits.json_preds.h5 -f datasets/videos/summe/frames/Air_Force_One -d summarizer_dataset_summe_google_pool5.h5 -v video_1
```

## Generate splits
```
python create_split.py -d datasets/summarizer_dataset_summe_google_pool5.h5 --save-dir splits --save-name summe_splits --num-splits 5
```

## Build new HDF5 files
While we provide (in `/datasets`) existing and new HDF5 files corresponding to a few of the classic datasets in video summarization (SumMe, TVSum, Twitch-LOL), you may create your own files using `generate_dataset.py`. This allows the generation of a custom HDF5 dataset following the same template as the existing datasets (including extracted features, keyshots, labels, etc.).

This script was derived from [Shin Donghwan](https://github.com/SinDongHwan/pytorch-vsumm-reinforce/blob/master/utils/generate_dataset.py)'s own generation script, and using [Tatsuya Shirakawa](https://github.com/TatsuyaShirakawa/KTS)'s Cython rewrite of KTS for Python 3.

For example, to rebuild [Twitch-LOL](https://github.com/chengyangfu/Pytorch-Twitch-LOL#dataset-download---google-drive) with a 2-second uniform segmentation, you may use:
```
python generate_dataset.py --video datasets/videos/EMNLP17_Twitch_LOL/final_data --annotations datasets/videos/EMNLP17_Twitch_LOL/gt --h5 datasets/summarizer_dataset_LOL_google_pool5.h5 --changepoint-method uniform --changepoint-duration 4 --keyshot-sampling 15 --extractor googlenet --layer-limit=-2
```

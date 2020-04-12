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
python summary2video.py -p logs/1586668539_LogisticRegressionModel/summe_splits.json_preds.h5 -f datasets/videos/summe/frames/Air_Force_One -d eccv16_dataset_summe_google_pool5.h5 -v video_1
```

## Generate splits
```
python create_split.py -d datasets/eccv16_dataset_summe_google_pool5.h5 --save-dir splits --save-name summe_splits --num-splits 5
```
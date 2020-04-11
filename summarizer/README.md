# Summarizer

## How to implement a new model
Go to `/models` and create a new file of the name of the model. Create Pytorch model inherited from `nn.Module` to design the model architecture. In the same file create a class inherited from `Model` (in `/models/__init__.py`) to handle the epoch loop. A simple example is the baseline model in `/models/baseline.py`.

In `utils/config.py` add in `load_from_args` method your model.

Finally to train this model run:
```
python main.py --model <model_name>
```

## Splits
To generate dataset splits:
```
python create_split.py -d datasets/eccv16_dataset_summe_google_pool5.h5 --save-dir splits --save-name summe_splits --num-splits 5
python create_split.py -d datasets/eccv16_dataset_tvsum_google_pool5.h5 --save-dir splits --save-name tvsum_splits --num-splits 5
```
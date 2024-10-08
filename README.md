# Cross-modal Recipe Retrieval based on Unified Text Encoder with Fine-grained Contrastive Learning

This is the PyTorch companion code for the paper:

*Bolin Zhang, Haruya Kyutoku, Keisuke Doman, Takahiro Komamizu, Ichiro Ide, Jiangbo Qian.* **Cross-modal Recipe Retrieval based on Unified Text Encoder with Fine-grained Contrastive Learning**, Knowledge-Based Systems, 2024.
Note: The configuration file will be uploaded following the acceptance of the paper.


# Installation

- Create conda environment: 

```
conda env create -f environment.yml
```

- Activate it with: 

```
conda activate UTE_FCL
```



## Data preparation



- Download & uncompress Recipe1M [dataset](http://im2recipe.csail.mit.edu/dataset/download). The contents of the directory `DATASET_PATH` should be the following:

```
layer1.json
layer2.json
train/
val/
test/
```



The directories `train/`, `val/`, and `test/` must contain the image files for each split after uncompressing.

- Make splits and create vocabulary by running:

```
python preprocessing.py --root DATASET_PATH
```



This process will create auxiliary files under `DATASET_PATH/traindata`, which will be used for training.

## Training



- Launch training with:

```
python train.py --model_name model --root DATASET_PATH --save_dir /path/to/saved/model/checkpoints
```



Tensorboard logging can be enabled with `--tensorboard`. Then, from the checkpoints directory run:

```
tensorboard --logdir "./" --port PORT
```



Run `python train.py --help` for the full list of available arguments.

## Evaluation



- Extract features from the trained model for the test set samples of Recipe1M:

```
python test.py --model_name model --eval_split test --root DATASET_PATH --save_dir /path/to/saved/model/checkpoints
```



- Compute MedR and recall metrics for the extracted feature set:

```
python eval.py --embeddings_file /path/to/saved/model/checkpoints/model/feats_test.pkl --medr_N 10000
```



## Pretrained models



- We provide pretrained model weights under the `checkpoints` directory. Make sure you run `git lfs pull` to download the model files.
- Extract the zip files. For each model, a folder named `MODEL_NAME` with two files, `args.pkl`, and `model-best.ckpt` is provided.
- Extract features for the test set samples of Recipe1M using one of the pretrained models by running:

```
python test.py --model_name MODEL_NAME --eval_split test --root DATASET_PATH --save_dir ../checkpoints
```



- A file with extracted features will be saved under `../checkpoints/MODEL_NAME`.Pretrained models

##  Acknowledgements

We are grateful for [image-to-recipe-transformers](https://github.com/amzn/image-to-recipe-transformers); it has been very helpful to us.

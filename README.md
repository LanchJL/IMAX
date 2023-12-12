# IMAX
This is the official PyTorch code for Imaginatively-connected Embedding in Complex Space for Unseen Attribute-Object Discrimination (IMAX)

## Setup
Our work is implemented in PyTorch 1.7.0 and tested with Ubuntu 20.14

Before running codes, you can create a environment using 
```shell
conda env create -n IMAX -f environment.yml
conda activate IMAX
```

## Datasets
The splits of dataset and its attributes can be found in utils/download_data.sh, the complete installation process can be found in [CGE&CompCos](https://github.com/ExplainableML/czsl).
You can download the datasets using
```shell
bash utils/download_data.sh
```
And you can set the --DATA_FOLDER in flag.py as your dataset path.

DINO pretrained VIT-B-16 can be found [Here](https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth). Please place the downloaded file in the `IMAX/pretrain/`

## Train
If you wish to try training our model from scratch, please run train.py, for example: 

```shell
  python train.py --config CONFIG_FILE
```

## Test
Please specify the path for the trained weights, and than run:

```shell
   python test.py --config CONFIG_FILE test_weights_path --WEIGHTS_PATH
```
## Acknowledgements
Our overall code is built on top of [CGE&CompCos](https://github.com/ExplainableML/czsl), and we sincerely appreciate the great help this work has given us.

# IMAX
Code of Imaginatively-connected Embedding in Complex Space for Unseen Attribute-Object Discrimination (IMAX)

The full code file is in the process of being organized and will be fully open-sourced by December 15th.

## Datasets
The splits of dataset and its attributes can be found in utils/download_data.sh, the complete installation process can be found in [CGE] https://github.com/ExplainableML/czsl

Set the --DATA_FOLDER in flag.py as your dataset path.

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

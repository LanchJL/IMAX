# Imaginary-Connected Embedding in Complex Space for Unseen Attribute-Object Discrimination
* **Title**: **[Imaginary-Connected Embedding in Complex Space for Unseen Attribute-Object Discrimination](https://durham-repository.worktribe.com/preview/3020370/2994465AAM.pdf)**
* **Institutes**: Nanjing University of Science and Technology, Newcastle University, Durham University, University of Chinese Academy of Sciences
* **Publication Status**: This paper has been accepted for publication in IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)

# News
**[2025.1]** To enhance readability, we restructured the IMAX code based on [Troika](https://github.com/bighuang624/Troika?tab=readme-ov-file) and introduced an Adapter module to facilitate the adaptation of the CLIP visual encoder to the test dataset. 

**[2025.1]** We have completed the open-sourcing of the CLIP-based IMAX code, and the remaining encoder implementations will be uploaded shortly.

## Setup
```bash
conda create --name imax python=3.8
conda activate imax
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip3 install git+https://github.com/openai/CLIP.git
```
The remaining dependencies can be found in the ``./requirements.txt`` file and installed using `pip install -r requirements.txt`.

DINO pretrained VIT-B-16 can be found [Here](https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth). Please place the downloaded file in the `./pretrain/`

The CLIP weights can be downloaded via [CLIP](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) and should be placed in the ./clip_modules directory.

## Datasets
The splits of dataset and its attributes can be found in utils/download_data.sh, the complete installation process can be found in [CGE&CompCos](https://github.com/ExplainableML/czsl).
You can download the datasets using
```shell
bash utils/download_data.sh
```

## Train
If you wish to try training our model from scratch, for example, for ``UT-Zappos``: 
```shell
python -u train.py \
--clip_arch ./clip_modules/ViT-L-14.pt \
--dataset_path ../dataset/CZSL/ut-zap50k \
--save_path ./save_dir/ut-zappos \
--yml_path ./config/clip/ut-zappos.yml \
--num_workers 4 \
--seed 0 \
--adapter
```

``MIT-States``: 

```shell
python -u train.py \
--clip_arch ./clip_modules/ViT-L-14.pt \
--dataset_path ../dataset/CZSL/mit-states \
--save_path ./save_dir/mit-states \
--yml_path ./config/clip/mit-states.yml \
--num_workers 2 \
--seed 0 \
--adapter
```

``C-GQA``:
```shell
python -u train.py \
--clip_arch ./clip_modules/ViT-L-14.pt \
--dataset_path ../dataset/CZSL/cgqa \
--save_path ./save_dir/cgqa \
--yml_path ./config/clip/cgqa.yml \
--num_workers 2 \
--seed 0 \
--adapter
```

## Acknowledgement
The code we publish is based on the following outstanding repositories, which have helped us a lot
* [DFSP](https://github.com/Forest-art/DFSP)
* [Troika](https://github.com/bighuang624/Troika?tab=readme-ov-file)

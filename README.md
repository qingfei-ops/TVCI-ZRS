## Installation
Please see [Installation Instructions](INSTALL.md).

## Datasets
Download datasets for zero-shot remote sensing instance segmentation from [Hugging Face <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face Logo" width="20"/>](https://huggingface.co/datasets/HuangShiqi008/ZoRI).

## Getting Started
Please see [Getting Started with TVCI-ZRS](GETTING_STARTED.md).

## Training
```
python train_net.py --config-file configs/tvci_isaid_11_4.yaml
```
## Inference
###
For GZSRI setting, run
```
python train_net.py  --config-file configs/tvci_isaid_11_4.yaml --eval-only MODEL.WEIGHTS [path_to_weights]
```
###
For ZSRI setting, run
```
python train_net.py  --config-file configs/tvci_isaid_11_4.yaml --eval-only MODEL.WEIGHTS [path_to_weights] DATASETS.TEST '("isaid_zsi_11_4_val_unseen",)' MODEL.GENERALIZED False MODEL.CACHE_BANK.ALPHA 0.6
```
###
Then get pseudo unseen visual prototypes from previous inference results, run
```
python -m tvci.utils.cache_model_unseen --config configs/cache.yaml DATASET 'isaid_zsi_11_4_val' PROTOTYPE_NUM [1] RESULTS [path_to_json]
```
###
Finally, inference again with pseudo unseen visual prototypes to get final predictions.


## Acknowledgement
This project is based on [FC-CLIP](https://github.com/bytedance/fc-clip). Many thanks to the authors for their great work!

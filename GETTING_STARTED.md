## Getting Started with TVCI-ZRS
Please see [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) for full usage.

###
Get modules ready before training:

###
To get refined text embedding indices, please run
```
python -m tvci.tools.ctSEC_get_optimise_textchannel --config configs/cache.yaml DATASET 'isaid_zsi_11_4_val'
```

###
To select visual channels, please run
```
python -m tvci.tools.ctCDA_get_optimise_vischannel --config configs/cache.yaml DATASET 'isaid_zsi_11_4_train'
```

###
To prepare cache bank, please run
```
python -m tvci.tools.ctVTP-FP_cache_unseenlibrary --config configs/cache.yaml DATASET 'isaid_zsi_11_4_train_all' PROTOTYPE_NUM [4]
```


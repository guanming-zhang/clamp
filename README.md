Contrastive Learning As Manifold Packing (CLAMP)
---------------------------------------------------------------
# Usage
Use the following command to run the training:
```
python main.py /path/to/directory /path/to/default_config.ini
```
where we use config files(config_file.ini and default_config.ini) to spcify the training setup and hyperparameters. /path/to/directory must contain the input configureation, config.ini (its name must be config.ini). Missing parameters in your_config.ini will be replaced by its default value specified in default_config.ini. Note that directory to your own imagenet training and validation dataset needs to be specified in either config.ini. Pytorchlightning checkpoints and tensorboard/csv logging file are strored in /path/to/directory .

To run the example file for training ImageNet-1K:
```
python main.py /example /default_cofigs/default_config_imagenet1k.ini
```


# Dependencies:
python 3.9 (python 2 is note supported) <br/>
numpy <br/>
scipy <br/>
matplotlib <br/>
lmdb <br/>
pytorch 2.5.1 <br/>
pytrochlightning 2.4.0 <br/>
tensorboard <br/>
albumentation 1.4.24

import sys
from utils import data_utils
import helper
import matplotlib.pyplot as plt
from utils import data_utils
import torch
from model import models
import os
from model import lightning_models
import math
import json
import pytorch_lightning as pl
import gc
if __name__ == '__main__':
    input_dir= sys.argv[1]
    default_config_file = sys.argv[2]
    config = helper.Config(input_dir, default_config_file)
    if config.INFO["fix_random_seed"]:
        pl.seed_everything(137) # To be reproducable
    # save the starting time as the last line of file staring-time.txt
    current_datetime,zone = helper.get_est_time_now()
    if os.path.isfile(os.path.join(input_dir,"starting-time.txt")):
        with open(os.path.join(input_dir,"starting-time.txt"),"a") as f:
            f.write("\n")
            f.write(current_datetime.strftime("%Y-%m-%d %H:%M:%S"))
    else:
        with open(os.path.join(input_dir,"starting-time.txt"),"a") as f:
            f.write(current_datetime.strftime("%Y-%m-%d %H:%M:%S"))

    ###################################################
    # load pretrained model
    ###################################################
    print("---------------LOAD PRETRAINED MODEL-----------------------")
    # dataset and dataloader
    # for multi-gpu trainning, effective batch size = batch_size*num_gpus
    ssl_batch_size = config.SSL["batch_size"] // (config.INFO["num_nodes"]*config.INFO["gpus_per_node"])
    # note that standardize_to_imagenet=Flase and augment_val_set = True are recomended
    ssl_train_loader,ssl_test_loader,ssl_val_loader = data_utils.get_dataloader(config.DATA,ssl_batch_size,
                                                                                num_workers = config.INFO["cpus_per_gpu"],
                                                                                standardized_to_imagenet=False,
                                                                                augment_val_set = True,
                                                                                prefetch_factor=config.INFO["prefetch_factor"],
                                                                                skip_validation= config.SSL["skip_validation"],
                                                                                aug_pkg = config.DATA["augmentation_package"])

    # setup the self-supervised learning
    if config.SSL["lr_scale"] == "linear":
        ssl_lr = config.SSL["lr"]*config.SSL["batch_size"]/256.0 # lr ~ 0.1
    elif config.SSL["lr_scale"] == "sqrt":
        ssl_lr = config.SSL["lr"]*math.sqrt(config.SSL["batch_size"]) # lr ~ 0.05
    if "CIFAR" in config.DATA["dataset"] or "MNIST" in config.DATA["dataset"]:
        prune_backbone = True
    else:
        prune_backbone = False
    ssl_model = lightning_models.CLAMP(backbone_name = config.SSL["backbone"],
                                  prune = prune_backbone,
                                  use_projection_head=config.SSL["use_projection_head"],
                                  proj_dim = config.SSL["proj_dim"],
                                  proj_out_dim = config.SSL["proj_out_dim"],
                                  loss_name= config.SSL["loss_function"],
                                  optim_name = config.SSL["optimizer"],
                                  lr = ssl_lr,
                                  scheduler_name = config.SSL["lr_scheduler"],
                                  momentum = config.SSL["momentum"],
                                  weight_decay = config.SSL["weight_decay"],
                                  eta = config.SSL["lars_eta"],
                                  warmup_epochs = config.SSL["warmup_epochs"],
                                  n_epochs = config.SSL["n_epochs"],
                                  exclude_bn_bias_from_weight_decay = config.SSL["exclude_bn_bias_from_weight_decay"], 
                                  n_views = config.DATA["n_views"],
                                  batch_size = ssl_batch_size,
                                  lw0 = config.SSL["lw0"],
                                  lw1 = config.SSL["lw1"],
                                  lw2 = config.SSL["lw2"],
                                  pot_pow = config.SSL["pot_pow"],
                                  rs = config.SSL["rs"])
    if config.INFO["num_nodes"]*config.INFO["gpus_per_node"] > 1:
        ssl_model.backbone = torch.nn.SyncBatchNorm.convert_sync_batchnorm(ssl_model.backbone)
    ssl_dir = os.path.join(config.loc,"ssl")
    last_ckpt = os.path.join(ssl_dir,'ssl-epoch={:d}.ckpt'.format(config.SSL["n_epochs"]-1))
    if os.path.isfile(last_ckpt):
        print(f'Found pretrained model at {last_ckpt}, loading...')
        ssl_model = lightning_models.CLAMP.load_from_checkpoint(last_ckpt)
    else:
        print(f'Pretrained model at {last_ckpt} not found !')
        raise Exception("Pretrained model not found") 
    
    ###################################################
    # linear classification
    ###################################################
    print("---------------LINEAR CLASSIFICATION-------------------------")
    lc_batch_size = config.LC["batch_size"] // (config.INFO["num_nodes"]*config.INFO["gpus_per_node"])
    # need to specify the location of the data for imagenet
    data_info = {"dataset":config.DATA["dataset"],"batch_size":lc_batch_size,"n_views":1,"n_trans":1,"augmentations":["RandomResizedCrop","RandomHorizontalFlip"],
            "crop_size":[config.DATA["crop_size"][0]],"crop_min_scale":[0.08],"crop_max_scale":[1.0],"hflip_prob":[0.5]}
    if "lc_dataset" in config.LC:
        data_info["dataset"] = config.LC["lc_dataset"]
    # need to specify the location of the data for imagenet
    if "IMAGENET1K" in config.DATA["dataset"]:
        data_info["imagenet_train_dir"] = config.DATA["imagenet_train_dir"]
        data_info["imagenet_val_dir"] = config.DATA["imagenet_val_dir"]

    lc_train_loader,lc_test_loader,lc_val_loader = data_utils.get_dataloader(data_info,lc_batch_size,num_workers=config.INFO["cpus_per_gpu"],
                                                                         standardized_to_imagenet=config.LC["standardize_to_imagenet"],
                                                                         prefetch_factor=config.INFO["prefetch_factor"])

    # setup the linear classification
    lc_dir = os.path.join(config.loc,"lc")
    if not os.path.isdir(lc_dir):
        os.makedirs(lc_dir,exist_ok=True)
    if "lr_sweep" in config.LC:
        lr_list = config.LC["lr_sweep"]
    else:
        lr_list = [config.LC["lr"]]
    # sweep learning rates
    best = {"best_test_acc1":0.0,"best_test_acc5":0.0,"best_test_loss":0.0,"best_model_dir":"none"}
    for lr in lr_list:
        lc_sub_dir = os.path.join(lc_dir,"lr_{}".format(lr))
        os.makedirs(lc_sub_dir,exist_ok=True)
        if config.LC["lr_scale"] == "linear":
            lc_lr = lr*config.LC["batch_size"]/256.0 # lr ~ 0.1
        elif config.LC["lr_scale"] == "sqrt":
            lc_lr = lr*math.sqrt(config.LC["batch_size"]) # lr ~ 0.05
        # load the backbone form the latest checkpoint
        # best_ssl_ckpt = os.path.join(ssl_dir,"best_val.ckpt")
        latest_ssl_ckpt = lightning_models.get_top_n_latest_checkpoints(ssl_dir,1)[0]
        ssl_model = lightning_models.CLAMP.load_from_checkpoint(latest_ssl_ckpt)
        ssl_model.backbone.remove_projection_head()
        ssl_model.backbone = torch.nn.SyncBatchNorm.convert_sync_batchnorm(ssl_model.backbone)  
        lc_model = lightning_models.LinearClassification(
                 backbone = ssl_model.backbone,
                 in_dim = ssl_model.backbone.feature_dim,
                 out_dim = config.LC["output_dim"],
                 use_batch_norm = config.LC["use_batch_norm"],
                 optim_name = config.LC["optimizer"],
                 scheduler_name = config.LC["lr_scheduler"],
                 lr = lc_lr, 
                 momentum = config.LC["momentum"],
                 weight_decay = config.LC["weight_decay"],
                 n_epochs = config.LC["n_epochs"])
        # convert batch norm to sync batch norm if ddp
        if config.INFO["num_nodes"]*config.INFO["gpus_per_node"] > 1:
            lc_model.linear_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(lc_model.linear_net)  
        with helper.Timer("LC Training"):
            if config.INFO["strategy"] == "ddp":
                strategy = "ddp_find_unused_parameters_true"
            else:
                strategy = config.INFO["strategy"]
            lc_model = lightning_models.train_lc(
                linear_model = lc_model,
                train_loader = lc_train_loader,
                val_loader = lc_val_loader,
                test_loader = lc_test_loader,
                max_epochs = config.LC["n_epochs"],
                every_n_epochs = config.LC["save_every_n_epochs"],
                precision = config.INFO["precision"],
                checkpoint_path = lc_sub_dir,
                strategy = strategy,
                num_nodes = config.INFO["num_nodes"],
                gpus_per_node = config.INFO["gpus_per_node"], 
                if_profile=config.INFO["if_profile"])
        # get the best performed one
        with open(os.path.join(lc_sub_dir,"results.json")) as f:
            result = json.load(f)
        if result["test_acc1"] > best["best_test_acc1"]:
            best["best_test_acc1"] = result["test_acc1"] 
            best["best_test_acc5"] = result["test_acc5"] 
            best["best_test_loss"] = result["test_loss"]
            best["best_model_dir"] = lc_sub_dir
    # clear the memory
    del lc_train_loader
    del lc_test_loader
    del lc_val_loader
    del lc_model
    gc.collect()
    torch.cuda.empty_cache()
        
    #save the information about the best model
    with open(os.path.join(lc_dir,"best_result.json"),"w") as f:
        json.dump(best,f,indent=4) 

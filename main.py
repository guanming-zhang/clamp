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
import pytorch_lightning as pl
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
    # self-superivesed learning
    ###################################################
    # dataset and dataloader
    # for multi-gpu trainning, effective batch size = batch_size*num_gpus
    ssl_batch_size = config.SSL["batch_size"] // (config.INFO["num_nodes"]*config.INFO["gpus_per_node"])
    ssl_train_loader,ssl_test_loader,ssl_val_loader = data_utils.get_dataloader(config.DATA,ssl_batch_size,
                                                                                num_workers = config.INFO["cpus_per_gpu"],validation=True,
                                                                                augment_val_set=True,
                                                                                standardized_to_imagenet=False)
    del ssl_test_loader
    # setup the self-supervised learning
    if config.SSL["lr_scale"] == "linear":
        ssl_lr = config.SSL["lr"]*config.SSL["batch_size"]/256.0 # lr ~ 0.1
    elif config.SSL["lr_scale"] == "sqrt":
        ssl_lr = config.SSL["lr"]*math.sqrt(config.SSL["batch_size"]) # lr ~ 0.05
    if "CIFAR" in config.DATA["dataset"] or "MNIST" in config.DATA["dataset"]:
        prune_backbone = True
    else:
        prune_backbone = False
    ssl_model = lightning_models.CLAP(backbone_name = config.SSL["backbone"],
                                  backbone_out_dim = config.SSL["backbone_out_dim"],
                                  prune = prune_backbone,
                                  use_projection_head=config.SSL["use_projection_head"],
                                  proj_dim = config.SSL["proj_dim"],
                                  proj_out_dim = config.SSL["proj_out_dim"],
                                  optim_name = config.SSL["optimizer"],
                                  lr = ssl_lr,
                                  momentum = config.SSL["momentum"],
                                  weight_decay = config.SSL["weight_decay"],
                                  eta = config.SSL["lars_eta"],
                                  warmup_epochs = config.SSL["warmup_epochs"],
                                  n_epochs = config.SSL["n_epochs"],
                                  n_views = config.DATA["n_views"],
                                  batch_size = ssl_batch_size,
                                  lw0 = config.SSL["lw0"],
                                  lw1 = config.SSL["lw1"],
                                  lw2 = config.SSL["lw2"],
                                  pot_pow = config.SSL["pot_pow"],
                                  rs = config.SSL["rs"])
    ssl_dir = os.path.join(config.loc,"ssl")
    if not os.path.isdir(ssl_dir):
        os.makedirs(ssl_dir,exist_ok=True)
    with helper.Timer("SSL Training"):
        ssl_model = lightning_models.train_clap(model=ssl_model, 
                                        train_loader = ssl_train_loader,
                                        val_loader = ssl_val_loader,
                                        max_epochs=config.SSL["n_epochs"],
                                        every_n_epochs = config.SSL["save_every_n_epochs"],
                                        precision = config.INFO["precision"],
                                        strategy = config.INFO["strategy"],
                                        num_nodes = config.INFO["num_nodes"],
                                        gpus_per_node = config.INFO["gpus_per_node"], 
                                        checkpoint_path=ssl_dir,
                                        restart = config.LC["restart_training"])
    ###################################################
    # linear classification
    ###################################################
    lc_batch_size = config.LC["batch_size"] // (config.INFO["num_nodes"]*config.INFO["gpus_per_node"])
    # need to specify the location of the data for imagenet
    if config.LC["apply_simple_augmentations"]:
        data_info = {"dataset":config.DATA["dataset"],"batch_size":lc_batch_size,"n_views":1,"augmentations":["RandomResizedCrop","RandomHorizontalFlip"],
                 "crop_size":config.DATA["crop_size"],"crop_min_scale":0.08,"crop_max_scale":1.0,"hflip_prob":0.5}
    else:
        data_info = {"dataset":config.DATA["dataset"],"batch_size":lc_batch_size,"n_views":1,"augmentations":[]}
    # need to specify the location of the data for imagenet
    if "IMAGENET1K" in config.DATA["dataset"]:
        data_info["imagenet_train_dir"] = config.DATA["imagenet_train_dir"]
        data_info["imagenet_val_dir"] = config.DATA["imagenet_val_dir"]

    lc_train_loader,lc_test_loader,lc_val_loader = data_utils.get_dataloader(data_info,lc_batch_size,num_workers=config.INFO["cpus_per_gpu"],
                                                                         standardized_to_imagenet=config.LC["standardize_to_imagenet"])
    # setup the linear classification
    lc_dir = os.path.join(config.loc,"lc")
    if not os.path.isdir(lc_dir):
        os.makedirs(lc_dir,exist_ok=True)
    if config.LC["lr_scale"] == "linear":
        lc_lr = config.LC["lr"]*config.LC["batch_size"]/256.0 # lr ~ 0.1
    elif config.LC["lr_scale"] == "sqrt":
        lc_lr = config.LC["lr"]*math.sqrt(config.LC["batch_size"]) # lr ~ 0.05
    # load the backbone form the latest checkpoint
    best_ssl_ckpt = os.path.join(ssl_dir,"best_val.ckpt")
    ssl_model = lightning_models.CLAP.load_from_checkpoint(best_ssl_ckpt)
    ssl_model.backbone.remove_projection_head()

    lc_model = lightning_models.LinearClassification(
                 backbone = ssl_model.backbone,
                 in_dim = config.SSL["backbone_out_dim"],
                 out_dim = config.LC["output_dim"],
                 use_batch_norm = config.LC["use_batch_norm"],
                 optim_name = config.LC["optimizer"],
                 lr = lc_lr, 
                 momentum = config.LC["momentum"],
                 weight_decay = config.LC["weight_decay"],
                 n_epochs = config.LC["n_epochs"])
    
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
                checkpoint_path = lc_dir,
                strategy = strategy,
                num_nodes = config.INFO["num_nodes"],
                gpus_per_node = config.INFO["gpus_per_node"], 
                restart = config.LC["restart_training"])
    ###################################################
    # Semi-supervised learning(if SemiSL section exists)
    ###################################################
    if len(config.SemiSL) > 0:
        semisl_batch_size = config.SemiSL["batch_size"] // (config.INFO["num_nodes"]*config.INFO["gpus_per_node"])
        if config.INFO["strategy"] == "ddp":
            strategy = "ddp_find_unused_parameters_true"
        else:
            strategy = config.INFO["strategy"]
        for dataset in ["IMAGENET1K-1%","IMAGENET1K-10%"]:
            if config.SemiSL["apply_simple_augmentations"]:
                data_info = {"dataset":dataset,"batch_size":semisl_batch_size,"n_views":1,"augmentations":["RandomResizedCrop","RandomHorizontalFlip"],
                     "crop_size":config.DATA["crop_size"],"crop_min_scale":0.08,"crop_max_scale":1.0,"hflip_prob":0.5}
            else:
                data_info = {"dataset":dataset,"batch_size":semisl_batch_size,"n_views":1,"augmentations":[]}
            # add the location for imagenet dataset
            data_info["imagenet_train_dir"] = config.DATA["imagenet_train_dir"]
            data_info["imagenet_val_dir"] = config.DATA["imagenet_val_dir"]
            semisl_train_loader,semisl_test_loader,semisl_val_loader = data_utils.get_dataloader(data_info,semisl_batch_size,num_workers=config.INFO["cpus_per_gpu"],
                                                                                 standardized_to_imagenet=config.SemiSL["standardize_to_imagenet"])
            semisl_dir = os.path.join(config.loc,"semisl-"+dataset)
            if not os.path.isdir(semisl_dir):
                os.makedirs(semisl_dir,exist_ok=True)
            if config.SemiSL["lr_scale"] == "linear":
                semisl_lr = config.SemiSL["lr"]*config.SemiSL["batch_size"]/256.0 # lr ~ 0.1
            elif config.LC["lr_scale"] == "sqrt":
                semisl_lr = config.SemiSL["lr"]*math.sqrt(config.SemiSL["batch_size"]) # lr ~ 0.05
            # load the backbone from the checkpoint
            best_ssl_ckpt = os.path.join(ssl_dir,"best_val.ckpt")
            ssl_model = lightning_models.CLAP.load_from_checkpoint(best_ssl_ckpt)
            ssl_model.backbone.remove_projection_head()
            # load the linear classifier from the checkpoint
            lc_model = lightning_models.LinearClassification.load_from_checkpoint(os.path.join(lc_dir,"best_val.ckpt"),backbone = ssl_model.backbone)
            semisl_model = lightning_models.FineTune(backbone = ssl_model.backbone,
                    linear_net= lc_model.linear_net,
                    optim_name = config.SemiSL["optimizer"],
                    lr = semisl_lr, 
                    momentum = config.SemiSL["momentum"],
                    weight_decay = config.LC["weight_decay"],
                    n_epochs = config.SemiSL["n_epochs"])
            semisl_model = lightning_models.train_finetune(
                    finetune_model = semisl_model,
                    train_loader = semisl_test_loader,
                    test_loader = semisl_test_loader,
                    val_loader = semisl_val_loader,
                    max_epochs = config.SemiSL["n_epochs"],
                    every_n_epochs = config.SemiSL["save_every_n_epochs"],
                    checkpoint_path = semisl_dir,
                    precision = config.INFO["precision"],
                    strategy = strategy,
                    num_nodes = config.INFO["num_nodes"],
                    gpus_per_node = config.INFO["gpus_per_node"],
                    restart = config.SemiSL["restart_training"])
    
    ##################################################
    # Transfer learning(freeze the backbone)
    ##################################################
    # Transfer learning(freeze backbone)
    if len(config.TL) > 0:
        tl_output_dim = {"CIFAR100":100,
                        "FOOD101":101,
                        "FLOWERS102":102}
        for dataset in ["CIFAR100","FOOD101","FLOWERS102"]:
            tl_batch_size = config.TL["batch_size"] // (config.INFO["num_nodes"]*config.INFO["gpus_per_node"])
            # must apply random cropping to normalize the image size to [224,224]
            data_info = {"dataset":dataset,"batch_size":semisl_batch_size,"n_views":1,"augmentations":["RandomResizedCrop","RandomHorizontalFlip"],
                     "crop_size":config.DATA["crop_size"],"crop_min_scale":0.08,"crop_max_scale":1.0,"hflip_prob":0.5}
            tl_train_loader,tl_test_loader,tl_val_loader = data_utils.get_dataloader(data_info,lc_batch_size,num_workers=config.INFO["cpus_per_gpu"],
                                                                                standardized_to_imagenet=config.TL["standardize_to_imagenet"])
            tl_dir = os.path.join(config.loc,"tl-"+dataset)
            if not os.path.isdir(tl_dir):
                os.makedirs(tl_dir,exist_ok=True)

            if config.TL["lr_scale"] == "linear":
                tl_lr = config.TL["lr"]*config.TL["batch_size"]/256.0 # lr ~ 0.1
            elif config.TL["lr_scale"] == "sqrt":
                tl_lr = config.TL["lr"]*math.sqrt(config.TL["batch_size"]) # lr ~ 0.05

            # load the backbone from the checkpoint
            best_ssl_ckpt = os.path.join(ssl_dir,"best_val.ckpt")
            ssl_model = lightning_models.CLAP.load_from_checkpoint(best_ssl_ckpt)
            ssl_model.backbone.remove_projection_head()
        
            tl_model = lightning_models.LinearClassification(
                        backbone = ssl_model.backbone,
                        in_dim = config.SSL["backbone_out_dim"],
                        out_dim = tl_output_dim[dataset],
                        use_batch_norm = config.TL["use_batch_norm"],
                        optim_name = config.TL["optimizer"],
                        lr = tl_lr, 
                        momentum = config.TL["momentum"],
                        weight_decay = config.TL["weight_decay"],
                        n_epochs = config.TL["n_epochs"])

            tl_model = lightning_models.train_lc(
                        linear_model = tl_model,
                        train_loader = tl_train_loader,
                        val_loader = tl_val_loader,
                        test_loader = tl_test_loader,
                        every_n_epochs = config.TL["save_every_n_epochs"],
                        max_epochs = config.TL["n_epochs"],
                        checkpoint_path = tl_dir,
                        precision = config.INFO["precision"],
                        strategy = config.INFO["strategy"],
                        num_nodes = config.INFO["num_nodes"],
                        gpus_per_node = config.INFO["gpus_per_node"],
                        restart = config.SemiSL["restart_training"])


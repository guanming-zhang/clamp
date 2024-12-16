import torch
import os
from model import models
from model import loss_module
from utils import data_utils
import torchvision
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as data
from torch import optim
import configparser
import time
import json
from torch.utils.data.distributed import DistributedSampler
import datetime
#import pathlib
#path = pathlib.Path(__file__).parents[1]/"model"
#sys.path.append(str(path))

def get_est_time_now():
    est_offset = datetime.timedelta(hours=-5)
    est = datetime.timezone(est_offset,name="EST")
    utc_time = datetime.datetime.now(datetime.timezone.utc)
    est_time = utc_time.astimezone(est)
    return est_time,est

def set_random_seed(s:int=137):
    torch.manual_seed(s) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(s)
        torch.cuda.manual_seed_all(s)

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu")
    return device

#####################################
# For Benchmarking
#####################################   
class Timer:
    def __init__(self,process_name = "Process A"):
        self._process_name = process_name
    def __enter__(self):
        self.start_time = time.time()
    def __exit__(self, *args):
        self.end_time = time.time()
        time_diff = self.end_time - self.start_time
        print(f"-------{self._process_name} took {time_diff} sec-----")
########################################
# For reading the input configuration
########################################    
class Config:
    def __init__(self,input_dir:str,default_config_file:str=""):
        input_file= os.path.join(input_dir,'config.ini')
        if not os.path.isfile(input_file):
            raise FileNotFoundError("The input file" + input_file + "does not exist")
        config = configparser.ConfigParser()
        if len(default_config_file) > 0:
            default_config = configparser.ConfigParser()
            default_config.read(default_config_file)

        config.read(input_file)
        self.loc = input_dir

        #----------------convert to properties----------------
        self.INFO = {}
        self.DATA = {}
        self.SSL = {} # self supervised learning
        self.LC = {}  # linear classification
        self.SemiSL = {}  # finetune(semi-supervised learning)
        self.TL = {}  # transfer learning(freeze backbone)
        
        #----------------set the default configuration first ----------
        if len(default_config_file)>0:
            print("Loading default settings...")
            self._set_options(section="INFO",config = default_config)
            self._set_options(section="DATA",config = default_config)
            self._set_options(section="SSL",config = default_config)
            self._set_options(section="LC",config = default_config)
            self._set_options(section="SemiSL",config = default_config)
            self._set_options(section="TL",config = default_config)
        #----------------set the configuration  ----------
        self._set_options(section="INFO",config = config)
        self._set_options(section="DATA",config = config)
        self._set_options(section="SSL",config = config)
        self._set_options(section="LC",config = config)
        self._set_options(section="SemiSL",config = config)
        self._set_options(section="TL",config = config)

        compulsory = {  "INFO":["num_nodes","gpus_per_node"],
                        "DATA":["dataset","augmentations","n_views"],
                        "SSL":["batch_size","backbone","use_projection_head","backbone_out_dim",
                            "optimizer","loss_function","n_epochs"],
                        "LC":["use_batch_norm","batch_size","optimizer","output_dim","n_epochs"]
                        }
        #--------------check information -------------------
        for section in config.sections():
            print(f"[{section}]")
            for key, value in getattr(self,section).items():
                print(f"{key} = {value}")
            print() 

        for section in compulsory:
            if not hasattr(self,section):
                raise ValueError(section + "section is missing in the config.ini")
            else:
                for option in compulsory[section]:
                    if not option in getattr(self,section):
                        raise ValueError(option + " is missing in the [{}] section".format(section))
                if "lr" in getattr(self,section) and "lr_sweep" in getattr(self,section):
                    getattr(self,section).pop("lr")
                    print("lr overrided by lr_sweep!!") 

    def _check_existence(self,str_list,container):
        for s in str_list:
            if not s in container:
                raise KeyError(s + " does not exists")
    def _options_type(self,section:str):
        if section == "INFO":
            options_type = {
            "num_nodes":"int",
            "gpus_per_node":"int",
            "cpus_per_gpu":"int",
            "precision":"string",
            "strategy":"string",
            "fix_random_seed":"boolean",
            "prof_mem":"string"}
        elif section == "DATA":
            options_type = {
            "dataset":"string",
            "imagenet_train_dir":"string",
            "imagenet_val_dir":"string",
            "augmentations":"string_list",
            "n_views":"int",
            "batch_size":"int",
            # for image augmentations
            "crop_size":"int",
            "crop_min_scale":"float",
            "crop_max_scale":"float",
            "jitter_brightness":"float",
            "jitter_contrast":"float",
            "jitter_saturation":"float",
            "jitter_hue":"float",
            "jitter_prob":"float",
            "grayscale_prob":"float",
            "blur_kernel_size":"int",
            "blur_prob":"float",
            "hflip_prob":"float",
            "solarize_prob":"float"
            }
        elif section == "SSL":
            options_type = {
            "batch_size":"int",
            "backbone":"string",
            "use_projection_head":"boolean",
            "proj_dim":"int",
            "proj_out_dim":"int",
            "backbone_out_dim":"int",
            "optimizer":"string",
            "lr":"float",
            "lr_scale":"string",
            "lr_sweep":"float_list",
            "grad_accumulation_steps":"int",
            "momentum":"float",
            "weight_decay":"float",
            "lars_eta":"float",
            "loss_function":"string",
            "lw0":"float",
            "lw1":"float",
            "lw2":"float",
            "pot_pow":"float",
            "rs":"float",
            # tau is for info nce loss
            "tau":"float", 
            "warmup_epochs":"int",
            "n_epochs":"int",
            "save_every_n_epochs":"int",
            "restart_training":"boolean"
            }
        elif section == "LC":
            options_type = {
            "output_dim":"int",
            "use_batch_norm":"boolean",
            "apply_simple_augmentations":"boolean",
            "standardize_to_imagenet":"boolean",
            "loss_function":"string",
            "optimizer":"string",
            "lr":"float",
            "lr_scale":"string",
            "lr_sweep":"float_list",
            "lr_scheduler":"string",
            "momentum":"float",
            "weight_decay":"float",
            "n_epochs":"int",
            "batch_size":"int",
            "save_every_n_epochs":"int",
            "restart_training":"boolean"
            }
        elif section == "SemiSL":
            # Semi-superivsed learning
            options_type = {
                "loss_function":"string",
                "apply_simple_augmentations":"boolean",
                "standardize_to_imagenet":"boolean",
                "optimizer":"string",
                "lr":"float",
                "lr_scale":"string",
                "lr_sweep":"float_list",
                "momentum":"float",
                "weight_decay":"float",
                "n_epochs":"int",
                "batch_size":"int",
                "save_every_n_epochs":"int",
                "restart_training":"boolean"
            }
        elif section == "TL":
            # transfer learning(freeze the backbone)
            options_type = {
                "use_batch_norm":"boolean",
                "standardize_to_imagenet":"boolean",
                "loss_function":"string",
                "optimizer":"string",
                "lr":"float",
                "lr_scale":"string",
                "lr_sweep":"float_list",
                "lr_scheduler":"string",
                "momentum":"float",
                "weight_decay":"float",
                "n_epochs":"int",
                "batch_size":"int",
                "save_every_n_epochs":"int",
                "dataset":"string",
                "restart_training":"boolean"
            }
        return options_type
    
    def _set_options(self,section:str,config:configparser.ConfigParser):
        if not section in config.sections():
            print("[" + section + "]" + "does not exist in the config file")
            return
        options = config.options(section)
        options_type = self._options_type(section)
        for opt in options:
            if not (opt in options_type):
                raise KeyError("[{}] is not a valid key, check the spelling or register it before using".format(opt))
            if options_type[opt] == "int":
                getattr(self,section)[opt] = config[section].getint(opt)
            elif options_type[opt] == "float":
                getattr(self,section)[opt] = config[section].getfloat(opt)
            elif options_type[opt] == "boolean":
                getattr(self,section)[opt] = config[section].getboolean(opt)
            elif options_type[opt] == "string":
                getattr(self,section)[opt] = config[section].get(opt)
            elif options_type[opt] == "string_list":
                getattr(self,section)[opt] = config[section][opt].split(",")
            elif options_type[opt] == "int_list":
                str_list = config[section][opt].split(",")
                getattr(self,section)[opt] = [int(s) for s in str_list]
            elif options_type[opt] == "float_list":
                str_list = config[section][opt].split(",")
                getattr(self,section)[opt] = [float(s) for s in str_list]
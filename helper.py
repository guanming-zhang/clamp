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

#import pathlib
#path = pathlib.Path(__file__).parents[1]/"model"
#sys.path.append(str(path))
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
        print(f"{self._process_name} took {time_diff} sec")
########################################
# For reading the input configuration
########################################    
class Config:
    def __init__(self,input_dir):
        input_file= os.path.join(input_dir,'config.ini')
        if not os.path.isfile(input_file):
            raise FileNotFoundError("The input file" + input_file + "does not exist")
        config = configparser.ConfigParser()
        config.read(input_file)
        self.loc = input_dir
        self.config = config
        compulsory = {  "INFO":["num_gpus"],
                        "DATA":["dataset","augmentations","n_views"],
                        "SSL":["batch_size","backbone","use_projection_header","embedded_dim",
                            "optimizer","loss_function","n_epoch"],
                        "LC":["use_batch_norm","batch_size","optimizer","output_dim","n_epoch"],
                        "IO":["training_mode"]}
        #--------------check information -------------------
        for section in config.sections():
            print(f"[{section}]")
            for key, value in config.items(section):
                print(f"{key} = {value}")
            print() 

        for section in compulsory:
            if not section in config:
                raise ValueError(section + "section is missing in the config.ini")
            else:
                for option in compulsory[section]:
                    if not option in config[section]:
                        raise ValueError(option + " is missing in the [{}] section".format(section))
        #----------------convert to properties----------------
        self.INFO,self.DATA,self.SSL,self.LC,self.IO= {},{},{},{},{}
        self._set_default()
        self._dataset_info()
        self._ssl_info()
        self._lc_info()
        self._io_info()
    def _check_existence(self,str_list,container):
        for s in str_list:
            if not s in container:
                raise KeyError(s + " does not exists")
    def _set_default(self):
        self.config["DEFAULT"] = {
            "batch_size":24,
            "use_projection_header":"no",
            "optimizer":"SGD",
            "crop_size":32,
            "jitter_brightness":0.5,
            "jitter_contrast":0.5,
            "jitter_saturation":0.5,
            "jitter_hue":0.2,
            "jitter_prob":0.8,
            "grayscale_prob":0.8,
            "blur_kernel_size":3,
            "use_batch_norm":"yes"
        }
    def _options_type(self,section:str):
        if section == "INFO":
            options_type = {"num_nodes":"int","gpu_per_node":"int"}
        elif section == "DATA":
            options_type = {
            "dataset":"string",
            "augmentations":"string_list",
            "n_views":"int",
            "batch_size":"int",
            # for image augmentations
            "crop_size":"int",
            "jitter_brightness":"float",
            "jitter_contrast":"float",
            "jitter_saturation":"float",
            "jitter_hue":"float",
            "jitter_prob":"float",
            "grayscale_prob":"float",
            "blur_kernel_size":"int"
            }
        elif section == "SSL":
            options_type == {
            "batch_size":"int",
            "backbone":"string",
            "use_projection_header":"boolean",
            "proj_dim":"int",
            "embedded_dim":"int",
            "optimizer":"string",
            "lr":"float",
            "momentum":"float",
            "weight_decay":"float",
            "lars_eta":"float",
            "loss_function":"string",
            "lw0":"float",
            "lw1":"float",
            "lw2":"float",
            # tau is for info nce loss
            "tau":"float", 
            "warmup_epochs":"int",
            "n_epoch":"int"
            }
        elif section == "LC":
            options_type = {
            "out_dim":"int",
            "use_batch_norm":"boolean",
            "optimizer":"string",
            "lr":"float",
            "momentum":"float",
            "weight_decay":"float",
            "n_epochs":"int"
            }
        elif section == "IO":
            options_type = {
            "output_dir":"string",
            "mode":"string"
            }
        return options_type
    
    def _set_options(self,section:str,options:list):
        options_type = self._options_type(section)
        for opt in options:
            if not (opt in self.options_type):
                raise KeyError("[{}] is not a valid key, check the spelling or register it before using".format(opt))
            if options_type[opt] == "int":
                getattr(self,section)[opt] = self.config[section].getint(opt)
            elif options_type[opt] == "float":
                getattr(self,section)[opt] = self.config[section].getfloat(opt)
            elif options_type[opt] == "boolean":
                getattr(self,section)[opt] = self.config[section].getboolean(opt)
            elif options_type[opt] == "string":
                getattr(self,section)[opt] = self.config[section].get(opt)
            elif options_type[opt] == "string_list":
                getattr(self,section)[opt] = self.config[section][opt].split(",")
            elif options_type[opt] == "int_list":
                str_list = self.config[section][opt].split(",")
                getattr(self,section)[opt] = [int(s) for s in str_list]
            elif options_type[opt] == "float_list":
                str_list = self.config[section][opt].split(",")
                getattr(self,section)[opt] = [float(s) for s in str_list]

    def _general_info(self):
        self._set_options("INFO",self.config.options("INFO"))
    def _dataset_info(self):
        self._set_options("DATA",self.config.options("DATA"))
    def _ssl_info(self):
        self._set_options("SSL",self.config.options("SSL"))
    def _lc_info(self):
        self._set_options("LC",self.config.options("LC"))
    def _io_info(self):
        self._set_options("IO",self.config.options("IO"))



def save_checkpoint(save_dir,model,optimizer,trainer):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'trainer_state_dict':trainer.state_dict()}, save_dir)

def load_checkpoint(load_path,model,optimizer=None,trainer=None,device = "cpu"):
    loaded_data = torch.load(load_path,map_location=device)
    model.load_state_dict(loaded_data["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(loaded_data["optimizer_state_dict"])
    if trainer:
        trainer.load_state_dict(loaded_data["trainer_state_dict"])

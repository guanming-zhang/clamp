import os
import copy
import configparser
from itertools import product
from copy import deepcopy
import subprocess
import time
import shutil
import datetime

class JobManager:
    def __init__(self,default_config_path:str):
        config = configparser.ConfigParser()
        config.read(default_config_path)
        self.base_config = config 
        self.batch_dict = {"NUM_NODES":"1",
                           "GPUS_PER_NODE":"1",
                           "CPUS_PER_TASK":"1",
                           "NTASKS_PER_NODE":"1",
                           "GRES":"gpu",
                           "CONDA_ENV":"dl_env",
                           "TIME":"100:00:00",
                           "MEM_PER_NODE":"6GB",
                           "PYTHON_EXE":"main.py",
                           "ARG1":"[input_dir_path]",
                           "ARG2":default_config_path}
        self.default_comp_res = True
    def print_config(self,config:configparser.ConfigParser):
        for section in config.sections():
            print(f"[{section}]")
            for key, value in config[section].items():
                print(f"{key} = {value}")
            print()
    def set_computation_resource(self,num_nodes:int,gpus_per_node:int,cpus_per_gpu:int,gres:str = "gpu"):
        # update the base config
        self.base_config.set("INFO","num_nodes",str(num_nodes))
        self.base_config.set("INFO","gpus_per_node",str(gpus_per_node))
        self.base_config.set("INFO","cpus_per_gpu",str(cpus_per_gpu))
        # update the batch options
        self.batch_dict["NUM_NODES"] = str(num_nodes)
        self.batch_dict["GPUS_PER_NODE"] = str(gpus_per_node)
        self.batch_dict["CPUS_PER_TASK"] = str(cpus_per_gpu)
        # set the ntasks_per_node = gpus_per_node
        # see https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html
        self.batch_dict["NTASKS_PER_NODE"] = str(gpus_per_node)
        self.batch_dict["GRES"] = gres
        if gpus_per_node*num_nodes == 1:
            self.base_config.set("INFO","strategy","auto")
        self.default_comp_res = False
    
    def generate_config_combinations(self,config_options:dict)->configparser.ConfigParser:
        # Separate sections and values for each option
        sections = config_options.keys()
        options = {section: [dict(zip(config_options[section], values))
                         for values in product(*config_options[section].values())]
               for section in sections}

        # Generate Cartesian product of options across all sections
        all_combinations = product(*options.values())
        configs = []

        # Create a ConfigParser object for each combination
        for combination in all_combinations:
            config = configparser.ConfigParser()
            for section, option_dict in zip(sections, combination):
                config[section] = {key: str(value) for key, value in option_dict.items()}
            configs.append(deepcopy(config))
        return configs

    def create_directory_from_config(self,base_dir:str, config:configparser.ConfigParser,
                                     suffix:str = "",prefix:str = "")->str:
        # Generate a unique subdirectory name based on the hyperparameters
        folder_name_parts = []
        for section, options in config.items():
            if section == "DEFAULT":
                continue
            folder_name_parts.append(f"#{section}#")
            for key, value in options.items():
                folder_name_parts.append(f"{key}-{value}")
        
        # Join all parts to form the folder name
        folder_name = prefix + "-".join(folder_name_parts) + suffix

        # Create the directory using Python
        dir_path = os.path.join(base_dir,folder_name)
        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        print(dir_path)
        return dir_path
    def update_configparser(self,base_config:configparser.ConfigParser, update_config:configparser.ConfigParser):
        # Iterate over all sections in the update_config
        for section in update_config.sections():
            if not base_config.has_section(section):
                raise ValueError(f"Section '{section}' not found in base_config.")
            # Iterate over all options in the section and update base_config
            for key, value in update_config.items(section):
                base_config.set(section, key, value) 
                if not base_config.has_option(section, key):
                    raise ValueError(f"Key '{key}' not found in section '{section}' of base_config.")
    
    def write_config(self,file_path,config):
        with open(file_path, 'w') as configfile:
            config.write(configfile)
    
    def create_sbatch_file(self,batch_dict):
        with open('./submit_batch.ini', 'r') as file:
            fstring = file.read()
        for key in batch_dict:
            fstring = fstring.replace(key, batch_dict[key])
        with open('./submit_batch.sbatch', 'w') as file:
            file.write(fstring)
    
    def submit_sbatch(self,nsleep = 0.05):
        subprocess.run(["sbatch", "submit_batch.sbatch"])
        time.sleep(nsleep)
        subprocess.run(["rm", "submit_batch.sbatch"])
    
    def submit(self,base_dir:str,config_dict:dict,batch_dict:dict,n_repeat:int=1):
        if self.default_comp_res:
            print("The default computional resorcue setup is applied, use set_computation_resource() to reset if needed!")
        configs = self.generate_config_combinations(config_dict)
        print("there are {} configs in total".format(len(configs)))
        os.makedirs(base_dir, exist_ok=True)
        count = 0
        for i in range(n_repeat):
            suffix = "-run-" + f"{i:02}"
            for config in configs:
                dir_path = self.create_directory_from_config(base_dir,config,suffix,prefix="dir"+str(count))
                base_config = copy.deepcopy(self.base_config)
                self.update_configparser(base_config,config)
                self.write_config(os.path.join(dir_path,"config.ini"),base_config)
                base_batch_dict = copy.deepcopy(self.batch_dict)
                base_batch_dict["ARG1"] = dir_path
                base_batch_dict.update(batch_dict)
                self.create_sbatch_file(base_batch_dict)
                self.submit_sbatch()
                count += 1
    def hours_from_starting(self,dir_path):
        def get_est_time_now():
            est_offset = datetime.timedelta(hours=-5)
            est = datetime.timezone(est_offset,name="EST")
            utc_time = datetime.datetime.now(datetime.timezone.utc)
            est_time = utc_time.astimezone(est)
            return est_time,est_time
        with open(os.path.join(dir_path,"starting-time.txt"),'r') as f :
            lines = [line for line in f]
            starting_time = datetime.strptime(lines[-1],"%Y-%m-%d %H:%M:%S")
            time_now,est = get_est_time_now()
        return (time_now - starting_time.astimezone(est)).total_seconds()/3600.0
    
    def continue_prev_submit(self,base_dir:str,batch_dict:dict,hours_before:float=1.0):
        folder_list = os.listdir(base_dir)
        for folder in folder_list:
            folder_path = os.path.join(base_dir,folder)
            flag = True
            if not "run" in folder:
                flag = False
            if os.path.isfile(os.path.join(folder_path,"lc","results.json")):
                flag = False
            if os.path.isfile(os.path.join(folder_path,"starting-time.txt")) and self.hours_from_starting(folder_path) < hours_before:
                flag = False
            if not os.path.isdir(os.path.join(folder_path,"ssl")):
                flag = True
            if not flag:
                continue
            config = configparser.ConfigParser()
            config.read(os.path.join(folder_path,"config.ini"))
            num_nodes = config["INFO"].getint("num_nodes")
            gpus_per_node = config["INFO"].getint("gpus_per_node")
            cpus_per_gpu = config["INFO"].getint("cpus_per_gpu")
            self.set_computation_resource(num_nodes,gpus_per_node,cpus_per_gpu,gres="gpu")
            base_batch_dict = copy.deepcopy(self.batch_dict)
            base_batch_dict.update(batch_dict)
            base_batch_dict["ARG1"] = folder_path

            self.create_sbatch_file(base_batch_dict)
            self.submit_sbatch()       
        
        
        
if __name__ == "__main__":
    ########################################
    # cifar10
    ########################################
    # for resnet+batch size=256+cifat10, need around 3GB mem per GPU, 3GB*gpus_per_node per node
    # around 5 minutes per epoch
    # if batch size is too small or num_cpus is too low then GPU utility will be low
    jm = JobManager("./default_configs/default_config_cifar10.ini")
    options = {"DATA":{"n_views":[8,12,16]},
               "SSL":{"lr":[0.1,0.2],"batch_size":[256],"lw0":[0.5,1.0,1.5],"lw2":[0.5,1.0,1.5]},
               "LC":{"lr":[0.2]}}
    # cpus_per_taks is equivalent to cpus_per_gpu in our setting
    jm.set_computation_resource(num_nodes=1,gpus_per_node=2,cpus_per_gpu=4,gres="gpu")
    #jm.submit("./simulations/cifar10/resnet18/linear/grid_search1",options,{"TIME":"47:55:00","MEM_PER_NODE":"6GB"})
    jm.continue_prev_submit("./simulations/cifar10/resnet18/linear/grid_search1",{"TIME":"30:55:00","MEM_PER_NODE":"6GB"})

    '''
    ########################################
    # imagenet1k
    ########################################
    # for resnet+batch size=256+cifat10, need around 3GB mem per GPU, 3GB*gpus_per_node per node
    # around 5 minutes per epoch
    # if batch size is too small or num_cpus is too low then GPU utility will be low
    jm = JobManager("./default_configs/default_config_imagenet1k.ini")
    options = {"DATA":{"n_views":[8]},
               "SSL":{"lr":[0.1,0.2],"batch_size":[64],"n_epochs":[2],"warmup_epochs":[1],"save_every_n_epochs":[1]},
               "LC":{"lr":[0.2]}}
    # cpus_per_taks is equivalent to cpus_per_gpu in our setting
    jm.set_computation_resource(num_nodes=1,gpus_per_node=2,cpus_per_gpu=4,gres="gpu")
    jm.submit("./simulations/imagenet/linear/test1",options,{"TIME":"12:30:00","MEM_PER_NODE":"16GB"})
    '''
    

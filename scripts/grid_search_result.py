import os
import anaysis_utils
import configparser
import json
import csv
import re
root_dir = "/home/guanming/HPC-Scratch/sig-ml/clap/simulations/cifar10/resnet18/linear/grid_search"
folders = anaysis_utils.get_directories(root_dir)
csv_data = [["dir#","ssl_lr","lc_lr","ssl_bs","lw0","lw2","test_acc1","test_acc5"]]
print(folders)
for folder in folders:
    config = configparser.ConfigParser()
    config.read(os.path.join(folder,"config.ini"))
    config_dict = anaysis_utils.configparser_to_dict(config)
    with open(os.path.join(folder,"lc","results.json"),"r") as f:
        result_dict = json.load(f)
    test_acc1 = result_dict["best_test_acc1"]
    test_acc5 = result_dict["best_test_acc5"]
    match = re.search(r'dir(\d+)', folder)
    if match:
        dir_num = int(match.group(1))

    ssl_lr = config_dict["SSL"]["lr"]
    lw0 = config_dict["SSL"]["lw0"]
    lw2 = config_dict["SSL"]["lw2"]
    ssl_bs = config_dict["SSL"]["batch_size"]
    lc_lr = config_dict["LC"]["lr"]
    csv_line = [dir_num,ssl_lr,lc_lr,ssl_bs,lw0,lw2,test_acc1,test_acc5]
    csv_data.append(csv_line)
print(csv_data)
# write the csv
with open(os.path.join(root_dir,"combined_results.csv"), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

    


            
    

import configparser
import os
def configparser_to_dict(config):
    config_dict = {section: dict(config.items(section)) for section in config.sections()}
    return config_dict

def get_directories(root_dir:str,get_finished:bool = True):
    folder_list = []
    for folder,_,_ in os.walk(root_dir):
        folder_path = os.path.join(root_dir,folder)
        if not "run" in folder:
            continue
        if not get_finished:
            folder_list.append(folder_path)
        elif os.path.isfile(os.path.join(folder_path,"lc","results.json")):
            folder_list.append(folder_path)
    return folder_list
import configparser
import os
import numpy as np
import torch
from typing import Union
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

def get_dist(vecs:Union[np.ndarray, torch.Tensor])->np.ndarray:
    '''
    vecs: is a N*D matrix contains N vectors of D dimension
    return a distance matrix d, d[i,j] is the distance between vector i and j
    '''
    if isinstance(vecs,torch.Tensor):
        vecs = vecs.detach()
        dist = torch.norm(vecs[:,None] - vecs[None,:],dim=-1)
        return dist.detach().cpu().numpy()
    elif isinstance(vecs,np.ndarray):
        diff = vecs[:,None] - vecs[None,:]
        return np.linalg.norm(diff,axis=-1)
    else:
        raise TypeError("input must be an ndarray or tensor")

def get_cosine_sim(vecs:Union[np.ndarray, torch.Tensor],rm_mean=False)->np.ndarray:
    '''
    vecs: is a N*D matrix contains N vectors of D dimension
    return a distance matrix sim, sim[i,j] is the cosine similarity between vector i and j
    '''
    if isinstance(vecs,torch.Tensor):
        vecs = vecs.detach()
        if rm_mean:
            vecs -= torch.mean(vecs,dim=0)
        sim = torch.nn.functional.cosine_similarity(vecs[:,None],vecs[None,:],dim=-1)
        return sim.detach().cpu().numpy()
    elif isinstance(vecs,np.ndarray):
        if rm_mean:
            vecs -= np.mean(vecs,dim=0)
        norm = np.linalg.norm(vecs,axis=-1)
        norm_prod = norm[None,:]*norm[:,None]
        dot = np.matmul(vecs,vecs.transpose())
        return dot/(norm_prod + 1e-6)
    else:
        raise TypeError("input must be an ndarray or tensor")

def get_cov_alignments(covs:Union[np.ndarray, torch.Tensor])->np.ndarray:
    '''
    covs: is a N*D*D matrix contains N covariance matrices
    sim[i,j] = trace(covs[i]*covs[j])/(|covs[i]|*|covs[j]|)
    return a distance matrix sim, sim[i,j] is the cosine similarity between cov i and j
    '''
    if isinstance(covs,torch.Tensor):
        covs = covs.detach()
        mat_prod = torch.einsum("imn,jnm->ij",covs,covs)
        norm = torch.linalg.matrix_norm(covs)
        norm_prod = norm[None,:]*norm[:,None]
        sim = mat_prod/(norm_prod + 1e-6)
        return sim.detach().cpu().numpy()
    elif isinstance(covs,np.ndarray):
        mat_prod = np.einsum("imn,jnm->ij",covs,covs)
        # norm is a vecotor of shape (N,)
        norm = np.linalg.norm(covs,axis=(1,2))
        norm_prod = norm[None,:]*norm[:,None]
        sim = mat_prod/(norm_prod + 1e-6)
        return sim
    else:
        raise TypeError("input must be an ndarray or tensor")
        
def get_cov_traces(covs:Union[np.ndarray, torch.Tensor])->np.ndarray:
    '''
    covs: is a N*D*D matrix contains N covariance matrices
    sim[i,j] = trace(covs[i]*covs[j])/(|covs[i]|*|covs[j]|)
    return a distance matrix sim, sim[i,j] is the cosine similarity between cov i and j
    '''
    if isinstance(covs,torch.Tensor):
        covs = covs.detach()
        trace = torch.einsum("ijj->i",covs)
        return trace.detach().cpu().numpy()
    elif isinstance(covs,np.ndarray):
        trace = np.einsum("ijj->i",covs)
        return trace
    else:
        raise TypeError("input must be an ndarray or tensor")
    

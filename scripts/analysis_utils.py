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

def power_iteration(matraces, num_iterations=100, epsilon=1e-6):
    """
    Finds the principal eigenvector of a matrix using the power iteration method.

    Args:
        matrix (torch.Tensor): The matrix for which to find the principal eigenvector. 
                               Shape: (n, D, D)
        num_iterations (int): Number of iterations for convergence.
        epsilon (float): Convergence tolerance.

    Returns:
        torch.Tensor: The principal eigenvector.
    """
    # Initialize a random vector
    eigens = []
    # Ensure the matrix is square
    n, m = matraces.shape[1],matraces.shape[2]
    if n != m:
        raise ValueError("Matrix must be square for this method.")
    for i in range(matraces.shape[0]):
        vec = torch.rand(n, device=matraces.device)
        vec = vec / vec.norm()  # Normalize the initial vector
        for _ in range(num_iterations):
            # Multiply matrix by the vector
            next_vec = torch.matmul(matraces[i], vec)
            # Normalize the vector
            next_vec = next_vec / next_vec.norm()

            # Check for convergence
            if torch.norm(next_vec - vec) < epsilon:
                break
            vec = next_vec
        eigens.append(vec.reshape(1,vec.shape[0]))
    return torch.cat(eigens,dim=0)
    

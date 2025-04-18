import torch
import torch.nn.functional as F
from torch import linalg as LA
from typing import Tuple
import torch.distributed as dist
class CrossEntropy:
    def __init__(self):
        self.loss_name = "cross_entropy_loss"
        self.loss = torch.nn.CrossEntropyLoss()
    def __call__(self,preds,labels):
        loss = self.loss(preds,labels)
        return loss

class InfoNCELoss:
    def __init__(self,n_views,batch_size,tau):
        self.loss_name = "info_nce_loss"
        self.n_views = n_views
        self.batch_size = batch_size  
        self.tau = tau  
        self.hyper_parameters = {"n_views":n_views,"batch_size":batch_size,"tau":tau}
    def __call__(self,preds,labels):
        sim = F.cosine_similarity(preds[:,None,:],preds[None,:,:],dim=-1)
        mask_self = torch.eye(preds.shape[0],dtype=torch.bool,device=sim.device)
        sim.masked_fill_(mask_self,0.0)
        positive_mask = mask_self.roll(shifts=self.batch_size,dims=0)
        sim /= self.tau
        ll = torch.mean(-sim[positive_mask] + torch.logsumexp(sim,dim=-1))
        return ll

class EllipsoidPackingLoss:
    def __init__(self,n_views:int,batch_size:int,lw0:float=1.0,lw1:float=1.0,lw2:float=1.0,
                 n_pow_iter:int=20,rs:float=2.0,pot_pow:float=2.0,record:bool = False):
        self.n_views = n_views
        self.batch_size = batch_size
        self.n_pow_iter = n_pow_iter # for power iteration
        self.rs = rs # scale of radii
        self.pot_pow = pot_pow # power for the repulsive potential
        self.lw0 = lw0 # loss weight for the ellipsoid size
        self.lw1 = lw1 # loss weight for the repulsion
        self.lw2 = lw2 # loss weight for the alignment
        self.loss_name = "ellipoids_packing_loss"
        self.hyper_parameters = {"n_views":n_views,"batch_size":batch_size,
                                "lw0":lw0,"lw1":lw1,"lw2":lw2,"n_pow_iter":n_pow_iter,"rs":rs}
        self.record = record
        if record:
            self.status = dict()
    def __call__(self,preds,labels):
        # preds is [(V*B),O] dimesional matrix
        com = torch.mean(preds,dim=0)
        # make the center of mass of pres locate at the origin
        preds -= com
        # normalize to make all the preds in the unit sphere
        std = torch.sqrt(torch.sum(preds*preds,dim=0)/(preds.shape[0] - 1.0) + 1e-12)
        preds = preds/std
        # reshape [(V*B),O] shape tensor to shape [V,B,O] 
        preds = torch.reshape(preds,(self.n_views,self.batch_size,preds.shape[-1]))
        # centers.shape = B*O for B ellipsoids
        centers = torch.mean(preds,dim=0)
        # correlation matrix 
        preds -= centers
        
        corr = torch.matmul(torch.permute(preds,(1,2,0)), torch.permute(preds,(1,0,2)))/self.n_views # size B*O*O
        # average radius for each ellipsoid
        # trace for each ellipsoids, t = torch.diagonal(corr,dim1=1,dim2=2).sum(dim=1)
        # t[i] = sum of eigenvalues for ellipsoid i, semi-axial length = sqrt(eigenvalue)
        # average radii = sqrt(sum of eigenvalues/rank) rank = min(n_views,output_dim) 
        # average radii.shape = (B,)
        radii = self.rs*torch.sqrt(torch.diagonal(corr,dim1=1,dim2=2).sum(dim=1)/min(preds.shape[-1],self.n_views)+ 1e-12)
        # loss 1: repulsive loss
        diff = centers[:, None, :] - centers[None, :, :]
        dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-12)
        #add 1e-6 to avoid dividing by zero
        sum_radii = radii[None,:] + radii[:,None] + 1e-6
        nbr_mask = dist_matrix < sum_radii
        self_mask = torch.eye(self.batch_size,dtype=bool,device=preds.device)
        mask = torch.logical_and(nbr_mask,torch.logical_not(self_mask))
        ll = 0.5*((1.0 - dist_matrix[mask]/sum_radii[mask])**self.pot_pow).sum()*self.lw1
        if abs(self.lw0) > 1e-6:
            # loss 0: minimize the size of each ellipsoids
            # to make sure radii =0 and dij = inf is not a valid state 
            ll += self.lw0*torch.sum(radii)
        if abs(self.lw2) > 1e-6:
            # calculate the largest eigenvectors by the [power iteration] method
            # devided by matrix norm to make sure |corr^n_power| not too small, and ~ 1
            corr_norm = torch.linalg.matrix_norm(corr,keepdim=True)
            normalized_corr = corr/(corr_norm + 1e-6).detach()
            corr_pow = torch.stack([torch.matrix_power(normalized_corr[i], self.n_pow_iter) for i in range(corr.shape[0])])
            b0 = torch.rand(preds.shape[-1],device=preds.device)
            eigens = torch.matmul(corr_pow,b0) # size = B*O
            eigens = eigens/(torch.norm(eigens,dim=1,keepdim=True) + 1e-6) 
            # loss 2: alignment loss (1 - cosine-similarity)
            sim = torch.matmul(eigens,eigens.transpose(0,1))**2
            ll += 0.5*(1.0 - torch.square(sim[mask])).sum()*self.lw2
        if self.record:
            self.status["corrs"] = corr.cpu().detach()
            self.status["centers"] = centers.cpu().detach()
            self.status["principle_vec"] = eigens.cpu().detach()
            self.status["preds"] = preds.cpu().detach()
        return ll
    
    

class RepulsiveEllipsoidPackingLossStdNorm:
    def __init__(self,n_views:int,batch_size:int,lw0:float=1.0,lw1:float=1.0,
                 rs:float=2.0,pot_pow:float=2.0,min_margin:float=1e-3):
        self.n_views = n_views
        self.batch_size = batch_size
        self.rs = rs # scale of radii
        self.pot_pow = pot_pow # power for the repulsive potential
        self.lw0 = lw0 
        self.lw1 = lw1 # loss weight for the repulsion
        self.min_margin = min_margin
        self.record = dict()
        self.hyper_parameters = {"n_views":n_views,"batch_size":batch_size,
                                "lw1":lw1,"rs":rs,"min_margin":min_margin}

    def __call__(self,preds,labels):   
        # reshape [(V*B),O] shape tensor to shape [V,B,O] 
        # V-number of views, B-batch size, O-output embedding dim
        preds_local = torch.reshape(preds,(self.n_views,self.batch_size,preds.shape[-1]))
        # get the embedings from all the processes(GPUs) if ddp
        if dist.is_available() and dist.is_initialized():
            ws = dist.get_world_size() # world size
            outputs = [torch.zeros_like(preds_local) for _ in range(ws)]
            dist.all_gather(outputs,preds_local,async_op=False)
            outputs[dist.get_rank()] = preds_local
            # preds is now [V,(B*ws),O]
            preds = torch.cat(outputs,dim=1)
        else:
            preds = preds_local
            ws = 1
        # preds is [V,B*ws,O] dimesional matrix
        com = torch.mean(preds,dim=(0,1))
        # make the center of mass of pres locate at the origin
        preds -= com
        # normalize the vectors by dividing their standard deviation
        std = torch.sqrt(torch.sum(preds*preds,dim=(0,1))/(preds.shape[0]*preds.shape[1] - 1.0) + 1e-12)
        preds = preds/std
        # centers.shape = [B*ws,O] for B*ws ellipsoids
        centers = torch.mean(preds,dim=0)
        # correlation matrix 
        preds -= centers
        # traces[i] = 1/(n-1)*trace(pred[:,i,:]*pred[:,i,:]^T)
        traces = torch.sum(torch.permute(preds,(1,0,2))**2,dim=(1,2))/(self.n_views -1.0)
        # average radius for each ellipsoid
        # trace for each ellipsoids, t = torch.diagonal(corr,dim1=1,dim2=2).sum(dim=1)
        # t[i] = sum of eigenvalues for ellipsoid i, semi-axial length = sqrt(eigenvalue)
        # average radii = sqrt(sum of eigenvalues/rank) rank = min(n_views,output_dim) 
        # average radii.shape = (B,)
        radii = self.rs*torch.sqrt(traces/min(preds.shape[-1],self.n_views)+ 1e-12)
        # loss 1: repulsive loss
        diff = centers[:, None, :] - centers[None, :, :]
        dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-12)
        #add 1e-6 to avoid dividing by zero
        sum_radii = radii[None,:] + radii[:,None] + 1e-6
        nbr_mask = torch.logical_and(dist_matrix < sum_radii,dist_matrix > self.min_margin)
        self_mask = torch.eye(self.batch_size*ws,dtype=bool,device=preds.device)
        mask = torch.logical_and(nbr_mask,torch.logical_not(self_mask))
        ll = 0.5*((1.0 - dist_matrix[mask]/sum_radii[mask])**self.pot_pow).sum()*self.lw1
        # loss 0: minimize the size of each ellipsoids
        ll += self.lw0*torch.sum(radii)
        self.record["radii"] =radii.detach()
        self.record["dist"] = dist_matrix[torch.logical_not(self_mask)].reshape((-1,)).detach()
        self.record["norm_center"] = torch.linalg.norm(centers,dim=-1).detach()
        return ll


class RepulsiveEllipsoidPackingLossUnitNorm:
    def __init__(self,n_views:int,batch_size:int,lw0:float=1.0,lw1:float=1.0,
                 rs:float=2.0,pot_pow:float=2.0,min_margin:float=1e-3,max_range:float=1.5):
        self.n_views = n_views
        self.batch_size = batch_size
        self.rs = rs # scale of radii
        self.pot_pow = pot_pow # power for the repulsive potential
        self.lw0 = lw0 
        self.lw1 = lw1 # loss weight for the repulsion
        self.min_margin = min_margin
        self.max_range = max_range
        self.record = dict()
        self.hyper_parameters = {"n_views":n_views,"batch_size":batch_size,
                                "lw1":lw1,"rs":rs,"min_margin":min_margin}

    def __call__(self,preds,labels):   
        # reshape [(V*B),O] shape tensor to shape [V,B,O] 
        # V-number of views, B-batch size, O-output embedding dim
        preds_local = torch.reshape(preds,(self.n_views,self.batch_size,preds.shape[-1]))
        # get the embedings from all the processes(GPUs) if ddp
        if dist.is_available() and dist.is_initialized():
            ws = dist.get_world_size() # world size
            outputs = [torch.zeros_like(preds_local) for _ in range(ws)]
            dist.all_gather(outputs,preds_local,async_op=False)
            # it is important to set the outputs[rank] to local output, since computational graph is not 
            # copied through different gpus, preds_local preserves the local computational graphs
            outputs[dist.get_rank()] = preds_local
            # preds is now [V,(B*ws),O]
            preds = torch.cat(outputs,dim=1)
        else:
            preds = preds_local
            ws = 1
        # preds is [V,B*ws,O] dimesional matrix
        com = torch.mean(preds,dim=(0,1))
        # make the center of mass of pres locate at the origin
        preds -= com
        # normalize
        preds = torch.nn.functional.normalize(preds,dim=-1)
        # centers.shape = [B*ws,O] for B*ws ellipsoids
        centers = torch.mean(preds,dim=0)
        # correlation matrix 
        preds -= centers
        # traces[i] = 1/(n-1)*trace(pred[:,i,:]*pred[:,i,:]^T)
        traces = torch.sum(torch.permute(preds,(1,0,2))**2,dim=(1,2))/(self.n_views -1.0)
        # average radius for each ellipsoid
        # trace for each ellipsoids, t = torch.diagonal(corr,dim1=1,dim2=2).sum(dim=1)
        # t[i] = sum of eigenvalues for ellipsoid i, semi-axial length = sqrt(eigenvalue)
        # average radii = sqrt(sum of eigenvalues/rank) rank = min(n_views,output_dim) 
        # average radii.shape = (B,)
        radii = self.rs*torch.sqrt(traces/min(preds.shape[-1],self.n_views)+ 1e-12)
        # loss 1: repulsive loss
        diff = centers[:, None, :] - centers[None, :, :]
        dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-12)
        #add 1e-6 to avoid dividing by zero
        sum_radii = radii[None,:] + radii[:,None] + 1e-6
        sum_radii = torch.min(sum_radii,self.max_range*torch.ones_like(sum_radii,device=sum_radii.device))
        nbr_mask = torch.logical_and(dist_matrix < sum_radii,dist_matrix > self.min_margin)
        self_mask = torch.eye(self.batch_size*ws,dtype=bool,device=preds.device)
        mask = torch.logical_and(nbr_mask,torch.logical_not(self_mask))
        ll = 0.5*((1.0 - dist_matrix[mask]/sum_radii[mask])**self.pot_pow).sum()*self.lw1
        # loss 0: minimize the size of each ellipsoids
        ll += self.lw0*torch.sum(radii)
        self.record["radii"] =radii.detach()
        self.record["dist"] = dist_matrix[torch.logical_not(self_mask)].reshape((-1,)).detach()
        self.record["norm_center"] = torch.linalg.norm(centers,dim=-1).detach()
        return ll

class RepulsiveEllipsoidPackingLossStdNormMem:
    def __init__(self,n_views:int,batch_size:int,lw0:float=1.0,lw1:float=1.0,
                 max_mem_size:int=1024,
                 rs:float=2.0,pot_pow:float=2.0):
        self.n_views = n_views
        self.batch_size = batch_size
        self.rs = rs # scale of radii
        self.pot_pow = pot_pow # power for the repulsive potential
        self.lw0 = lw0 
        self.lw1 = lw1 # loss weight for the repulsion
        self.hyper_parameters = {"n_views":n_views,"batch_size":batch_size,
                                "lw1":lw1,"rs":rs}
        self.mem_centers = None
        self.mem_radii = None
        self.max_mem_size = max_mem_size
        self.current_mem_size = 0
    def __call__(self,preds,labels):
        # preds is [(V*B),O] dimesional matrix
        com = torch.mean(preds,dim=0)
        # make the center of mass of pres locate at the origin
        preds -= com
        # normalize the vectors by dividing their standard deviation
        std = torch.sqrt(torch.sum(preds*preds,dim=0)/(preds.shape[0] - 1.0) + 1e-12)
        preds = preds/std
        # reshape [(V*B),O] shape tensor to shape [V,B,O] 
        preds = torch.reshape(preds,(self.n_views,self.batch_size,preds.shape[-1]))
        # centers.shape = B*O for B ellipsoids
        centers = torch.mean(preds,dim=0)
        # correlation matrix 
        preds -= centers
        trace = torch.sum(torch.permute(preds,(1,0,2))**2,dim=(1,2))/(self.n_views - 1.0) 
        # average radius for each ellipsoid
        # trace for each ellipsoids, t = torch.diagonal(corr,dim1=1,dim2=2).sum(dim=1)
        # t[i] = sum of eigenvalues for ellipsoid i, semi-axial length = sqrt(eigenvalue)
        # average radii = sqrt(sum of eigenvalues/rank) rank = min(n_views,output_dim) 
        # average radii.shape = (B,)
        radii = self.rs*torch.sqrt(trace/min(preds.shape[-1],self.n_views)+ 1e-12)
        # loss 1: repulsive loss
        diff = centers[:, None, :] - centers[None, :, :]
        dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-12)
        #add 1e-6 to avoid dividing by zero
        sum_radii = radii[None,:] + radii[:,None] + 1e-6
        nbr_mask = dist_matrix < sum_radii
        self_mask = torch.eye(self.batch_size,dtype=bool,device=preds.device)
        mask = torch.logical_and(nbr_mask,torch.logical_not(self_mask))
        ll = 0.5*((1.0 - dist_matrix[mask]/sum_radii[mask])**self.pot_pow).sum()*self.lw1
        # loss 0: minimize the size of each ellipsoids
        ll += self.lw0*torch.sum(radii)
        # loss from the repulsion between the current ellipsoids and the ones in the memory bank
        if self.current_mem_size >= self.max_mem_size:
            # shape (B,M,O)
            diff = centers[:, None, :] - self.mem_centers[None, :, :]
            # shape (B,M)
            sum_radii = radii[:,None] + self.mem_radii[None,:] + 1e-6
            # shape (B,M)
            dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-12)
            nbr_mask = dist_matrix < sum_radii
            ll+= 0.5*((1.0 - dist_matrix[nbr_mask]/sum_radii[nbr_mask])**self.pot_pow).sum()*self.lw1
        #update the memory bank
        if self.current_mem_size == 0:
            self.mem_centers = centers.detach().clone()
            self.mem_radii = radii.detach().clone()
            self.current_mem_size = centers.shape[0] # batch size
        elif self.current_mem_size < self.max_mem_size:
            self.mem_centers = torch.cat((centers.detach().clone(),self.mem_centers),dim=0)
            self.mem_radii = torch.cat((radii.detach().clone(),self.mem_radii),dim=0)
            self.current_mem_size += centers.shape[0]
        else:
            self.mem_centers = torch.cat((centers.detach().clone(),self.mem_centers),dim=0)
            self.mem_radii = torch.cat((radii.detach().clone(),self.mem_radii),dim=0)
            self.mem_centers = self.mem_centers[:self.max_mem_size]
            self.mem_radii = self.mem_radii[:self.max_mem_size]
            self.current_mem_size = self.max_mem_size
        return ll

class RepulsiveEllipsoidPackingLoss:
    def __init__(self,n_views:int,batch_size:int,lw0:float=1.0,lw1:float=1.0,
                 rs:float=2.0,pot_pow:float=2.0,min_margin:float=1e-3):
        self.n_views = n_views
        self.batch_size = batch_size
        self.rs = rs # scale of radii
        self.pot_pow = pot_pow # power for the repulsive potential
        self.lw0 = lw0 
        self.lw1 = lw1 # loss weight for the repulsion
        self.hyper_parameters = {"n_views":n_views,"batch_size":batch_size,
                                "lw1":lw1,"rs":rs}
        self.min_margin = min_margin
        self.record = dict()
    def __call__(self,preds,labels):   
        # reshape [(V*B),O] shape tensor to shape [V,B,O] 
        # V-number of views, B-batch size, O-output embedding dim
        preds_local = torch.reshape(preds,(self.n_views,self.batch_size,preds.shape[-1]))
        # get the embedings from all the processes(GPUs) if ddp
        if dist.is_available() and dist.is_initialized():
            ws = dist.get_world_size() # world size
            outputs = [torch.zeros_like(preds_local) for _ in range(ws)]
            dist.all_gather(outputs,preds_local,async_op=False)
            outputs[dist.get_rank()] = preds_local
            # preds is now [V,(B*ws),O]
            preds = torch.cat(outputs,dim=1)
        else:
            preds = preds_local
            ws = 1
        # preds is [V,B*ws,O] dimesional matrix
        com = torch.mean(preds,dim=(0,1))
        # make the center of mass of pres locate at the origin
        preds -= com
        # centers.shape = [B*ws,O] for B*ws ellipsoids
        centers = torch.mean(preds,dim=0)
        # correlation matrix 
        preds -= centers
        # traces[i] = 1/(n-1)*trace(pred[:,i,:]*pred[:,i,:]^T)
        traces = torch.sum(torch.permute(preds,(1,0,2))**2,dim=(1,2))/(self.n_views -1.0)
        # average radius for each ellipsoid
        # trace for each ellipsoids, t = torch.diagonal(corr,dim1=1,dim2=2).sum(dim=1)
        # t[i] = sum of eigenvalues for ellipsoid i, semi-axial length = sqrt(eigenvalue)
        # average radii = sqrt(sum of eigenvalues/rank) rank = min(n_views,output_dim) 
        # average radii.shape = (B,)
        radii = self.rs*torch.sqrt(traces/min(preds.shape[-1],self.n_views)+ 1e-12)
        # loss 1: repulsive loss
        diff = centers[:, None, :] - centers[None, :, :]
        dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-12)
        #add 1e-6 to avoid dividing by zero
        sum_radii = radii[None,:] + radii[:,None] + 1e-6
        nbr_mask = torch.logical_and(dist_matrix < sum_radii,dist_matrix > self.min_margin)
        self_mask = torch.eye(self.batch_size*ws,dtype=bool,device=preds.device)
        mask = torch.logical_and(nbr_mask,torch.logical_not(self_mask))
        ll = 0.5*((1.0 - dist_matrix[mask]/sum_radii[mask])**self.pot_pow).sum()*self.lw1
        # loss 0: minimize the size of each ellipsoids
        ll += self.lw0*torch.sum(radii)
        self.record["radii"] =radii.detach()
        self.record["dist"] = dist_matrix[torch.logical_not(self_mask)].reshape((-1,)).detach()
        self.record["norm_center"] = torch.linalg.norm(centers,dim=-1).detach()
        return ll

class RepulsiveLoss:
    def __init__(self,n_views:int,batch_size:int,lw0:float=1.0,lw1:float=1.0,
                 rs:float=2.0,min_margin:float=1e-3):
        self.n_views = n_views
        self.batch_size = batch_size
        self.rs = rs # scale of radii
        self.lw0 = lw0 
        self.lw1 = lw1 # loss weight for the repulsion
        self.min_margin = min_margin
        self.hyper_parameters = {"n_views":n_views,"batch_size":batch_size,
                                "lw1":lw1,"rs":rs,"min_margin":min_margin}
        self.record = dict()
    def __call__(self,preds,labels):   
        # reshape [(V*B),O] shape tensor to shape [V,B,O] 
        # V-number of views, B-batch size, O-output embedding dim
        preds_local = torch.reshape(preds,(self.n_views,self.batch_size,preds.shape[-1]))
        # get the embedings from all the processes(GPUs) if ddp
        if dist.is_available() and dist.is_initialized():
            ws = dist.get_world_size() # world size
            outputs = [torch.zeros_like(preds_local) for _ in range(ws)]
            dist.all_gather(outputs,preds_local,async_op=False)
            outputs[dist.get_rank()] = preds_local
            # preds is now [V,(B*ws),O]
            preds = torch.cat(outputs,dim=1)
        else:
            preds = preds_local
            ws = 1
        # preds is [V,B*ws,O] dimesional matrix
        com = torch.mean(preds,dim=(0,1))
        # make the center of mass of pres locate at the origin
        preds -= com
        # centers.shape = [B*ws,O] for B*ws ellipsoids
        centers = torch.mean(preds,dim=0)
        # correlation matrix 
        preds -= centers
        # traces[i] = 1/(n-1)*trace(pred[:,i,:]*pred[:,i,:]^T)
        traces = torch.sum(torch.permute(preds,(1,0,2))**2,dim=(1,2))/(self.n_views -1.0)
        # average radius for each ellipsoid
        # trace for each ellipsoids, t = torch.diagonal(corr,dim1=1,dim2=2).sum(dim=1)
        # t[i] = sum of eigenvalues for ellipsoid i, semi-axial length = sqrt(eigenvalue)
        # average radii = sqrt(sum of eigenvalues/rank) rank = min(n_views,output_dim) 
        # average radii.shape = (B,)
        radii = self.rs*torch.sqrt(traces/min(preds.shape[-1],self.n_views)+ 1e-12)
        # loss 1: repulsive loss
        diff = centers[:, None, :] - centers[None, :, :]
        dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-12)
        self_mask = torch.eye(self.batch_size*ws,dtype=bool,device=preds.device)
        mask = torch.logical_and(torch.logical_not(self_mask),dist_matrix > self.min_margin)
        ll = - torch.sum(dist_matrix[mask])*self.lw1
        # loss 0: minimize the size of each ellipsoids
        ll += self.lw0*torch.sum(radii)
        self.record["radii"] =radii.detach()
        self.record["dist"] = dist_matrix[torch.logical_not(self_mask)].reshape((-1,)).detach()
        self.record["norm_center"] = torch.linalg.norm(centers,dim=-1).detach()
        return ll

class MMCR_Loss(torch.nn.Module):
    def __init__(self, n_views: int,batch_size:int):
        super(MMCR_Loss, self).__init__()
        self.n_views = n_views
        self.batch_size = batch_size
        self.distribured = dist.is_available() and dist.is_initialized()

    def forward(self, z: torch.Tensor,labels:torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # z is [(V*B),O] dimesional matrix
        z = F.normalize(z, dim=-1)
        z = torch.reshape(z,(self.n_views,self.batch_size,z.shape[-1]))
        # gather across devices into list
        if self.distribured:
            ws = torch.distributed.get_world_size()
            z_list = [
                torch.zeros_like(z)
                for _ in range(ws)
            ]
            torch.distributed.all_gather(z_list, z, async_op=False)
            z_list[torch.distributed.get_rank()] = z
            # append all
            z = torch.cat(z_list)
        else:
            ws = 1
        centroids = torch.mean(z, dim=0)
        global_nuc = torch.linalg.svdvals(centroids).sum()
        loss = - global_nuc

        return loss

class LogRepulsiveEllipsoidPackingLossUnitNorm:
    def __init__(self,n_views:int,batch_size:int,lw0:float=1.0,lw1:float=1.0,
                 rs:float=2.0,pot_pow:float=2.0,min_margin:float=1e-3,max_range:float=1.5):
        self.n_views = n_views
        self.batch_size = batch_size
        self.rs = rs # scale of radii
        self.pot_pow = pot_pow # power for the repulsive potential
        self.lw0 = lw0 
        self.lw1 = lw1 # loss weight for the repulsion
        self.min_margin = min_margin
        self.max_range = max_range
        self.record = dict()
        self.hyper_parameters = {"n_views":n_views,"batch_size":batch_size,
                                "lw1":lw1,"rs":rs,"min_margin":min_margin}

    def __call__(self,preds,labels):   
        # reshape [(V*B),O] shape tensor to shape [V,B,O] 
        # V-number of views, B-batch size, O-output embedding dim
        preds_local = torch.reshape(preds,(self.n_views,self.batch_size,preds.shape[-1]))
        # get the embedings from all the processes(GPUs) if ddp
        if dist.is_available() and dist.is_initialized():
            ws = dist.get_world_size() # world size
            outputs = [torch.zeros_like(preds_local) for _ in range(ws)]
            dist.all_gather(outputs,preds_local,async_op=False)
            # it is important to set the outputs[rank] to local output, since computational graph is not 
            # copied through different gpus, preds_local preserves the local computational graphs
            outputs[dist.get_rank()] = preds_local
            # preds is now [V,(B*ws),O]
            preds = torch.cat(outputs,dim=1)
        else:
            preds = preds_local
            ws = 1
        # preds is [V,B*ws,O] dimesional matrix
        com = torch.mean(preds,dim=(0,1))
        # make the center of mass of pres locate at the origin
        preds -= com
        # normalize
        preds = torch.nn.functional.normalize(preds,dim=-1)
        # centers.shape = [B*ws,O] for B*ws ellipsoids
        centers = torch.mean(preds,dim=0)
        # correlation matrix 
        preds -= centers
        # traces[i] = 1/(n-1)*trace(pred[:,i,:]*pred[:,i,:]^T)
        traces = torch.sum(torch.permute(preds,(1,0,2))**2,dim=(1,2))/(self.n_views -1.0)
        # average radius for each ellipsoid
        # trace for each ellipsoids, t = torch.diagonal(corr,dim1=1,dim2=2).sum(dim=1)
        # t[i] = sum of eigenvalues for ellipsoid i, semi-axial length = sqrt(eigenvalue)
        # average radii = sqrt(sum of eigenvalues/rank) rank = min(n_views,output_dim) 
        # average radii.shape = (B,)
        radii = self.rs*torch.sqrt(traces/min(preds.shape[-1],self.n_views)+ 1e-12)
        # loss 1: repulsive loss
        diff = centers[:, None, :] - centers[None, :, :]
        dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-12)
        #add 1e-6 to avoid dividing by zero
        sum_radii = radii[None,:] + radii[:,None] + 1e-6
        sum_radii = torch.min(sum_radii,self.max_range*torch.ones_like(sum_radii,device=sum_radii.device))
        nbr_mask = torch.logical_and(dist_matrix < sum_radii,dist_matrix > self.min_margin)
        self_mask = torch.eye(self.batch_size*ws,dtype=bool,device=preds.device)
        mask = torch.logical_and(nbr_mask,torch.logical_not(self_mask))
        ll = 0.5*((1.0 - dist_matrix[mask]/sum_radii[mask])**self.pot_pow).sum()*self.lw1
        # loss 0: minimize the size of each ellipsoids
        ll += self.lw0*torch.sum(radii)
        self.record["radii"] =radii.detach()
        self.record["dist"] = dist_matrix[torch.logical_not(self_mask)].reshape((-1,)).detach()
        self.record["norm_center"] = torch.linalg.norm(centers,dim=-1).detach()
        return torch.log(ll + 1e-6)

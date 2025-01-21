import torch
import torch.nn.functional as F
from torch import linalg as LA
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
                 n_pow_iter:int=20,rs:float=2.0,pot_pow:float=2.0,margin:float=1e-7,record:bool = False):
        self.n_views = n_views
        self.batch_size = batch_size
        self.n_pow_iter = n_pow_iter # for power iteration
        self.rs = rs # scale of radii
        self.pot_pow = pot_pow # power for the repulsive potential
        self.lw0 = lw0 # loss weight for the ellipsoid size
        self.lw1 = lw1 # loss weight for the repulsion
        self.lw2 = lw2 # loss weight for the alignment
        self.margin = margin # no replsion if the distance between two elliposids < margins 
        self.loss_name = "ellipoids_packing_loss"
        self.hyper_parameters = {"n_views":n_views,"batch_size":batch_size,
                                "lw0":lw0,"lw1":lw1,"lw2":lw2,"n_pow_iter":n_pow_iter,"rs":rs,
                                "margin":margin}
        self.record = record
        if record:
            self.status = dict()
    def __call__(self,preds,labels):
        # preds is [(V*B),O] dimesional matrix
        com = torch.mean(preds,dim=0)
        # make the center of mass of pres locate at the origin
        preds -= com
        # normalize to make all the preds in the unit sphere
        preds_norm_max = torch.max(torch.linalg.vector_norm(preds,dim=1)) + 1e-6
        preds = preds/preds_norm_max
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
        nbr_mask = torch.logical_and(dist_matrix < sum_radii,dist_matrix>self.margin)
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
                 rs:float=2.0,pot_pow:float=2.0,margin:float=1e-7):
        self.n_views = n_views
        self.batch_size = batch_size
        self.rs = rs # scale of radii
        self.pot_pow = pot_pow # power for the repulsive potential
        self.lw0 = lw0 
        self.lw1 = lw1 # loss weight for the repulsion
        self.margin = margin # no replsion if the distance between two elliposids < margins 
        self.hyper_parameters = {"n_views":n_views,"batch_size":batch_size,
                                "lw1":lw1,"rs":rs,"margin":margin}
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
        trace = torch.sum(torch.permute(preds,(1,0,2))**2,dim=(1,2))/self.n_views 
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
        nbr_mask = torch.logical_and(dist_matrix < sum_radii,dist_matrix>self.margin)
        self_mask = torch.eye(self.batch_size,dtype=bool,device=preds.device)
        mask = torch.logical_and(nbr_mask,torch.logical_not(self_mask))
        ll = 0.5*((1.0 - dist_matrix[mask]/sum_radii[mask])**self.pot_pow).sum()*self.lw1
        # loss 0: minimize the size of each ellipsoids
        ll += self.lw0*torch.sum(radii)
        return ll

class PackingLoss:
    def __init__(self,n_views:int,batch_size:int,lw1:float=1.0,lw2:float=1.0,
                 rs:float=2.0,pot_pow:float=2.0,margin:float=1e-7,record:bool = False):
        self.n_views = n_views
        self.batch_size = batch_size
        self.rs = rs # scale of radii
        self.pot_pow = pot_pow # power for the repulsive potential
        self.lw1 = lw1 # loss weight for the repulsion
        self.lw2 = lw2 # loss weight for the alignment
        self.margin = margin # no replsion if the distance between two elliposids < margins 
        self.loss_name = "ellipoids_packing_loss"
        self.hyper_parameters = {"n_views":n_views,"batch_size":batch_size,
                                "lw1":lw1,"lw2":lw2,"rs":rs,
                                "margin":margin}
        self.record = record
        if record:
            self.status = dict()
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
        # covariance matrix
        cov = torch.matmul(torch.permute(preds,(1,2,0)), torch.permute(preds,(1,0,2)))/self.n_views # size B*O*O
        # average radius for each ellipsoid
        # trace for each ellipsoids, t = torch.diagonal(corr,dim1=1,dim2=2).sum(dim=1)
        # t[i] = sum of eigenvalues for ellipsoid i, semi-axial length = sqrt(eigenvalue)
        # average radii = sqrt(sum of eigenvalues/rank) rank = min(n_views,output_dim) 
        # average radii.shape = (B,)
        radii = self.rs*torch.sqrt(torch.diagonal(cov,dim1=1,dim2=2).sum(dim=1)/min(preds.shape[-1],self.n_views)+ 1e-12)
        # loss 1: repulsive loss
        diff = centers[:, None, :] - centers[None, :, :]
        dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-12)
        #add 1e-6 to avoid dividing by zero
        sum_radii = radii[None,:] + radii[:,None]
        nbr_mask = torch.logical_and(dist_matrix < sum_radii,dist_matrix>self.margin)
        self_mask = torch.eye(self.batch_size,dtype=bool,device=preds.device)
        mask = torch.logical_and(nbr_mask,torch.logical_not(self_mask))
        ll = 0.5*((1.0 - dist_matrix[mask]/sum_radii[mask])**self.pot_pow).sum()*self.lw1
        # loss 2: alignment loss
        cov_norm = torch.linalg.matrix_norm(cov,keepdim=True) + 1e-6
        normalized_cov = cov/cov_norm
        # sim is similar to cos^2(theta)
        sim = torch.einsum("mij,nji->mn",normalized_cov,normalized_cov)
        ll += 0.5*(1.0 - sim[mask]).sum()*self.lw2
        return ll
import pytorch_lightning as pl
from torch import optim
import torch.utils
import torch.utils.data
from . import loss_module
from . import models
from . import lars
from . import lr_scheduler
import torch
import torch.nn.functional as F 
import os
import re
import subprocess
import json
import torch.distributed as dist
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger


#####################################################
#  Memory profiling
#####################################################
# To profile the memory
# Keep a max of 100,000 alloc/free events in the recorded history
# leading up to the snapshot.
# see https://pytorch.org/blog/understanding-gpu-memory-1/
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000

def start_record_memory_history() -> None:
   if not torch.cuda.is_available():
       print("CUDA unavailable. Not recording memory history")
       raise ValueError("CUDA unavailable. Not recording memory history")
   torch.cuda.memory._record_memory_history(
       max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
   )

def stop_record_memory_history() -> None:
    if not torch.cuda.is_available():
       print("CUDA unavailable. Not recording memory history")
       raise ValueError("CUDA unavailable. Not recording memory history")
    torch.cuda.memory._record_memory_history(enabled=None)

def export_memory_snapshot(file_path:str) -> None:
   if not torch.cuda.is_available():
       print("CUDA unavailable. Not recording memory history")
       raise ValueError("CUDA unavailable. Not recording memory history")
   try:
       print(f"Saving snapshot to local file:" + file_path )
       torch.cuda.memory._dump_snapshot(file_path)
   except Exception as e:
       print(f"Failed to capture memory snapshot {e}")
       return
#####################################################
#  Helper functions
#####################################################
def get_top_n_latest_checkpoints(directory, n):
    # Regular expression to extract epoch number from filename
    pattern = re.compile(r".*-epoch=(\d+)\.ckpt$")
    # List all files in the directory
    files = os.listdir(directory)
    
    # Create a list to store filenames and corresponding epoch numbers
    epoch_files = []
    for file in files:
        match = pattern.match(file)
        if match:
            # Extract the epoch number and convert to int
            epoch = int(match.group(1))
            epoch_files.append((epoch, file))
    
    # Sort files by epoch number in descending order and get the top N
    top_n_files = sorted(epoch_files, key=lambda x: x[0], reverse=True)[:n]
    # Return the filenames of the top N files
    return [os.path.join(directory,file) for _, file in top_n_files]


#####################################################
#  Self-supervise learning
#####################################################
class CLAP(pl.LightningModule):
    def __init__(self,backbone_name:str,prune:bool,use_projection_head:bool,proj_dim:list,proj_out_dim:int,
                 loss_name:str,
                 optim_name:str,scheduler_name:str,lr:float,momentum:float,weight_decay:float,eta:float,
                 warmup_epochs:int,n_epochs:int,restart_epochs:int=-1,exclude_bn_bias_from_weight_decay:bool=True,
                 n_views:int=4,batch_size:int=256,lw0:float=0.0,lw1:float=1.0,lw2:float=0.0,
                 max_mem_size:int=1024,n_pow_iter:int=20,rs:float=2.0,pot_pow:float=2.0):
        super().__init__()
        self.backbone = models.BackboneNet(backbone_name,prune,use_projection_head,proj_dim,proj_out_dim)
        if loss_name == "EllipsoidPackingLoss":
            self.loss_fn = loss_module.EllipsoidPackingLoss(n_views,batch_size,lw0,lw1,lw2,n_pow_iter,rs,pot_pow)
            print("max_mem_size is dummy for " + loss_name)
        elif loss_name == "RepulsiveEllipsoidPackingLossStdNorm":
            self.loss_fn = loss_module.RepulsiveEllipsoidPackingLossStdNorm(n_views,batch_size,lw0,lw1,rs,pot_pow)
            print("max_mem_size is dummy for " + loss_name)
            print("lw2 is dummy for " + loss_name)
        elif loss_name == "RepulsiveEllipsoidPackingLossUnitNorm":
            self.loss_fn = loss_module.RepulsiveEllipsoidPackingLossUnitNorm(n_views,batch_size,lw0,lw1,rs,pot_pow)
            print("max_mem_size is dummy for " + loss_name)
            print("lw2 is dummy for " + loss_name)
        elif loss_name == "LogRepulsiveEllipsoidPackingLossUnitNorm":
            self.loss_fn = loss_module.LogRepulsiveEllipsoidPackingLossUnitNorm(n_views,batch_size,lw0,lw1,rs,pot_pow)
            print("max_mem_size is dummy for " + loss_name)
            print("lw2 is dummy for " + loss_name)
        elif loss_name == "RepulsiveEllipsoidPackingLossStdNormMem":
            self.loss_fn = loss_module.RepulsiveEllipsoidPackingLossStdNormMem(n_views,batch_size,lw0,lw1,max_mem_size,rs,pot_pow)
        elif loss_name == "RepulsiveEllipsoidPackingLoss":
            self.loss_fn = loss_module.RepulsiveEllipsoidPackingLoss(n_views,batch_size,lw0,lw1,rs,pot_pow)
            print("max_mem_size is dummy for " + loss_name)
            print("lw2 is dummy for " + loss_name)
        elif loss_name == "RepulsiveLoss":
            self.loss_fn = loss_module.RepulsiveLoss(n_views,batch_size,lw0,lw1,rs)
            print("max_mem_size is dummy for " + loss_name)
            print("lw2 is dummy for " + loss_name)
        elif loss_name == "MMCR_Loss":
            self.loss_fn = loss_module.MMCR_Loss(n_views,batch_size)
        
        self.train_epoch_loss = []  # To store epoch loss for training
        self.train_step_outputs = []
        self.val_step_outputs = []
        # all the hyperparameters are added as attributes to this class
        self.save_hyperparameters()
    def remove_weightdecay_for_bias_and_bn(self):
        decay = []
        no_decay = []
        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue
            # Exclude bias and batchnorm params(dim=1) from weight decay
            # see https://github.com/facebookresearch/barlowtwins/blob/main/main.py 
            # or https://github.com/vturrisi/solo-learn/blob/main/solo/utils/misc.py for references
            # note that lars normalization is bypassed if weight_decay == 0
            # see https://lightning-flash.readthedocs.io/en/stable/_modules/flash/core/optimizers/lars.html#LARS
            # or lars.py for more details
            if param.ndim == 1:
                no_decay.append(param)
            else:
                decay.append(param)

        return [
            {'params': decay, 'weight_decay': self.hparams.weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}]
    def configure_optimizers(self):
        if self.hparams.exclude_bn_bias_from_weight_decay:
            param_groups = self.remove_weightdecay_for_bias_and_bn()
        else:
            param_groups = [{"params":self.backbone.parameters(),'weight_decay': self.hparams.weight_decay}]
        if self.hparams.optim_name == "SGD":
            optimizer = optim.SGD(params=param_groups,
                                  lr=self.hparams.lr,
                                  momentum=self.hparams.momentum,
                                  nesterov=True)
        elif self.hparams.optim_name == "Adam":
            optimizer = optim.Adam(params=param_groups,
                                  lr=self.hparams.lr)
        elif self.hparams.optim_name == "LARS":
            optimizer = lars.LARS(params=param_groups,
                                  lr=self.hparams.lr,
                                  trust_coefficient = self.hparams.eta,
                                  momentum=self.hparams.momentum)
        else:
            raise NotImplementedError("optimizer:"+ self.optimizer +" not implemented")

        if self.hparams.scheduler_name == "cosine-warmup":
            #linear = optim.lr_scheduler.LinearLR(optimizer,total_iters=self.hparams.warmup_epochs)
            #cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=self.hparams.n_epochs - self.hparams.warmup_epochs)
            #scheduler = optim.lr_scheduler.SequentialLR(optimizer,schedulers=[linear, cosine], milestones=[self.hparams.warmup_epochs])
            scheduler = lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer,warmup_epochs=self.hparams.warmup_epochs,max_epochs=self.hparams.n_epochs)
        elif self.hparams.scheduler_name == "cosine-restart":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.hparams.restart_epochs)
        elif self.hparams.scheduler_name == "multi_step":
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[int(self.hparams.n_epochs*0.6),
                                                                int(self.hparams.n_epochs*0.8)],
                                                    gamma=0.1)
        else:
            return [optimizer]

        return [optimizer],[scheduler]
    
    def training_step(self, batch, batch_idx):
        imgs,labels = batch
        imgs = torch.cat(imgs,dim=0)
        preds = self.backbone(imgs)
        # the labels are dummy since label is not used in ssl
        loss = self.loss_fn(preds,None)
        self.train_step_outputs.append(loss.detach())
        self.log('train_iteration_loss', loss.detach(), prog_bar=True,sync_dist=True)  # Log iteration loss
        self.log_histogram()
        return loss
    def on_train_epoch_end(self):
        # measure the norm of the gradient
        avg_loss = torch.stack([x for x in self.train_step_outputs]).mean()  # Compute the average loss for the epoch
        self.log('train_epoch_loss', avg_loss, prog_bar=True,sync_dist=True)  # Log epoch loss
        #self.log('grad_norm', total_norm, prog_bar=True,sync_dist=True) 
        # refresh the iteration loss at the end of every epoch
        self.train_step_outputs = []
        # Save epoch loss for future reference
        self.train_epoch_loss.append(avg_loss.item())
    def on_after_backward(self):
        # Calculate the total gradient norm for all parameters
        convnet_norm = 0.0
        convnet_grad_norm = 0.0
        head_norm = 0.0
        head_grad_norm = 0.0
        for p in self.backbone.net.parameters():
            if p.grad is not None:
                # Calculate the norm for each parameter
                param_norm = p.data.norm(2)
                grad_norm = p.grad.data.norm(2)
                convnet_grad_norm += grad_norm.item() ** 2
                convnet_norm += param_norm.item()**2
        
        for p in self.backbone.projection_head.parameters():
            if p.grad is not None:
                # Calculate the norm for each parameter
                param_norm = p.data.norm(2)
                grad_norm = p.grad.data.norm(2)
                head_grad_norm += grad_norm.item() ** 2
                head_norm += param_norm.item()**2

        convnet_grad_norm = convnet_grad_norm ** 0.5
        convnet_norm = convnet_norm ** 0.5
        head_grad_norm = head_grad_norm ** 0.5
        head_norm = head_norm ** 0.5
        # Log the gradient norm; this can be viewed in TensorBoard or your logger
        self.log('convnet_grad_norm', convnet_grad_norm, prog_bar=False)
        self.log('convnet_param_norm', convnet_norm, prog_bar=False)
        self.log('head_grad_norm', head_grad_norm, prog_bar=False)
        self.log('head_param_norm', head_norm, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        imgs = torch.cat(imgs,dim=0)
        preds = self.backbone(imgs)
        preds = torch.reshape(preds,(self.hparams.n_views,self.hparams.batch_size,preds.shape[-1]))
        # get the embedings from all the processes(GPUs) if ddp
        if dist.is_available() and dist.is_initialized():
            ws = dist.get_world_size() # world size
            outputs = [torch.zeros_like(preds) for _ in range(ws)]
            dist.all_gather(outputs,preds,async_op=False)
            # preds is now [V,(B*ws),O]
            preds = torch.cat(outputs,dim=1)
        else:
            ws = 1
        ####### measure the validation accuracy by point to cluster distance
        # preds is [V,B*ws,O] dimesional matrix
        com = torch.mean(preds,dim=(0,1))
        # make the center of mass of pres locate at the origin
        preds -= com
        # normalize the vectors by dividing their standard deviation
        std = torch.sqrt(torch.sum(preds*preds,dim=(0,1))/(preds.shape[0]*preds.shape[1] - 1.0) + 1e-12)
        if "StdNorm" in self.hparams.loss_name:
            preds = preds/std
        elif "UnitNorm" in self.hparams.loss_name:
            preds = torch.nn.functional.normalize(preds,dim=-1)
        elif "MMCR" in self.hparams.loss_name:
            preds = torch.nn.functional.normalize(preds,dim=-1)
        # centers.shape = [(B*ws),O] for B*ws ellipsoids
        centers = torch.mean(preds,dim=0)
        # point to cluster distance matrix (V,B*ws,B*ws)
        diff = preds[:,:,None,:] - centers[None,None,:,:]
        pt_cluster_dist = torch.sum(diff**2,dim=-1)    
        # nearest (V,B), nearest[1,2] = 4 
        # nearest[1,2] = 4 means the the nearest cluster to
        # the 1th view of in cluster 2 is cluster 4 
        nearest = torch.argmin(pt_cluster_dist,dim=-1)
        correct = nearest == torch.arange(self.hparams.batch_size*ws,device=nearest.device).repeat(self.hparams.n_views,1)
        acc = (correct.sum()/(self.hparams.n_views*self.hparams.batch_size*ws)).float()
        ####### measure the average distance and radius
        # correlation matrix 
        preds -= centers
        corr = torch.matmul(torch.permute(preds,(1,2,0)), torch.permute(preds,(1,0,2)))/self.hparams.n_views # size B*O*O
        # average radii.shape = (B,)
        radii = torch.sqrt(torch.diagonal(corr,dim1=1,dim2=2).sum(dim=1)/min(preds.shape[-1],self.hparams.n_views)+ 1e-12)
        diff = centers[:, None, :] - centers[None, :, :]
        dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-12)
        # add 1e-6 to avoid dividing by zero
        sum_radii = self.hparams.rs*(radii[None,:] + radii[:,None] + 1e-6)
        nbr_mask = dist_matrix < sum_radii*0.99
        num_nbr = torch.sum(nbr_mask,dim=1)
        activity = torch.sum(num_nbr > 0)/(self.hparams.batch_size*ws)
        mean_radius = torch.mean(radii)
        mean_nbr = torch.mean(num_nbr.float())
        mean_dist = torch.sum(dist_matrix)/(self.hparams.batch_size*ws*(self.hparams.batch_size*ws - 1.0))
        # detect complete collapse and dimensional collapse
        # average std for center points in each direction
        raw_mean_std = torch.mean(std)
        mean_std = torch.mean(torch.std(centers,dim=0))
        sig_vals = torch.linalg.svdvals(centers.float())
        # use 1e-3 as the threshold to estimate the rank and filter out small eigenvalues
        mean_sigval = torch.mean(sig_vals) + 1e-6
        std_sigval = torch.std(sig_vals)
        
        self.val_step_outputs.append({"val_acc":acc, 
                                    "val_radius":mean_radius,
                                    "val_activity":activity,
                                    "val_num_nbr":mean_nbr,
                                    "val_dist":mean_dist,
                                    "val_raw_std":raw_mean_std,
                                    "val_std":mean_std,
                                    "val_sig_ratio":std_sigval/mean_sigval})
        return acc
    
    def on_validation_epoch_end(self):
        val_radius =  torch.stack([x["val_radius"] for x in self.val_step_outputs]).mean() 
        val_activity = torch.stack([x["val_activity"] for x in self.val_step_outputs]).mean()
        val_num_nbr = torch.stack([x["val_num_nbr"] for x in self.val_step_outputs]).mean()
        val_acc = torch.stack([x["val_acc"] for x in self.val_step_outputs]).mean()
        val_dist = torch.stack([x["val_dist"] for x in self.val_step_outputs]).mean()
        val_raw_std = torch.stack([x["val_raw_std"] for x in self.val_step_outputs]).mean()
        val_std = torch.stack([x["val_std"] for x in self.val_step_outputs]).mean()
        val_sig_ratio = torch.stack([x["val_sig_ratio"] for x in self.val_step_outputs]).mean()
        self.log('val_acc',val_acc,prog_bar=True,sync_dist=True)
        self.log('val_radius',val_radius,prog_bar=True,sync_dist=True)
        self.log('val_activity',val_activity,prog_bar=True,sync_dist=True)
        self.log('val_num_nbr',val_num_nbr,prog_bar=True,sync_dist=True)
        self.log("val_dist",val_dist,prog_bar=True,sync_dist=True)
        self.log("val_raw_std",val_raw_std,prog_bar=True,sync_dist=True)
        self.log("val_std",val_std,prog_bar=True,sync_dist=True)
        self.log("val_sig_ratio",val_sig_ratio,prog_bar=True,sync_dist=True)
        self.val_step_outputs = []
    
    def log_histogram(self):
        if self.global_step %100 != 0:
            return 
        if not hasattr(self.loss_fn,"record"):
            return
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_histogram("radii", self.loss_fn.record["radii"], self.global_step)
                logger.experiment.add_histogram("norm_center", self.loss_fn.record["norm_center"], self.global_step)
                logger.experiment.add_histogram("dist", self.loss_fn.record["dist"], self.global_step)

def train_clap(model:pl.LightningModule, train_loader: torch.utils.data.DataLoader,
            val_loader:torch.utils.data.DataLoader,
            max_epochs:int,every_n_epochs:int,
            checkpoint_path:str,
            grad_accumulation_steps:int=1,
            max_grad_norm:float=0.0,
            num_nodes:int=1,
            gpus_per_node:int=1,
            strategy:str="auto",
            precision:str="16-true",
            restart:bool=False,
            if_profile:bool=False):
    logger_version = None if restart else 0
    csv_logger = CSVLogger(os.path.join(checkpoint_path,"logs"), name="csv",version=logger_version)
    tensorboard_logger = TensorBoardLogger(os.path.join(checkpoint_path,"logs"), name="tensorboard",version=logger_version)
    if max_grad_norm <= 0.0:
        max_grad_norm = None
    sync_batchnrom = True if gpus_per_node*num_nodes > 1 else False
    trainer = pl.Trainer(default_root_dir=checkpoint_path,
                         logger=[csv_logger, tensorboard_logger],
                         accumulate_grad_batches=grad_accumulation_steps,
                         gradient_clip_val=max_grad_norm,
                         accelerator="gpu",
                         devices=gpus_per_node,
                         num_nodes=num_nodes,
                         sync_batchnorm=sync_batchnrom,
                         precision=precision,
                         strategy=strategy,
                         max_epochs=max_epochs,
                         callbacks=[pl.callbacks.ModelCheckpoint(monitor = "val_acc",
                                                                mode = "max",
                                                                dirpath=os.path.join(checkpoint_path),
                                                                filename = 'best_val'),
                                    pl.callbacks.ModelCheckpoint(save_top_k = -1,
                                                                  save_last = False,
                                                                  every_n_epochs = every_n_epochs,
                                                                  dirpath=checkpoint_path,
                                                                  filename = "ssl-{epoch:d}"),
                                    pl.callbacks.LearningRateMonitor('epoch')],
                         profiler="simple" if if_profile else None)
    trainer.logger._default_hp_metric = False 
    # Check whether pretrained model exists and finished. If yes, load it and skip training
    trained_filename = os.path.join(checkpoint_path, 'best_val.ckpt')
    last_ckpt = os.path.join(checkpoint_path,'ssl-epoch={:d}.ckpt'.format(max_epochs-1))
    if os.path.isfile(last_ckpt) and (not restart):
        print(f'Found pretrained model at {last_ckpt}, loading...')
        model = CLAP.load_from_checkpoint(last_ckpt)
        return model
    else:
        # continue training
        ckpt_files = get_top_n_latest_checkpoints(checkpoint_path,1)
        if ckpt_files and (not restart):
            print("loading ...." + ckpt_files[0])
            trainer.fit(model, train_loader,val_loader,ckpt_path=ckpt_files[0])
        else:
            trainer.fit(model, train_loader,val_loader)
         # Load last checkpoint after training(best val_acc is just a reference do not load best val_acc here)
        model = CLAP.load_from_checkpoint(last_ckpt)
    return model

#########################################################
#  Linear classification(frozen-bacbone transfer learning)
#########################################################

class LinearClassification(pl.LightningModule):
    def __init__(self,
                 backbone:torch.nn.Module,
                 in_dim:int,out_dim:int,
                 use_batch_norm:bool,
                 optim_name:str,scheduler_name:str,lr:float,momentum:float,weight_decay:float,
                 n_epochs:int,scale_weight_decay:bool=False):
        super().__init__()
        # do not save the whole neural net as the hyperparameter
        self.save_hyperparameters(ignore=['backbone'])
        self.backbone = backbone
        self.backbone.remove_projection_head()
        if use_batch_norm:
            self.linear_net = models.BnLinearNet(in_dim,out_dim)
        else:
            self.linear_net = torch.nn.Linear(in_dim,out_dim)
        self.train_step_outputs = []
        self.test_step_outputs = []
        self.val_step_outputs = []
        self.train_epoch_loss = []
        self.test_acc1 = 0.0
        self.test_acc5 = 0.0

    def on_fit_start(self):
        if not self.backbone:
            raise Exception("need to set the backbone before training of validating") 
        # Save the initial state of the model
        init_ckpt_path = os.path.join(self.trainer.default_root_dir,"init.ckpt")
        if not os.path.isfile(init_ckpt_path):
            self.trainer.save_checkpoint(init_ckpt_path)

    def forward(self, x):
        # Extract features from the frozen backbone
        with torch.no_grad():  # Backbone is frozen
            features = self.backbone(x)
        return self.linear_net(features)

    def training_step(self, batch, batch_idx):
        imgs,labels = batch
        imgs = torch.cat(imgs,dim=0)
        labels = torch.cat(labels,dim=0)        
        preds = self.forward(imgs)
        loss = F.cross_entropy(preds, labels)
        self.train_step_outputs.append(loss.detach())
        self.log('train_iteration_loss', loss.detach(), prog_bar=True,sync_dist=True)  # Log iteration loss
        return loss
    
    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        imgs = torch.cat(imgs,dim=0)
        labels = torch.cat(labels,dim=0)  
        preds = self.forward(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.val_step_outputs.append(acc)
        return acc
    
    def test_step(self, batch, batch_idx):
        # Unpack the batch (input data and labels)
        imgs, labels = batch
        imgs = torch.cat(imgs,dim=0)
        labels = torch.cat(labels,dim=0)  
        logits = self.forward(imgs)
        loss = F.cross_entropy(logits, labels)

        # Compute accuracy
        # Calculate top-1 accuracy
        acc1 = (logits.argmax(dim=1) == labels).float().mean()
        
        # Calculate top-5 accuracy
        top5 = torch.topk(logits, k=5, dim=1).indices
        acc5 = (top5 == labels.view(-1, 1)).float().sum(dim=1).mean()
        # Log the test loss and accuracy
        self.log('batch_test_loss', loss, prog_bar=True,sync_dist=True)
        self.log('batch_test_acc1', acc1, prog_bar=True,sync_dist=True)
        self.log('batch_test_acc5', acc5, prog_bar=True,sync_dist=True)
        self.test_step_outputs.append({'test_loss': loss.detach(), 'test_acc1': acc1, 'test_acc5':acc5})
    
    def on_validation_epoch_end(self):
        val_acc =  torch.stack([x for x in self.val_step_outputs]).mean() 
        self.log('val_acc',val_acc,prog_bar=True,sync_dist=True)
        self.val_step_outputs = []
        return super().on_validation_epoch_end()

    def on_train_epoch_end(self):
        # `outputs` is a list of losses from each batch returned by training_step()
        avg_loss = torch.stack([x for x in self.train_step_outputs]).mean()  # Compute the average loss for the epoch
        # Save epoch loss for future reference
        self.log('train_epoch_loss', avg_loss, prog_bar=True,sync_dist=True)  # Log epoch loss
        # refresh the iteration loss at the end of every epoch
        self.train_step_outputs = []
        # Save epoch loss for future reference
        self.train_epoch_loss.append(avg_loss.item())
    
    def on_test_epoch_end(self):
        # Aggregate the losses and accuracies for the entire test set
        avg_loss = torch.stack([x['test_loss'] for x in self.test_step_outputs]).mean()
        avg_top1_acc = torch.stack([x['test_acc1'] for x in self.test_step_outputs]).mean()
        avg_top5_acc = torch.stack([x['test_acc5'] for x in self.test_step_outputs]).mean()
        # Log the aggregated metrics
        self.log('test_loss', avg_loss,sync_dist=True)
        self.log('test_acc1', avg_top1_acc,sync_dist=True)
        self.log('test_acc5', avg_top5_acc,sync_dist=True)
        return {'test_loss': avg_loss, 'test_acc1': avg_top1_acc, 'test_acc5': avg_top5_acc}


    def configure_optimizers(self):
        if self.hparams.optim_name == "SGD":
            optimizer = optim.SGD(params=self.linear_net.parameters(),
                                  lr=self.hparams.lr,
                                  momentum=self.hparams.momentum,
                                  weight_decay=self.hparams.weight_decay,
                                  nesterov=True)
        elif self.hparams.optim_name == "Adam":
            optimizer = optim.Adam(params=self.linear_net.parameters(),
                                  lr=self.hparams.lr,
                                  weight_decay=self.hparams.weight_decay)
        else:
            raise NotImplementedError("optimizer:"+ self.optimizer +" not implemented")
        if self.hparams.scheduler_name == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=self.hparams.n_epochs)
        elif self.hparams.scheduler_name == "multi_step":
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[int(self.hparams.n_epochs*0.6),
                                                                int(self.hparams.n_epochs*0.8)],
                                                    gamma=0.1)
        else:
            return [optimizer]
        return [optimizer],[scheduler]
    '''
    def reset_optimizer_scheduler(self,optimizer = None,scheduler = None):
        # Reinitialize both optimizer and scheduler by calling configure_optimizers
        if optimizer and scheduler:
            self.optimizers = optimizer
            self.schedulers = scheduler
        else:
            optimizers, schedulers = self.configure_optimizers()
            self.optimizers = optimizers
            self.schedulers = schedulers

    def state_dict(self):
        # override the default state dict to avoid saving the backbone
        return {
            'model_state_dict': self.linear_net.state_dict(),
            'optimizer_state_dict': self.optimizers().state_dict(),
            'scheduler_state_dict': self.lr_schedulers().state_dict() 
        }

    def load_from_customized_checkpoint(self,path:str):
        state_dict = torch.load(path,weights_only=True)["state_dict"]
        self.linear_net.load_state_dict(state_dict['model_state_dict'])
        self.optimizers().load_state_dict(state_dict['optimizer_state_dict'])
        self.lr_schedulers().load_state_dict(state_dict['scheduler_state_dict'])
    '''

def train_lc(linear_model:pl.LightningModule,
            train_loader: torch.utils.data.DataLoader,
            test_loader:torch.utils.data.DataLoader,
            val_loader:torch.utils.data.DataLoader,
            max_epochs:int,
            every_n_epochs:int,
            checkpoint_path:str,
            num_nodes:int=1,
            gpus_per_node:int=1,
            strategy:str = "auto",
            precision:str="16-true",
            restart:bool = False,
            if_profile:bool = False):
    # Check whether pretrained model exists and finished. If yes, load it and skip training
    trained_filename = os.path.join(checkpoint_path, 'best_val.ckpt')
    last_ckpt = os.path.join(checkpoint_path,'lc-epoch={:d}.ckpt'.format(max_epochs-1))
    if os.path.isfile(trained_filename) and os.path.isfile(last_ckpt) and (not restart):
        print(f'Found pretrained model at {trained_filename}, loading...')
        model = LinearClassification.load_from_checkpoint(trained_filename,backbone = linear_model.backbone) # Automatically loads the model with the saved hyperparameters
        return model
    logger_version = None if restart else 0
    csv_logger = CSVLogger(os.path.join(checkpoint_path,"logs"), name="csv",version=logger_version)
    tensorboard_logger = TensorBoardLogger(os.path.join(checkpoint_path,"logs"), name="tensorboard",version=logger_version)
    trainer = pl.Trainer(default_root_dir=checkpoint_path,
                         logger=[csv_logger,tensorboard_logger],
                         accelerator="gpu",
                         devices=gpus_per_node,
                         num_nodes=num_nodes,
                         strategy=strategy,
                         max_epochs=max_epochs,
                         precision=precision,
                         callbacks=[pl.callbacks.ModelCheckpoint(monitor = "val_acc",
                                                                mode = "max",
                                                                dirpath=os.path.join(checkpoint_path),
                                                                filename = 'best_val'),
                                    pl.callbacks.ModelCheckpoint(save_top_k = -1,
                                                                save_last = False,
                                                                every_n_epochs = every_n_epochs,
                                                                dirpath=checkpoint_path,
                                                                filename = "lc-{epoch:d}"),
                                    pl.callbacks.LearningRateMonitor('epoch')],
                         profiler="simple" if if_profile else None)
    trainer.logger._default_hp_metric = False 
    # continue training
    ckpt_files = get_top_n_latest_checkpoints(checkpoint_path,1)
    if ckpt_files and (not restart):
        print("loading ... " + ckpt_files[0])
        trainer.fit(linear_model, train_loader,val_loader,ckpt_path=ckpt_files[0])
    else:
        trainer.fit(linear_model, train_loader,val_loader)
    # load the model with the best validation accuracy to avoid overfitting
    linear_model = LinearClassification.load_from_checkpoint(trained_filename,backbone = linear_model.backbone) # Load best checkpoint after training
    test_output = trainer.test(linear_model,test_loader)
    result = {"test_loss":test_output[0]["test_loss"],
              "test_acc1":test_output[0]["test_acc1"],
              "test_acc5":test_output[0]["test_acc5"]
            }
    with open(os.path.join(checkpoint_path,"results.json"),"w") as fs:
        json.dump(result,fs,indent=4)
    
    return linear_model

#########################################################
#  Fine-tune(semi-supervised learning)
#########################################################
# this class is similar to linear classification
# the difference is that here both batckbon and linear_net
# will be fine-tuned
class FineTune(pl.LightningModule):
    def __init__(self,backbone:torch.nn.Module,
                 linear_net:torch.nn.Module,
                 optim_name:str,lr:float,momentum:float,weight_decay:float,
                 n_epochs:int):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone','linear_net'])
        self.backbone = backbone
        self.backbone.remove_projection_head()
        self.linear_net = linear_net
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.train_epoch_loss = []
        self.test_acc1 = 0.0
        self.test_acc5 = 0.0
    
    def on_fit_start(self):
        if not self.backbone:
            raise Exception("need to set the backbone before training or validating")
        if not self.linear_net:
            raise Exception("need to set the linear_net before training or validating") 
        # Save the initial state of the model
        init_ckpt_path = os.path.join(self.trainer.default_root_dir,"init.ckpt")
        if not os.path.isfile(init_ckpt_path):
            self.trainer.save_checkpoint(init_ckpt_path)
    def forward(self, x):
        # Extract features from the frozen backbone
        # Do NOT freeze backbone with nograd
        features = self.backbone(x)
        return self.linear_net(features)

    def training_step(self, batch, batch_idx):
        imgs,labels = batch
        imgs = torch.cat(imgs,dim=0)
        labels = torch.cat(labels,dim=0)        
        preds = self.forward(imgs)
        loss = F.cross_entropy(preds, labels)
        self.train_step_outputs.append(loss.detach())
        self.log('train_iteration_loss', loss.detach(), prog_bar=True,sync_dist=True)  # Log iteration loss
        return loss
    
    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        imgs = torch.cat(imgs,dim=0)
        labels = torch.cat(labels,dim=0)  
        preds = self.forward(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.val_step_outputs.append(acc)
        return acc
    
    def test_step(self, batch, batch_idx):
        # Unpack the batch (input data and labels)
        imgs, labels = batch
        imgs = torch.cat(imgs,dim=0)
        labels = torch.cat(labels,dim=0)  
        logits = self.forward(imgs)
        loss = F.cross_entropy(logits, labels)
        # Compute accuracy
        # Calculate top-1 accuracy
        acc1 = (logits.argmax(dim=1) == labels).float().mean()
        
        # Calculate top-5 accuracy
        top5 = torch.topk(logits, k=5, dim=1).indices
        acc5 = (top5 == labels.view(-1, 1)).float().sum(dim=1).mean()

        # Log the test loss and accuracy
        self.log('batch_test_loss', loss, prog_bar=True,sync_dist=True)
        self.log('batch_test_acc1', acc1, prog_bar=True,sync_dist=True)
        self.log('batch_test_acc5', acc5, prog_bar=True,sync_dist=True)
        self.test_step_outputs.append({'test_loss': loss.detach(), 'test_acc1': acc1, 'test_acc5':acc5})
    
    def on_validation_epoch_end(self):
        val_acc =  torch.stack([x for x in self.val_step_outputs]).mean() 
        self.log('val_acc',val_acc,prog_bar=True,sync_dist=True)
        self.val_step_outputs = []
        return super().on_validation_epoch_end()


    def on_train_epoch_end(self):
        # `outputs` is a list of losses from each batch returned by training_step()
        avg_loss = torch.stack([x for x in self.train_step_outputs]).mean()  # Compute the average loss for the epoch
        # Save epoch loss for future reference
        self.log('train_epoch_loss', avg_loss, prog_bar=True,sync_dist=True)  # Log epoch loss
        # refresh the iteration loss at the end of every epoch
        self.train_step_outputs = []
        # Save epoch loss for future reference
        self.train_epoch_loss.append(avg_loss.item())
    
    def on_test_epoch_end(self):
        # Aggregate the losses and accuracies for the entire test set
        avg_loss = torch.stack([x['test_loss'] for x in self.test_step_outputs]).mean()
        avg_top1_acc = torch.stack([x['test_acc1'] for x in self.test_step_outputs]).mean()
        avg_top5_acc = torch.stack([x['test_acc5'] for x in self.test_step_outputs]).mean()
        # Log the aggregated metrics
        self.log('test_loss', avg_loss,sync_dist=True)
        self.log('test_acc1', avg_top1_acc,sync_dist=True)
        self.log('test_acc5', avg_top5_acc,sync_dist=True)
        return {'test_loss': avg_loss, 'test_acc1': avg_top1_acc, 'test_acc5': avg_top5_acc}


    def configure_optimizers(self):
        # Need to use list(self.backbone.parameters()) + list(self.linear_net.parameters()),
        # to optimize the parameters from both networks
        if self.hparams.optim_name == "SGD":
            optimizer = optim.SGD(params=list(self.backbone.parameters()) + list(self.linear_net.parameters()),
                                  lr=self.hparams.lr,
                                  momentum=self.hparams.momentum,
                                  weight_decay=self.hparams.weight_decay,
                                  nesterov=True)
        elif self.hparams.optim_name == "Adam":
            optimizer = optim.Adam(params=list(self.backbone.parameters()) + list(self.linear_net.parameters()),
                                  lr=self.hparams.lr,
                                  weight_decay=self.hparams.weight_decay)
        else:
            raise NotImplementedError("optimizer:"+ self.optimizer +" not implemented")

        return [optimizer]

def train_finetune(
            finetune_model:pl.LightningModule,
            train_loader: torch.utils.data.DataLoader,
            test_loader:torch.utils.data.DataLoader,
            val_loader:torch.utils.data.DataLoader,
            every_n_epochs:int,
            max_epochs:int,
            checkpoint_path:str,
            num_nodes:int=1,
            gpus_per_node:int=1,
            strategy:str = "auto",
            precision:str="16-true",
            restart:bool=False,
            if_profile:bool=False):
    # Check whether pretrained model exists. If yes, load it and skip training
    trained_filename = os.path.join(checkpoint_path, 'best_val.ckpt')
    last_ckpt = os.path.join(checkpoint_path,'ft-epoch={:d}.ckpt'.format(max_epochs-1))
    if os.path.isfile(trained_filename) and os.path.isfile(last_ckpt) and (not restart):
        print(f'Found pretrained model at {trained_filename}, loading...')
        # Automatically loads the model with the saved hyperparameters
        # backbone and linear_net are ignored when saving the hyperparameters
        # loading it by providing them in the constructor
        # the backbone and linear_net will be updated from the state_dict() after the object is constucted
        model = FineTune.load_from_checkpoint(trained_filename,backbone = finetune_model.backbone,linear_net = finetune_model.linear_net) 
        return model
    
    logger_version = None if restart else 0
    csv_logger = CSVLogger(os.path.join(checkpoint_path,"logs"), name="csv",version=logger_version)
    tensorboard_logger = TensorBoardLogger(os.path.join(checkpoint_path,"logs"), name="tensorboard",version=logger_version)
    trainer = pl.Trainer(default_root_dir=checkpoint_path,
                         logger=[csv_logger,tensorboard_logger],
                         accelerator="gpu",
                         devices=gpus_per_node,
                         num_nodes=num_nodes,
                         strategy=strategy,
                         max_epochs=max_epochs,
                         precision=precision,
                         callbacks=[pl.callbacks.ModelCheckpoint(monitor = "val_acc",
                                                                mode = "max",
                                                                dirpath=os.path.join(checkpoint_path),
                                                                filename = 'best_val'),
                                    pl.callbacks.ModelCheckpoint(save_top_k = -1,
                                                                save_last = False,
                                                                every_n_epochs = every_n_epochs,
                                                                dirpath=checkpoint_path,
                                                                filename = "ft-{epoch:d}"),
                                    pl.callbacks.LearningRateMonitor('epoch')],
                         profiler="simple" if if_profile else None )
    trainer.logger._default_hp_metric = False  
    # continue training
    ckpt_files = get_top_n_latest_checkpoints(checkpoint_path,1)
    if ckpt_files and (not restart):
        print("loading ..." + ckpt_files[0])
        trainer.fit(finetune_model, train_loader,val_loader,ckpt_path=ckpt_files[0])
    else:
        trainer.fit(finetune_model, train_loader,val_loader)
    # the backbone and linear_net will be updated to the latest version
    # since they are registered in the pytorchlightning module
    # can check this point by print the state_dict() (e.g. key = "net.conv1.weight" in backbone.state_dict() before and after training)
    finetune_model = FineTune.load_from_checkpoint(trained_filename,backbone = finetune_model.backbone,linear_net = finetune_model.linear_net) 
    test_output = trainer.test(finetune_model,test_loader)
    result = {"test_loss":test_output[0]["test_loss"],
              "test_acc1":test_output[0]["test_acc1"],
              "test_acc5":test_output[0]["test_acc5"]}
    with open(os.path.join(checkpoint_path,"results.json"),"w") as fs:
        json.dump(result,fs,indent=4)

    return finetune_model
       
       
#########################################################
#  KNN(K neareast neighbour)
#########################################################
# This class is used to test the pre-trained representation
# no training step,only predict and test steps
class KNN(pl.LightningModule):
    def __init__(self,backbone:torch.nn.Module,k_nbrs:int,dist_type:str="euclidean"):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone'])
        self.backbone = backbone
        self.backbone.remove_projection_head()
        self.k = k_nbrs
        # cosine or euclidean
        self.dist_type = dist_type
        self.features_local = []
        self.labels_local = []
        self.features  = None
        self.labels = None
        self.test_step_outputs = []
        self.test_acc1 = 0.0
        self.test_acc5 = 0.0
    
    def forward(self, x):
        # Extract features from the frozen backbone
        # Do NOT freeze backbone with nograd
        features = self.backbone(x)
        return self.linear_net(features)
    def on_fit_start(self):
        if not self.backbone:
            raise Exception("need to set the backbone before training or validating")
    @torch.no_grad()    
    def predict_step(self, batch, batch_idx):
        imgs,labels = batch
        imgs = torch.cat(imgs,dim=0)
        labels = torch.cat(labels,dim=0)
        features =  self.forward(imgs)   
        self.features_local.append(features)
        self.labels_local.append(labels)
        return features
    
    def on_test_start(self):
        features_local = torch.cat(self.features_local,dim=0)
        labels_local = torch.cat(self.labels_local,dim=0)
        if dist.is_available() and dist.is_initialized():
            ws = dist.get_world_size()
            features_list = [torch.zeros_like(features_local) for _ in range(ws)]
            labels_list = [torch.zeros_like(labels_local) for _ in range(ws)]
            dist.all_gather(features_list,features_local,async_op=False)
            dist.all_gather(labels_list,labels_local,async_op=False)
            self.features = torch.cat(features_list,dim=0)
            self.labels = torch.cat(labels_list,dim=0)
        else:
            self.features = features_local
            self.labels = labels_local
        if self.dist_type =="cosine":
            self.features = F.normalize(self.features,dim=-1)
        return super().on_test_start()
    
    def test_step(self, batch, batch_idx):
        # Unpack the batch (input data and labels)
        imgs, labels = batch
        imgs = torch.cat(imgs,dim=0)
        test_labels = torch.cat(labels,dim=0)  
        # shape of preds:[B,O]
        preds = self.forward(imgs)
        if self.dist_type == "euclidean":
            # shape of features = [N,O], diff: [B,N,O], 
            diff = preds[:,None,:] - self.features[None,:,:]
            # shape of distance =[B,N]
            distance = torch.sum(diff**2,dim=-1)
        elif self.dist_type == "cosine":
            preds = F.normalize(preds,dim=-1)
            distance = torch.matmul(preds,self.features.t())
        # shape of indices = [B,k]
        indices = torch.topk(-distance,self.k)
        # for create labels of the nearest neighbour, (B,k)
        nbr_labels = self.labels[indices]
        acc1,acc5 = 0
        for i, test_label in enumerate(test_labels):
            acc1 += test_label in torch.bincount(nbr_labels[i]).topk(1).indices
            acc5 += test_label in torch.bincount(nbr_labels[i]).topk(5).indices
        acc1 /= test_labels.shape[0]
        acc5 /= test_labels.shape[0]
        self.test_step_outputs.append({'test_acc1': acc1, 'test_acc5':acc5})
    
    def on_test_epoch_end(self):
        # Aggregate the losses and accuracies for the entire test set
        avg_top1_acc = torch.stack([x['test_acc1'] for x in self.test_step_outputs]).mean()
        avg_top5_acc = torch.stack([x['test_acc5'] for x in self.test_step_outputs]).mean()
        # Log the aggregated metrics
        self.log('test_acc1', avg_top1_acc,sync_dist=True)
        self.log('test_acc5', avg_top5_acc,sync_dist=True)
        return {'test_acc1': avg_top1_acc, 'test_acc5': avg_top5_acc}

def train_knn(
            knn_model:pl.LightningModule,
            train_loader: torch.utils.data.DataLoader,
            test_loader:torch.utils.data.DataLoader,
            val_loader:torch.utils.data.DataLoader,
            checkpoint_path:str,
            num_nodes:int=1,
            gpus_per_node:int=1,
            strategy:str = "auto",
            precision:str="16-true"):
    
    trainer = pl.Trainer(default_root_dir=checkpoint_path,
                         accelerator="gpu",
                         devices=gpus_per_node,
                         num_nodes=num_nodes,
                         strategy=strategy,
                         precision=precision)
    trainer.logger._default_hp_metric = False  

    trainer.predict(knn_model,train_loader)
    test_output = trainer.test(knn_model,test_loader)
    result = {"test_acc1":test_output[0]["test_acc1"],
              "test_acc5":test_output[0]["test_acc5"]}
    with open(os.path.join(checkpoint_path,"results.json"),"w") as fs:
        json.dump(result,fs,indent=4)
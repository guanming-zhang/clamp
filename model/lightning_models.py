import pytorch_lightning as pl
from torch import optim
import torch.utils
import torch.utils.data
from . import loss_module
from . import models
from . import lars
import torch
import torch.nn.functional as F 
import os
import re
import subprocess
import json
from pytorch_lightning.profilers import SimpleProfiler
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
    def __init__(self,backbone_name:str,backbone_out_dim:int,prune:bool,use_projection_head:bool,proj_dim:int,proj_out_dim:int,
                 optim_name:str,lr:float,momentum:float,weight_decay:float,eta:float,
                 warmup_epochs:int,n_epochs:int,
                 n_views:int,batch_size:int,lw0:float,lw1:float,lw2:float,n_pow_iter:int=20,rs:float=2.0,pot_pow:float=2.0,margin:float=1e-7):
        super().__init__()
        self.backbone = models.BackboneNet(backbone_name,backbone_out_dim,prune,use_projection_head,proj_dim,proj_out_dim)
        self.loss_fn = loss_module.EllipsoidPackingLoss(n_views,batch_size,lw0,lw1,lw2,n_pow_iter,rs,pot_pow,margin)
        self.train_epoch_loss = []  # To store epoch loss for training
        self.train_step_outputs = []
        # all the hyperparameters are added as attributes to this class
        self.save_hyperparameters()
    def configure_optimizers(self):
        if self.hparams.optim_name == "SGD":
            optimizer = optim.SGD(params=self.backbone.parameters(),
                                  lr=self.hparams.lr,
                                  momentum=self.hparams.momentum,
                                  weight_decay=self.hparams.weight_decay,
                                  nesterov=True)
        elif self.hparams.optim_name == "AdamW":
            optimizer = optim.AdamW(params=self.backbone.parameters(),
                                  lr=self.hparams.lr,
                                  momentum=self.hparams.momentum,
                                  weight_decay=self.hparams.weight_decay)
        elif self.hparams.optim_name == "LARS":
            optimizer = lars.LARS(params=self.backbone.parameters(),
                                  lr=self.hparams.lr,
                                  trust_coefficient = self.hparams.eta,
                                  momentum=self.hparams.momentum,
                                  weight_decay=self.hparams.weight_decay)
        else:
            raise NotImplementedError("optimizer:"+ self.optimizer +" not implemented")

        linear = optim.lr_scheduler.LinearLR(optimizer,total_iters=self.hparams.warmup_epochs)
        cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=self.hparams.n_epochs - self.hparams.warmup_epochs)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer,schedulers=[linear, cosine], milestones=[self.hparams.warmup_epochs])
        return [optimizer],[scheduler]
    
    def training_step(self, batch, batch_idx):
        imgs,labels = batch
        imgs = torch.cat(imgs,dim=0)
        preds = self.backbone(imgs)
        # the labels are dummy since label is not used in ssl
        loss = self.loss_fn(preds,None)
        self.train_step_outputs.append(loss.detach())
        self.log('train_iteration_loss', loss.detach(), prog_bar=True,sync_dist=True)  # Log iteration loss
        return loss
    
    def on_train_epoch_end(self):
        # `outputs` is a list of losses from each batch returned by training_step()
        avg_loss = torch.stack([x for x in self.train_step_outputs]).mean()  # Compute the average loss for the epoch
        self.log('train_epoch_loss', avg_loss, prog_bar=True,sync_dist=True)  # Log epoch loss
        # refresh the iteration loss at the end of every epoch
        self.train_step_outputs = []
        # Save epoch loss for future reference
        self.train_epoch_loss.append(avg_loss.item())
    
    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        imgs = torch.cat(imgs,dim=0)
        preds = self.backbone(imgs)
        # the shape of preds before reshaping [V*B,O]
        preds = torch.reshape(preds,(self.hparams.n_views,self.hparams.batch_size,preds.shape[-1]))
        # the shape of centers in [B,O]
        centers = torch.mean(preds,dim=0,keepdim=True)
        # the shape of diff : (V,B,B,O)
        diff = preds[:,:,None,:] - centers[None,None,:,:]
        # distance matrix (V,B,B)
        dist_matrix = torch.sum(diff**2,dim=-1)
        # nearest (V,B), nearest[1,2] = 4 
        # nearest[1,2] = 4 means the the nearest cluster to
        # the 1th view of in cluster 2 is cluster 4 
        nearest = torch.argmin(dist_matrix,dim=-1)
        correct = nearest == torch.arange(self.hparams.batch_size,device=nearest.device).repeat(self.hparams.n_views,1)
        acc = (correct.sum()/(self.hparams.n_views*self.hparams.batch_size)).float()
        self.log('val_acc', acc, prog_bar=True,sync_dist=True)
        return acc

def train_clap(model:pl.LightningModule, train_loader: torch.utils.data.DataLoader,
            val_loader:torch.utils.data.DataLoader,
            max_epochs:int,every_n_epochs:int,
            checkpoint_path:str,
            grad_accumulation_steps:int=1,
            num_nodes:int=1,
            gpus_per_node:int=1,
            strategy:str="auto",
            precision:str="16-true",
            restart:bool=False,
            prof_mem:bool=False):
    logger_version = None if restart else 0
    csv_logger = CSVLogger(os.path.join(checkpoint_path,"logs"), name="csv",version=logger_version)
    tensorboard_logger = TensorBoardLogger(os.path.join(checkpoint_path,"logs"), name="tensorboard",version=logger_version)

    trainer = pl.Trainer(default_root_dir=checkpoint_path,
                         logger=[csv_logger, tensorboard_logger],
                         accumulate_grad_batches=grad_accumulation_steps,
                         accelerator="gpu",
                         devices=gpus_per_node,
                         num_nodes=num_nodes,
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
                                    pl.callbacks.LearningRateMonitor('epoch')])
    
    trainer.logger._default_hp_metric = False 
    # Check whether pretrained model exists and finished. If yes, load it and skip training
    trained_filename = os.path.join(checkpoint_path, 'best_val.ckpt')
    last_ckpt = os.path.join(checkpoint_path,'ssl-epoch={:d}.ckpt'.format(max_epochs-1))
    if os.path.isfile(trained_filename) and os.path.isfile(last_ckpt) and (not restart):
        print(f'Found pretrained model at {trained_filename}, loading...')
        model = CLAP.load_from_checkpoint(trained_filename)
        return model
    else:
        # continue training
        ckpt_files = get_top_n_latest_checkpoints(checkpoint_path,1)
        if ckpt_files and (not restart):
            print("loading ...." + ckpt_files[0])
            trainer.fit(model, train_loader,val_loader,ckpt_path=ckpt_files[0])
        else:
            trainer.fit(model, train_loader,val_loader)
        model = CLAP.load_from_checkpoint(trained_filename) # Load best checkpoint after training
    return model

#########################################################
#  Linear classification(frozen-bacbone transfer learning)
#########################################################

class LinearClassification(pl.LightningModule):
    def __init__(self,
                 backbone:torch.nn.Module,
                 in_dim:int,out_dim:int,use_batch_norm:bool,
                 optim_name:str,scheduler_name:str,lr:float,momentum:float,weight_decay:float,
                 n_epochs:int):
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
        self.log('val_acc', acc, prog_bar=True,sync_dist=True)
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
        elif self.hparams.optim_name == "AdamW":
            optimizer = optim.AdamW(params=self.linear_net.parameters(),
                                  lr=self.hparams.lr,
                                  momentum=self.hparams.momentum,
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
            restart:bool = False):
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
                                    pl.callbacks.LearningRateMonitor('epoch')])
    trainer.logger._default_hp_metric = False 
    # continue training
    ckpt_files = get_top_n_latest_checkpoints(checkpoint_path,1)
    if ckpt_files and (not restart):
        print("loading ... " + ckpt_files[0])
        trainer.fit(linear_model, train_loader,val_loader,ckpt_path=ckpt_files[0])
    else:
        trainer.fit(linear_model, train_loader,val_loader)
    test_output = trainer.test(linear_model,test_loader)
    result = {"test_loss":test_output[0]["test_loss"],
              "test_acc1":test_output[0]["test_acc1"],
              "test_acc5":test_output[0]["test_acc5"]
            }
    with open(os.path.join(checkpoint_path,"results.json"),"w") as fs:
        json.dump(result,fs,indent=4)
    linear_model = LinearClassification.load_from_checkpoint(trained_filename,backbone = linear_model.backbone) # Load best checkpoint after training
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
        self.log('val_acc', acc, prog_bar=True,sync_dist=True)
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
        elif self.hparams.optim_name == "AdamW":
            optimizer = optim.AdamW(params=list(self.backbone.parameters()) + list(self.linear_net.parameters()),
                                  lr=self.hparams.lr,
                                  momentum=self.hparams.momentum,
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
            restart:bool=False):
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
                                    pl.callbacks.LearningRateMonitor('epoch')])
    trainer.logger._default_hp_metric = False   
    # continue training
    ckpt_files = get_top_n_latest_checkpoints(checkpoint_path,1)
    if ckpt_files and (not restart):
        print("loading ..." + ckpt_files[0])
        trainer.fit(finetune_model, train_loader,val_loader,ckpt_path=ckpt_files[0])
    else:
        trainer.fit(finetune_model, train_loader,val_loader)
    test_output = trainer.test(finetune_model,test_loader)
    result = {"test_loss":test_output[0]["test_loss"],
              "test_acc1":test_output[0]["test_acc1"],
              "test_acc5":test_output[0]["test_acc5"]}
    with open(os.path.join(checkpoint_path,"results.json"),"w") as fs:
        json.dump(result,fs,indent=4)
    # the backbone and linear_net will be updated to the latest version
    # since they are registered in the pytorchlightning module
    # can check this point by print the state_dict() (e.g. key = "net.conv1.weight" in backbone.state_dict() before and after training)
    finetune_model = FineTune.load_from_checkpoint(trained_filename,backbone = finetune_model.backbone,linear_net = finetune_model.linear_net) 
    return finetune_model
       

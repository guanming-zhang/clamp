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
from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler, PyTorchProfiler

class CLAP(pl.LightningModule):
    def __init__(self,backbone_name:str,backbone_out_dim:int,prune:bool,use_projection_header:bool,proj_out_dim:int,
                 optim_name:str,lr:float,momentum:float,weight_decay:float,eta:float, 
                 warmup_epochs:int,n_epochs:int,
                 n_views:int,batch_size:int,lw0:float,lw1:float,lw2:float,n_pow_iter:int=20,rs:float=2.0,margin:float=1e-7):
        super().__init__()
        self.backbone = models.BackboneNet(backbone_name,backbone_out_dim,prune,use_projection_header,proj_out_dim)
        self.loss_fn = loss_module.EllipsoidPackingLoss(n_views,batch_size,lw0,lw1,lw2,n_pow_iter,rs,margin)
        self.train_epoch_loss = []  # To store epoch loss for training
        self.train_step_outputs = []
        # all the hyperparameters are added as attributes to this class
        self.save_hyperparameters()
    def configure_optimizers(self):
        if self.hparams.optim_name == "SGD":
            optimizer = optim.SGD(params=self.backbone.parameters(),
                                  lr=self.hparams.lr,
                                  momentum=self.hparams.momentum,
                                  weight_decay=self.hparams.weight_decay)
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
        self.train_step_outputs.append(loss)
        return loss
    
    def on_train_epoch_end(self):
        # `outputs` is a list of losses from each batch returned by training_step()
        avg_loss = torch.stack([x for x in self.train_step_outputs]).mean()  # Compute the average loss for the epoch
        self.log('train_epoch_loss', avg_loss, prog_bar=True)  # Log epoch loss

        # Save epoch loss for future reference
        self.train_epoch_loss.append(avg_loss.item())

class LinearClassification(pl.LightningModule):
    def __init__(self,
                 in_dim:int,out_dim:int,use_batch_norm:bool,
                 optim_name:str,lr:float,momentum:float,weight_decay:float,
                 n_epochs:int):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = None
        if use_batch_norm:
            self.linear_net = models.BnLinearNet(in_dim,out_dim)
        else:
            self.linear_net = torch.nn.Linear(in_dim,out_dim)
        self.train_step_outputs = []
        self.test_step_outputs = []
        self.train_epoch_loss = []
        self.test_acc1 = 0.0
        self.test_acc5 = 0.0
    
    def set_backbone(self,backbone):
        # need to call this function
        self.backbone = backbone
        self.backbone.remove_projection_header()
        for param in self.backbone.parameters():
            param.requires_grad = False
    
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
        self.train_step_outputs.append(loss)
    
    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        imgs = torch.cat(imgs,dim=0)
        labels = torch.cat(labels,dim=0)  
        preds = self.forward(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.log('val_acc', acc, prog_bar=True)
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
        self.log('batch_test_loss', loss, prog_bar=True)
        self.log('batch_test_acc1', acc1, prog_bar=True)
        self.log('batch_test_acc5', acc5, prog_bar=True)
        self.test_step_outputs.append({'test_loss': loss, 'test_acc1': acc1, 'test_acc5':acc5})

    def on_train_epoch_end(self):
        # `outputs` is a list of losses from each batch returned by training_step()
        avg_loss = torch.stack([x for x in self.train_step_outputs]).mean()  # Compute the average loss for the epoch
        # Save epoch loss for future reference
        self.log('train_epoch_loss', avg_loss, prog_bar=True)  # Log epoch loss
        # Save epoch loss for future reference
        self.train_epoch_loss.append(avg_loss.item())
    
    def on_test_epoch_end(self):
        # Aggregate the losses and accuracies for the entire test set
        avg_loss = torch.stack([x['test_loss'] for x in self.test_step_outputs]).mean()
        avg_top1_acc = torch.stack([x['test_acc1'] for x in self.test_step_outputs]).mean()
        avg_top5_acc = torch.stack([x['test_acc5'] for x in self.test_step_outputs]).mean()
        
        # Log the aggregated metrics
        self.log('test_loss', avg_loss)
        self.log('test_acc1', avg_top1_acc)
        self.log('test_acc5', avg_top5_acc)
        return {'test_loss': avg_loss, 'test_acc1': avg_top1_acc, 'test_acc5': avg_top5_acc}


    def configure_optimizers(self):
        if self.hparams.optim_name == "SGD":
            optimizer = optim.SGD(params=self.linear_net.parameters(),
                                  lr=self.hparams.lr,
                                  momentum=self.hparams.momentum,
                                  weight_decay=self.hparams.weight_decay)
        elif self.hparams.optim_name == "AdamW":
            optimizer = optim.AdamW(params=self.linear_net.parameters(),
                                  lr=self.hparams.lr,
                                  momentum=self.hparams.momentum,
                                  weight_decay=self.hparams.weight_decay)
        else:
            raise NotImplementedError("optimizer:"+ self.optimizer +" not implemented")
        #cosine scheduler
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=self.hparams.n_epochs)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[int(self.hparams.n_epochs*0.6),
                                                                  int(self.hparams.n_epochs*0.8)],
                                                      gamma=0.1)
        return [optimizer],[scheduler]
    
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


def train_clap(model:pl.LightningModule, train_loader: torch.utils.data.DataLoader,
            max_epochs:int,every_n_epochs:int,
            checkpoint_path:str,
            num_nodes:int=1,gpu_per_node:int=1,strategy:str="auto",precision:str="16-true"):
    
    trainer = pl.Trainer(default_root_dir=checkpoint_path,
                         accelerator="gpu",
                         devices=gpu_per_node,
                         num_nodes=num_nodes,
                         precision=precision,
                         strategy=strategy,
                         max_epochs=max_epochs,
                         callbacks=[pl.callbacks.ModelCheckpoint(save_weights_only=True,
                                                                  save_top_k = -1,
                                                                  save_last = True,
                                                                  every_n_epochs = every_n_epochs,
                                                                  dirpath=checkpoint_path,
                                                                  filename = "CLAP-{epoch:02d}.ckpt"),
                                    pl.callbacks.LearningRateMonitor('epoch')])
    '''
    profiler = SimpleProfiler()
    trainer = pl.Trainer(profiler=profiler,
                         default_root_dir=checkpoint_path,
                         accelerator="gpu",
                         devices=1,
                         max_epochs=max_epochs,
                         precision=precision,
                         callbacks=[pl.callbacks.ModelCheckpoint(save_weights_only=True,
                                                                  save_top_k = -1,
                                                                  save_last = True,
                                                                  every_n_epochs = every_n_epochs,
                                                                  dirpath=checkpoint_path,
                                                                  filename = "CLAP-{epoch:02d}.ckpt"),
                                    pl.callbacks.LearningRateMonitor('epoch')])
    '''
    trainer.logger._default_hp_metric = False 
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(checkpoint_path, 'last.ckpt')
    if os.path.isfile(pretrained_filename):
        print(f'Found pretrained model at {pretrained_filename}, loading...')
        model = CLAP.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
    else:
        pl.seed_everything(137) # To be reproducable
        trainer.fit(model, train_loader)
        model = CLAP.load_from_checkpoint(os.path.join(checkpoint_path,"last.ckpt")) # Load best checkpoint after training
    return model

def get_top_n_latest_checkpoints(directory, n):
    # Regular expression to extract epoch number from filename
    pattern = re.compile(r'CLAP-epoch=(\d+)\.ckpt')
    
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
    top_n_files = sorted(epoch_files, key=lambda x: x[0], reverse=True) #[:n]
    # Return the filenames of the top N files
    return [file for _, file in top_n_files]

def train_lc(ssl_model:pl.LightningModule, 
            ssl_ckpt_path:str,
            linear_model:pl.LightningModule,
            train_loader: torch.utils.data.DataLoader,
            test_loader:torch.utils.data.DataLoader,
            val_loader:torch.utils.data.DataLoader,
            max_epochs:int,
            checkpoint_path:str,
            mode:str,
            num_nodes:int=1,
            gpu_per_nodes:int=1,
            strategy:str = "auto",
            precision:str="16-true"):
    # Check whether pretrained model exists. If yes, load it and skip training
    trained_filename = os.path.join(checkpoint_path, 'last.ckpt')
    if os.path.isfile(trained_filename):
        print(f'Found pretrained model at {trained_filename}, loading...')
        model = LinearClassification.load_from_checkpoint(trained_filename) # Automatically loads the model with the saved hyperparameters
        return model
    '''
    trainer = pl.Trainer(default_root_dir=checkpoint_path,
                         accelerator="gpu",
                         devices=1,
                         max_epochs=max_epochs,
                         callbacks=[pl.callbacks.ModelCheckpoint(save_weights_only=True,
                                                                  save_top_k = -1,
                                                                  save_last = True,
                                                                  every_n_epochs = every_n_epochs,
                                                                  dirpath=os.path.join(checkpoint_path,"_temp"),
                                                                  filename = 'LC.ckpt'),
                                    pl.callbacks.LearningRateMonitor('epoch')])
    '''
    #linear_model.save_customized_checkpoint(os.path.join(checkpoint_path,"init.ckpt"))
    if mode == "load_last_pretrained_epoch":
        trainer = pl.Trainer(default_root_dir=checkpoint_path,
                         accelerator="gpu",
                         devices=gpu_per_nodes,
                         num_nodes=num_nodes,
                         strategy=strategy,
                         max_epochs=max_epochs,
                         precision=precision,
                         callbacks=[pl.callbacks.ModelCheckpoint(monitor = "val_acc",
                                                                mode = "max",
                                                                dirpath=os.path.join(checkpoint_path,"_temp"),
                                                                filename = 'LC.ckpt'),
                                    pl.callbacks.LearningRateMonitor('epoch')])
        trainer.logger._default_hp_metric = False 
        ssl_model = CLAP.load_from_checkpoint(os.path.join(ssl_ckpt_path,"last.ckpt"))
        ssl_model.backbone.remove_projection_header()
        linear_model.set_backbone(ssl_model.backbone)
        trainer.fit(linear_model, train_loader,val_loader)
        test_output = trainer.test(linear_model,test_loader)
        temp_dir = os.path.join(checkpoint_path,"_temp")
        subprocess.check_output(["mv " + os.path.join(temp_dir,"*") + " " + checkpoint_path], text=True,shell=True)  
        subprocess.check_output(["rm", "-r", os.path.join(temp_dir)], text=True) 
        result = {"best_training_loss":test_output[0]["test_loss"],
                  "best_test_acc1":test_output[0]["test_acc1"],
                  "best_test_acc5":test_output[0]["test_acc5"]
                  }
        with open(os.path.join(checkpoint_path,"results.json"),"w") as fs:
            json.dump(result,fs,indent=4)
    elif mode == "scan_pretrained_epochs":
        best_dir = os.path.join(checkpoint_path,"_best_model")
        temp_dir = os.path.join(checkpoint_path,"_temp")
        os.makedirs(best_dir)
        os.makedirs(temp_dir)
        files = get_top_n_latest_checkpoints(ssl_ckpt_path,5)
        result = {"best_ssl_model_path":"",
                  "best_training_loss":torch.finfo(torch.float16).max,
                  "best_test_acc1":-1.0,
                  "best_test_acc5":-1.0,
                  "best_training_version":-1
                }
        training_version = 0
        for f in files:
            trainer = pl.Trainer(default_root_dir=checkpoint_path,
                         accelerator="gpu",
                         devices=gpu_per_nodes,
                         num_nodes=num_nodes,
                         strategy=strategy,
                         precision=precision,
                         max_epochs=max_epochs,
                         callbacks=[pl.callbacks.ModelCheckpoint(monitor = "val_acc",
                                                                mode = "max",
                                                                dirpath=os.path.join(checkpoint_path,"_temp"),
                                                                filename = 'LC'),
                                    pl.callbacks.LearningRateMonitor('epoch')])
            trainer.logger._default_hp_metric = False 
            print("loading ssl model from " + f)
            ssl_model = CLAP.load_from_checkpoint(os.path.join(ssl_ckpt_path,f))
            ssl_model.backbone.remove_projection_header()
            #linear_model.load_from_checkpoint(os.path.join(checkpoint_path,"init_ckpt")))
            if os.path.isfile(os.path.join(checkpoint_path,"init.ckpt")):
                linear_model.load_from_customized_checkpoint(os.path.join(checkpoint_path,"init.ckpt"))
            linear_model.set_backbone(ssl_model.backbone)
            trainer.fit(linear_model, train_loader,val_loader)
            test_output = trainer.test(linear_model,test_loader)
            if result["best_training_loss"]> test_output[0]["test_loss"]:
                subprocess.check_output(["rm", "-r", best_dir], text=True)
                subprocess.check_output(["mv",os.path.join(temp_dir),best_dir], text=True)
                os.makedirs(temp_dir)
                result["best_ssl_model_path"] = os.path.join(ssl_ckpt_path,f)
                result["best_training_loss"] = test_output[0]["test_loss"]
                result["best_test_acc1"] = test_output[0]["test_acc1"]
                result["best_test_acc5"] = test_output[0]["test_acc5"]
                result["best_training_version"] = training_version
            else:
                subprocess.check_output(["rm", "-r", temp_dir], text=True)
                os.makedirs(temp_dir)
            training_version += 1
        subprocess.check_output(["mv " + os.path.join(best_dir,"*") + " " + checkpoint_path], text=True,shell=True)   
        linear_model.load_from_customized_checkpoint(os.path.join(checkpoint_path,"LC.ckpt")) # Load best checkpoint after training
        with open(os.path.join(checkpoint_path,"results.json"),"w") as fs:
            json.dump(result,fs,indent=4)
        ssl_model = CLAP.load_from_checkpoint(result["best_ssl_model_path"])
        linear_model.set_backbone(ssl_model.backbone)
    else:
        raise NotImplementedError("mode = {} is not implemented".format(mode))
        
    return linear_model
    
        

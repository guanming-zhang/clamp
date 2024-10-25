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
class CLAP(pl.LightningModule):
    def __init__(self,embedded_dim:int,backbone_name:str,use_projection_header:bool,proj_dim:int,
                 optim_name:str,lr:float,momentum:float,weight_decay:float,eta:float, 
                 warmup_epochs:int,n_epochs:int,
                 n_views:int,batch_size:int,lw0:float,lw1:float,lw2:float,n_pow_iter:int=20,rs:float=2.0,margin:float=1e-7):
        super().__init__()
        # all the hyperparameters are added as attributes to this class
        self.save_hyperparameters()
        self.backbone = models.BackboneNet(embedded_dim,backbone_name,use_projection_header,proj_dim)
        self.loss_fn = loss_module.EllipsoidPackingLoss(n_views,batch_size,lw0,lw1,lw2,n_pow_iter,rs,margin)
        self.train_epoch_loss = []  # To store epoch loss for training
        self.train_step_outputs = []
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
                                  eta = self.hparams.eta,
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
    
    def on_training_epoch_end(self):
        # `outputs` is a list of losses from each batch returned by training_step()
        avg_loss = torch.stack([x['loss'] for x in self.train_step_outputs]).mean()  # Compute the average loss for the epoch
        self.log('train_epoch_loss', avg_loss, prog_bar=True)  # Log epoch loss

        # Save epoch loss for future reference
        self.train_epoch_loss.append(avg_loss.item())

class LinearClassification(pl.LightningModule):
    def __init__(self,backbone:torch.nn.modules,
                 in_dim:int,out_dim:int,use_batch_norm:bool,
                 optim_name:str,lr:float,momentum:float,weight_decay:float,
                 n_epochs:int):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone
        if use_batch_norm:
            self.linear_net = models.BnLinearNet(in_dim,out_dim)
        else:
            self.linear_net = torch.nn.Linear(in_dim,out_dim)
        self.backbone.remove_projection_header()
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.train_step_outputs = []
        self.test_step_outputs = []
        self.train_epoch_loss = []
        self.test_acc1 = 0.0
        self.test_acc5 = 0.0
    
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

    def on_training_epoch_end(self):
        # `outputs` is a list of losses from each batch returned by training_step()
        avg_loss = torch.stack([x['loss'] for x in self.train_step_outputs]).mean()  # Compute the average loss for the epoch
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
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                   milestones=[self.hparams.n_epochs*0.6,
                                                               self.hparams.n_epochs*0.8],gamma=0.1)
        return [optimizer],[scheduler]


def train_clap(model:pl.LightningModule, train_loader: torch.utils.data.DataLoader,
            max_epochs:int,every_n_epochs:int,checkpoint_path:str):
    trainer = pl.Trainer(default_root_dir=checkpoint_path,
                         accelerator="gpu",
                         devices=1,
                         max_epochs=max_epochs,
                         callbacks=[pl.callbacks.ModelCheckpoint(save_weights_only=True,
                                                                  save_top_k = -1,
                                                                  save_last = True,
                                                                  every_n_epochs = every_n_epochs,
                                                                  dirpath=checkpoint_path,
                                                                  filename = 'CLAP.ckpt'),
                                    pl.callbacks.LearningRateMonitor('epoch')])
    
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

def train_lc(model:pl.LightningModule, train_loader: torch.utils.data.DataLoader,
            test_loader:torch.utils.data.DataLoader,max_epochs:int,every_n_epochs:int,checkpoint_path:str):
    trainer = pl.Trainer(default_root_dir=checkpoint_path,
                         accelerator="gpu",
                         devices=1,
                         max_epochs=max_epochs,
                         callbacks=[pl.callbacks.ModelCheckpoint(save_weights_only=True,
                                                                  save_top_k = -1,
                                                                  save_last = True,
                                                                  every_n_epochs = every_n_epochs,
                                                                  dirpath=checkpoint_path,
                                                                  filename = 'LC.ckpt'),
                                    pl.callbacks.LearningRateMonitor('epoch')])
    
    trainer.logger._default_hp_metric = False 
    '''
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(checkpoint_path, 'last.ckpt')
    if os.path.isfile(pretrained_filename):
        print(f'Found pretrained model at {pretrained_filename}, loading...')
        model = LinearClassification.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
    else:
        pl.seed_everything(137) # To be reproducable
        trainer.fit(model, train_loader)
        model = LinearClassification.load_from_checkpoint(os.path.join(checkpoint_path,"last.ckpt")) # Load best checkpoint after training
    '''
    trainer.test(model,test_loader)
    return model
        

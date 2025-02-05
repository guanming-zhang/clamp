import os
import torch
import json
import torchvision

class BackboneNet(torch.nn.Module):
    def __init__(self,resnet_type:str="resnet18",prune:bool=False,
                 use_projection_head=False,proj_dim = -1, proj_out_dim=-1):
        super().__init__()
        self.model_name = "BackboneNet"
        # this is a sloppy way of recording hyperperameters
        self.hyper_parameters = {"resnet_type":resnet_type,
                                "use_projection_head":use_projection_head,
                                "proj_out_dim":proj_out_dim}
        if resnet_type == "resnet18":
            # the fc layer dim for resnet18 is 512*num_classes
            self.net = torchvision.models.resnet18()
            self.feature_dim = 512
        elif resnet_type == "resnet34":
            # the fc layer dim for resnet34 is 512*num_classes
            self.net = torchvision.models.resnet34()
            self.feature_dim = 512
        elif resnet_type == "resnet50":
            # the fc layer dim for resnet18 is 2048*num_classes
            self.net = torchvision.models.resnet50()
            self.feature_dim = 2048
        self.net.fc = torch.nn.Identity()
        if use_projection_head:
            if proj_out_dim < 0:
                raise ValueError("proj_dim must be larger than 0")
            if isinstance(proj_dim,list):
                network = [torch.nn.Linear(self.feature_dim,proj_dim[0]),torch.nn.ReLU()]
                for i in range(len(proj_dim)-1):
                    network.append(torch.nn.Linear(proj_dim[i],proj_dim[i+1]))
                    network.append(torch.nn.ReLU())
                network.append(torch.nn.Linear(proj_dim[-1],proj_out_dim))
                self.projection_head = torch.nn.Sequential(*network)
            else:
                self.projection_head = torch.nn.Sequential(
                        torch.nn.Linear(self.feature_dim,proj_dim),
                        torch.nn.ReLU(),
                        torch.nn.Linear(proj_dim,proj_out_dim)
                    )
        else:
            self.projection_head = torch.nn.Identity()
        # if the image is small, such as CIFAR10, MNIST, we need to prune the network
        if prune:
            self._remove_maxpool()
            self._replace_conv1()

    
    def remove_projection_head(self):
        self.projection_head = torch.nn.Identity()
    
    def _remove_maxpool(self):
        # remove the max pooling for CIFAR10 dataset
        self.net.maxpool = torch.nn.Identity()
    def _replace_conv1(self):
        # repalce the conv1 for CIFAR10 dataset
        self.net.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False) 
    
    def forward(self,x):
        return self.projection_head(self.net(x))

class BnLinearNet(torch.nn.Module):#
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.model_name = "LinearNet"
        # this is a sloppy way of recording hyperperameters
        self.hyper_parameters = {"in_dim":in_dim,"out_dim":out_dim}
        for k,v in self.hyper_parameters.items():
            setattr(self,k,v)
        self.bn = torch.nn.BatchNorm1d(num_features=in_dim)
        self.net = torch.nn.Linear(in_features=in_dim,out_features=out_dim)
    def forward(self,x):
        return self.net(self.bn(x))
    









    

        

        

import os
import torch
import json
import torchvision

class BackboneNet(torch.nn.Module):
    def __init__(self,resnet_type:str="resnet18",use_projection_header=False,backbone_out_dim,proj_out_dim=-1):
        super().__init__()
        self.model_name = "BackboneNet"
        # this is a sloppy way of recording hyperperameters
        self.hyper_parameters = {"resnet_type":resnet_type,
                                "use_projection_header":use_projection_header,
                                "backbone_out_dim":backbone_out_dim,
                                "proj_out_dim":proj_out_dim}
        if resnet_type == "resnet18":
            # the fc layer dim for resnet18 is 512*num_classes
            self.net = torchvision.models.resnet18(num_classes = backbone_out_dim)
        elif resnet_type == "resnet34":
            # the fc layer dim for resnet34 is 2048*num_classes
            self.net = torchvision.models.resnet34(num_classes = backbone_out_dim)
        elif resnet_type == "resnet50":
            # the fc layer dim for resnet18 is 2048*num_classes
            self.net = torchvision.models.resnet50(num_classes = backbone_out_dim)
        if use_projection_header:
            if proj_out_dim < 0:
                raise ValueError("proj_dim must be larger than 0")
            self.projection_header = torch.nn.Sequential(
                        torch.nn.ReLU(),
                        torch.nn.Linear(backbone_out_dim,proj_out_dim)
                    )
        else:
            self.projection_header = torch.nn.Identity()
    
    def remove_projection_header(self):
        self.projection_header = torch.nn.Identity()
    
    def remove_maxpool(self):
        # remove the max pooling for CIFAR10 dataset
        self.net.maxpool = torch.nn.Identity()
    def replace_conv1(self):
        # repalce the conv1 for CIFAR10 dataset
        self.net.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False) 
    

    def forward(self,x):
        return self.projection_header(self.net(x))

class BnLinearNet(torch.nn.Module):
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
    









    

        

        

import os
import torch
import json
import torchvision

class BackboneNet(torch.nn.Module):
    def __init__(self,embedded_dim:int,resnet_type:str="resnet18",use_projection_header=False,proj_dim=-1):
        super().__init__()
        self.model_name = "BackboneNet"
        # this is a sloppy way of recording hyperperameters
        self.hyper_parameters = {"embedded_dim":embedded_dim,"resnet_type":resnet_type,
                                "use_projection_header":use_projection_header,
                                "proj_dim":proj_dim}
        if resnet_type == "resnet18":
            self.net = torchvision.models.resnet18(num_classes = embedded_dim)
        elif resnet_type == "resnet34":
            self.net = torchvision.models.resnet34(num_classes = embedded_dim)
        elif resnet_type == "resnet50":
            self.net = torchvision.models.resnet50(num_classes = embedded_dim)
        if use_projection_header:
            if proj_dim < 0:
                raise ValueError("proj_dim must be larger than 0")
            self.projection_header = torch.nn.Sequential(
                        torch.nn.Linear(embedded_dim,proj_dim),
                        torch.nn.ReLU(),
                        torch.nn.Linear(proj_dim,embedded_dim)
                    )
        else:
            self.projection_header = torch.nn.Identity()
    
    def remove_projection_header(self):
        self.projection_header = torch.nn.Identity()

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
    









    

        

        

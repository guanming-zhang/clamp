import torch
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import random_split,Dataset
from torchvision import datasets
from torchvision import transforms
import os
import time

def show_images(imgs,nrow,ncol,titles = None):
    '''
    --args 
    imgs: a list of images(PIL or torch.tensor or numpy.ndarray)
    nrow: the number of rows
    ncol: the number of columns
    titles: the tile of each subimages
    note that the size an image represented by PIL or ndarray is (W*H*C),
              but for tensor it is (C*W*H)
    --returns
    fig and axes
    '''
    fig,axes = plt.subplots(nrow,ncol)
    for i in range(min(nrow*ncol,len(imgs))):
        row  = i // ncol
        col = i % ncol
        if titles:
            axes[row,col].set_title(titles[i])
        if isinstance(imgs[i],Image.Image):
            img = np.array(imgs[i])
        elif torch.is_tensor(imgs[i]):
            img = imgs[i].cpu().detach()
            img = img.permute((1,2,0)).numpy()
        elif isinstance(imgs[i], np.ndarray):
            img = imgs[i]
        else:
            raise TypeError("each image must be an PIL or torch.tensor or numpy.ndarray")
        axes[row,col].imshow(img)
        axes[row,col].set_axis_off()
        fig.tight_layout()
    return fig,axes

#####################################
# image augumentation
#####################################
class AugmentationTrans(object):
    '''
    for image augumentation
    '''
    def __init__(self, my_transforms, n_views=1):
        '''
        --args:
        my_transforms: torchvison.transforms object that transforms the image
        n_views: the number of transfomed images augmented for each orginal image
        '''
        self._transforms = my_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self._transforms(x) for i in range(self.n_views)]

class WrappedDataset(Dataset):
    '''
    This class is designed to apply diffent transforms to subdatasets
    subdatasets are not allowed to have different transforms by default
    By wrapping subdatasets to WrappedDataset, this problem is solved
    e.g 
    _train_set, _val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_set = WrappedDataset(_train_set,transforms.RandomHorizontalFlip(), n_views=3)
    val_set = WrappedDataset(_val_set,transforms.ToTensor())
    
    If using DataLoader object(denoted as loader) to load it, 
    then for one batch of data, (x,y), 
    x is a list of n_views elements, x[i] is of size batch_size*C*H*W where x[j] is the augmented version of x[i]
    y is a list of n_views elements, y[i] is of size batch_size
    train_loader = data.DataLoader(train_dataset,batch_size = batch_size,shuffle=True)

    Additional comments: after the data augmentation, one batch 
    data,label = next(iter(train_loader))
    data is a 2D-list of images(size = [n_veiws,batch_size] each element is a (C*W*H)-tensor)
    label is is a 2D list of integers(size = n_views*batch_size element is a 1-tensor)
    The label of image data[i_view][j_img] is label[i_view][j_img]
    '''
    def __init__(self, dataset, transform=None, n_views = 1):
        self.dataset = dataset
        self.transform = transform
        self.n_views = n_views
        
    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.transform:
            x = [self.transform(x) for i in range(self.n_views)]
            y = [y for i in range(self.n_views)]
        return x, y
        
    def __len__(self):
        return len(self.dataset)

#####################################
# For CIFAR10 dataset
#####################################   
def get_cifar10_classes():
    labels = ["airplane","automobile","bird","cat",
              "deer","dog","frog","horse","ship","truck"]
    return labels

#####################################
# For STL10 dataset
#####################################   
def get_stl10_classes():
    labels = ["airplane","bird","car","cat",
              "deer","dog","horse","monkey","ship","truck"]
    return labels

def download_dataset(dataset_path,dataset_name):
    if dataset_name == "CIFAR10":
        '''
        train_dataset contains 50000 images of size 32*32*3 
        '''
        train_dataset = datasets.CIFAR10(root=dataset_path, train=True,download=True)
        test_dataset = datasets.CIFAR10(root=dataset_path, train=False,download=True)
        data_mean = (train_dataset.data / 255.0).mean(axis=(0,1,2))
        data_std = (train_dataset.data / 255.0).std(axis=(0,1,2))
        return train_dataset,test_dataset,data_mean,data_std
    else:
        raise NotImplementedError("downloading for this dataset is not implemented")


def get_dataloader(info:dict,batch_size:int,num_workers:int,validation:bool=True):
    '''
    info: a dictionary provides the information of 
          1) dataset 
             e.g. info["dataset"] = "MNIST"
          2) augmentations
             e.g. info["augmentations"] = ["RandomResizedCrop","GaussianBlur" ] 
          3) batch_size
    * the average color value for different dataset are taken from 
      a)cifar10 & mnist https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
      b)imagenet https://pytorch.org/vision/stable/transforms.html
    '''
    # the default mean and average are assumed to be natural images such as imagenet 
    # therefore the default mean and std are as follow
    mean= [0.485, 0.456, 0.406]
    std= [0.229, 0.224, 0.225]
    if info["dataset"] == "MNIST01":
        data_dir = "./datasets/mnist"
        train_dataset = datasets.MNIST(data_dir,train = True,download = True)
        test_dataset = datasets.MNIST(data_dir,train = False,download = True)
        # select 0 and 1 from the trainning dataset
        train_indices = torch.where(torch.logical_or(train_dataset.targets == 0,train_dataset.targets == 1))
        train_dataset = torch.utils.data.Subset(train_dataset,train_indices[0])
        # select 0 and 1 from the test dataset
        test_indices = torch.where(torch.logical_or(test_dataset.targets == 0,test_dataset.targets == 1))
        test_dataset = torch.utils.data.Subset(test_dataset,test_indices[0])
        if validation:
            train_dataset,val_dataset = torch.utils.data.random_split(train_dataset,[0.9,0.1])
    if info["dataset"] == "MNIST":
        mean = [0.131,0.131,0.131]
        std = [0.308,0.308,0.308]
        data_dir = "./datasets/mnist"
        train_dataset = datasets.MNIST(data_dir,train = True,download = True)
        test_dataset = datasets.MNIST(data_dir,train = False,download = True)
        if validation:
            train_dataset,val_dataset = torch.utils.data.random_split(train_dataset,[0.9,0.1])
    elif info["dataset"] == "CIFAR10":
        data_dir = "./datasets/cifar10"
        mean = [0.491,0.482,0.446]
        std = [0.247,0.243,0.261]
        train_dataset = datasets.CIFAR10(root=data_dir, train=True,download=True)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False,download=True)
        if validation:
            train_dataset,val_dataset = torch.utils.data.random_split(train_dataset,[0.9,0.1])
    elif info["dataset"] == "FLOWERS102":
        train_dataset = datasets.Flowers102(root=data_dir,split="train",download=True)
        test_dataset = datasets.Flowers102(root=data_dir,split="test",download=True)
        val_dataset = datasets.Flowers102(root=data_dir,split="val",download=True)
    elif info["dataset"] == "FOOD101":
        train_dataset = datasets.Food101(root=data_dir,split="train",download=True)
        test_dataset = datasets.Food101(root=data_dir,split="test",download=True)
        val_dataset = datasets.Food101(root=data_dir,split="val",download=True)
    elif info["dataset"] == "PascalVOC":
        train_dataset = datasets.VOCDetection(root=data_dir,image_set="train",download=True)
        test_dataset = datasets.VOCDetection(root=data_dir,image_set="test",download=True)
        val_dataset = datasets.VOCDetection(root=data_dir,image_set="val",download=True)

    elif info["dataset"] == "IMAGENET1K":
        data_dir = "./datasets/imagenet1K"
        mean= [0.485, 0.456, 0.406]
        std= [0.229, 0.224, 0.225]
        train_dataset = datasets.ImageFolder(root=os.path.join(data_dir,"train"))
        test_dataset = datasets.ImageFolder(root=os.path.join(data_dir,"val"))
        if validation:
            train_dataset,val_dataset = torch.utils.data.random_split(train_dataset,[0.99,0.01])
    elif info["dataset"] == "IMAGENET1K-1%":
        data_dir = "./datasets/imagenet1K"
        mean= [0.485, 0.456, 0.406]
        std= [0.229, 0.224, 0.225]
        train_dataset = datasets.ImageFolder(root=os.path.join(data_dir,"train"))
        test_dataset = datasets.ImageFolder(root=os.path.join(data_dir,"val"))
        if validation:
            train_dataset,val_dataset = torch.utils.data.random_split(train_dataset,[0.99,0.01])
        # Desired number of images per class ~ 12.8
        num_images_per_class = 13
        num_samples = len(train_dataset)
        # draw subset_ratio shuffled indices 
        indices = torch.randperm(num_samples)[:num_images_per_class*1000]
        train_dataset = torch.utils.data.Subset(train_dataset, indices=indices)
    elif info["dataset"] == "IMAGENET1K-10%":
        data_dir = "./datasets/imagenet1K"
        mean= [0.485, 0.456, 0.406]
        std= [0.229, 0.224, 0.225]
        train_dataset = datasets.ImageFolder(root=os.path.join(data_dir,"train"))
        test_dataset = datasets.ImageFolder(root=os.path.join(data_dir,"val"))
        if validation:
            train_dataset,val_dataset = torch.utils.data.random_split(train_dataset,[0.95,0.05])
        # Desired number of images per class ~ 128
        num_images_per_class = 128
        num_samples = len(train_dataset)
        # draw subset_ratio shuffled indices 
        indices = torch.randperm(num_samples)[:num_images_per_class*1000]
        train_dataset = torch.utils.data.Subset(train_dataset, indices=indices)
   
    trans_list = [transforms.ToTensor()]
    
    if info["dataset"] == "MNIST01" or info["dataset"]=="MNIST":
        trans_list.append(transforms.Lambda(lambda x:x.repeat(3,1,1)))# 3 channels
    
    if "RandomResizedCrop" in info["augmentations"]:
        trans_list.append(transforms.RandomResizedCrop(info["crop_size"],scale=(info["crop_min_scale"],info["crop_max_scale"])))
    if "ColorJitter" in info["augmentations"]:
        trans_list.append(transforms.RandomApply([transforms.ColorJitter(
                                                    brightness=info["jitter_brightness"],
                                                    contrast=info["jitter_contrast"],
                                                    saturation=info["jitter_saturation"],
                                                    hue=info["jitter_hue"])],p=info["jitter_prob"]))
    if "RandomGrayscale" in info["augmentations"]:
        trans_list.append(transforms.RandomGrayscale(p=info["grayscale_prob"]))
    if "GaussianBlur" in info["augmentations"]:
        trans_list.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=info["blur_kernel_size"])],p=info["blur_prob"]))
    if "RandomHorizontalFlip" in info["augmentations"]:
        trans_list.append(transforms.RandomHorizontalFlip(p=info["hflip_prob"]))
    #trans_list.append(transforms.ToTensor())
    trans_list.append(transforms.Normalize(mean=mean,std=std))
   
    aug_transforms = transforms.Compose(trans_list)
    if info["dataset"] == "MNIST01" or info["dataset"]=="MNIST":
        norm_transforms = transforms.Compose([transforms.ToTensor(),
                            transforms.Lambda(lambda x:x.repeat(3,1,1)),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    else:
        norm_transforms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=mean,std=std)])
    train_dataset = WrappedDataset(train_dataset,aug_transforms,n_views = info["n_views"])
    test_dataset = WrappedDataset(test_dataset,norm_transforms)
    if validation:
        val_dataset = WrappedDataset(val_dataset,norm_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = batch_size,shuffle=True,drop_last=True,num_workers=num_workers,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size = batch_size,shuffle=False,drop_last=True,num_workers = num_workers,pin_memory=True)
    if validation:
        val_loader = torch.utils.data.DataLoader(test_dataset,batch_size = batch_size,shuffle=False,drop_last=True,num_workers = num_workers,pin_memory=True)
    else:
        val_loader = None
    return train_loader,test_loader,val_loader

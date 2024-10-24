import torch
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import random_split,Dataset
from torchvision import datasets
from torchvision import transforms
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


def get_dataloader(info:dict,ssl_batch_size:int,lc_batch_size:int,num_workers:int):
    '''
    info: a dictionary provides the information of 
          1) dataset 
             e.g. info["dataset"] = "MNIST"
          2) augmentations
             e.g. info["augmentations"] = ["RandomResizedCrop","GaussianBlur" ] 
          3) batch_size
    '''
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
    elif info["dataset"] == "MNIST":
        data_dir = "./datasets/mnist"
        train_dataset = datasets.MNIST(data_dir,train = True,download = True)
        test_dataset = datasets.MNIST(data_dir,train = False,download = True)
    elif info["dataset"] == "CIFAR10":
        data_dir = "./datasets/cifar10"
        train_dataset = datasets.CIFAR10(root=data_dir, train=True,download=True)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False,download=True)
        
    trans_list = [transforms.ToTensor()]
    
    if info["dataset"] == "MNIST01" or info["dataset"]=="MNIST":
        trans_list.append(transforms.Lambda(lambda x:x.repeat(3,1,1)))# 3 channels
    
    if "RandomResizedCrop" in info["augmentations"]:
        trans_list.append(transforms.RandomResizedCrop(info["crop_size"]))
    if "ColorJitter" in info["augmentations"]:
        trans_list.append(transforms.RandomApply([transforms.ColorJitter(
                                                    brightness=info["jitter_brightness"],
                                                    contrast=info["jitter_contrast"],
                                                    saturation=info["jitter_saturation"],
                                                    hue=info["jitter_hue"])],p=info["jitter_prob"]))
    if "RandomGrayscale" in info["augmentations"]:
        trans_list.append(transforms.RandomGrayscale(p=info["grayscale_prob"]))
    if "GaussianBlur" in info["augmentations"]:
        trans_list.append(transforms.GaussianBlur(kernel_size=info["blur_kernel_size"]))
    #trans_list.append(transforms.ToTensor())
    trans_list.append(transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)))
   
    aug_transforms = transforms.Compose(trans_list)
    if info["dataset"] == "MNIST01" or info["dataset"]=="MNIST":
        norm_transforms = transforms.Compose([transforms.ToTensor(),
                            transforms.Lambda(lambda x:x.repeat(3,1,1)),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    else:
        norm_transforms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    ssl_train_dataset = WrappedDataset(train_dataset,aug_transforms,n_views = info["n_views"])
    lc_train_dataset = WrappedDataset(train_dataset,norm_transforms)
    test_dataset = WrappedDataset(test_dataset,norm_transforms)
  
    ssl_train_loader = torch.utils.data.DataLoader(ssl_train_dataset,batch_size = ssl_batch_size,shuffle=True,drop_last=True,num_workers=num_workers)
    lc_train_loader = torch.utils.data.DataLoader(lc_train_dataset,batch_size = lc_batch_size,shuffle=True,drop_last=True,num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size = lc_batch_size,shuffle=True,drop_last=True,num_workers = num_workers)

    return ssl_train_loader,lc_train_loader,test_loader

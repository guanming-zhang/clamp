import torch
from matplotlib import pyplot as plt
import numpy as np
import re
from PIL import Image
from torch.utils.data import random_split,Dataset
from torchvision import datasets
from torchvision.transforms import v2
from .lmdb_dataset import ImageFolderLMDB
import cv2 
import albumentations as A
from albumentations.pytorch import ToTensorV2
import urllib.request
import os
# this dataset loads images into numpy array format
# the default dataset loads images into PIL format
# credit to
# https://github.com/albumentations-team/autoalbument/blob/master/examples/cifar10/dataset.py
class Cifar10SearchDataset(datasets.CIFAR10):
    def __init__(self, root="./data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

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

class WrappedDataset(Dataset):
    '''
    This class is designed to apply diffent transforms to subdatasets
    subdatasets are not allowed to have different transforms by default
    By wrapping subdatasets to WrappedDataset, this problem is solved
    e.g 
    _train_set, _val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_set = WrappedDataset(_train_set,transforms.RandomHorizontalFlip(), n_views=3)
    val_set = WrappedDataset(_val_set,transforms.ToTensor())
    Parameters:
    dataset: dataset for training/testing
    transforms: a list of transforms
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
    def __init__(self, dataset, transforms=None, n_views = 1, aug_pkg = "torchvision"):
        self.dataset = dataset
        self.transforms = transforms
        self.n_trans = len(transforms)
        self.n_views = n_views
        self.aug_pkg = aug_pkg
        if not aug_pkg in ["torchvision","albumentations"]:
            raise NotImplemented("augmentation from package [" + aug_pkg +"] is not implemented")
    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.transforms and self.aug_pkg == "torchvision":
            x = [self.transforms[i % self.n_trans](x) for i in range(self.n_views)]
            y = [y for i in range(self.n_views)]
        elif self.transforms and self.aug_pkg == "albumentations":
            if type(x) is Image.Image:
                x = np.array(x)
            x = [self.transforms[i % self.n_trans](image=x)["image"] for i in range(self.n_views)]
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

def download_dataset(dataset_path,dataset_name):
    if dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(root=dataset_path, train=True,download=True)
        test_dataset = datasets.CIFAR10(root=dataset_path, train=False,download=True)
        data_mean = (train_dataset.data / 255.0).mean(axis=(0,1,2))
        data_std = (train_dataset.data / 255.0).std(axis=(0,1,2))
        return train_dataset,test_dataset,data_mean,data_std
    else:
        raise NotImplementedError("downloading for this dataset is not implemented")

def get_transform(aug_ops:list,aug_params:dict,aug_pkg="torchvision"):
    '''
    aug_ops : augmentation operations e.g. ["RandomGrayscale","GaussianBlur","RandomHorizontalFlip"]
    aug_params: aumentations parameters e.g. {"jitter_brightness":2,"mean4norm":[0,1,2]}
    aug_pkg: package for image augmentaions, either "torchvision" or "albumentations"
    '''
    # sanity check for image augmentaion
    avaiable_augs = ["RandomResizedCrop","ColorJitter","RandomGrayscale","GaussianBlur","RandomHorizontalFlip",
                     "RandomSolarize","ToNumpyArr","ToTensor","Normalize","RepeatChannel"]
    for aug in aug_ops:
        if not aug in avaiable_augs:
            raise ValueError(aug + " is not avaible for augmention")
    if aug_pkg == "torchvision":
        trans_list = []
        for aug in aug_ops:
            if aug == "RandomResizedCrop":
                trans_list.append(v2.RandomResizedCrop(aug_params["crop_size"],
                                                       scale=(aug_params["crop_min_scale"],aug_params["crop_max_scale"])))
            elif aug == "ColorJitter":
                trans_list.append(v2.RandomApply([v2.ColorJitter(
                                                    brightness=aug_params["jitter_brightness"],
                                                    contrast=aug_params["jitter_contrast"],
                                                    saturation=aug_params["jitter_saturation"],
                                                    hue=aug_params["jitter_hue"])],p=aug_params["jitter_prob"]))
            elif aug == "RandomGrayscale":
                trans_list.append(v2.RandomGrayscale(p=aug_params["grayscale_prob"]))
            elif aug == "GaussianBlur":
                trans_list.append(v2.RandomApply([v2.GaussianBlur(kernel_size=aug_params["blur_kernel_size"])],
                                                 p=aug_params["blur_prob"]))
            elif aug == "RandomHorizontalFlip":
                trans_list.append(v2.RandomHorizontalFlip(p=aug_params["hflip_prob"]))
            elif aug == "RandomSolarize":
                trans_list.append(v2.RandomSolarize(threshold=0.5,p=aug_params["solarize_prob"]))
            elif aug == "ToTensor":
                trans_list.append(v2.ToImage())
                trans_list.append(v2.ToDtype(torch.float32,scale=True))
            elif aug == "Normalize":
                trans_list.append(v2.Normalize(mean=aug_params["mean4norm"],std=aug_params["std4norm"]))
            elif aug == "ToNumpyArr":
                trans_list.append(v2.Lambda(lambda pillow_img:np.array(pillow_img)))
            elif aug == "RepeatChannel":
                trans_list.append(v2.Lambda(lambda x:x.repeat(3,1,1)))
        return v2.Compose(trans_list)
    elif aug_pkg == "albumentations":
        trans_list = []
        for aug in aug_ops:
            if aug == "RandomResizedCrop":
                trans_list.append(A.RandomResizedCrop(size=(aug_params["crop_size"],aug_params["crop_size"]),
                                                    scale=(aug_params["crop_min_scale"],
                                                    aug_params["crop_max_scale"])) )
            elif aug == "ColorJitter":
                trans_list.append(A.ColorJitter(brightness=aug_params["jitter_brightness"],
                                                contrast=aug_params["jitter_contrast"],
                                                saturation=aug_params["jitter_saturation"],
                                                hue=aug_params["jitter_hue"],
                                                p=aug_params["jitter_prob"]))
            elif aug == "RandomGrayscale":
                trans_list.append(A.ToGray(p=aug_params["grayscale_prob"]))
            elif aug == "GaussianBlur":
                trans_list.append(A.GaussianBlur(blur_limit=(aug_params["blur_kernel_size"],aug_params["blur_kernel_size"]),
                                                 sigma_limit=(0.1, 2.0),
                                                 p=aug_params["blur_prob"]))
            elif aug == "RandomHorizontalFlip":
                trans_list.append(A.HorizontalFlip(p=aug_params["hflip_prob"]))
            elif aug == "RandomSolarize":
                trans_list.append(A.Solarize(p=aug_params["solarize_prob"]))
            elif aug == "ToTensor":
                trans_list.append(ToTensorV2())
            elif aug == "Normalize":
                trans_list.append(A.Normalize(mean=aug_params["mean4norm"],std=aug_params["std4norm"]))
            elif aug == "ToNumpyArr":
                trans_list.append(A.Lambda(lambda pillow_img:np.array(pillow_img)))
            elif aug == "RepeatChannel":
                trans_list.append(A.Lambda(lambda x:x.repeat(3,1,1)))
        return A.Compose(trans_list)

def get_dataloader(info:dict,batch_size:int,num_workers:int,
                   standardized_to_imagenet:bool=False,
                   augment_val_set=False,
                   prefetch_factor:int=2,
                   aug_pkg:str="torchvision",
                    skip_validation:bool=False):
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
    if not aug_pkg in ["torchvision","albumentations"]:
            raise NotImplemented("augmentation from package [" + aug_pkg +"] is not implemented")
    # set the default transform operation list
    # if the images are loaded as PIL, it needs to be converted into numpy if aug_ops == "albumentations"
    if info["dataset"] != "CIFAR10" and not "IMAGENET" in info["dataset"] and aug_pkg == "albumentations":
        aug_pkg = "torchvision"
        print("augmantiation method is set to [torchvision]")
        print("[albumentations] only support CIFAR10 or IMAGENET for now")
    # initialize tranformations
    train_aug_ops = [[] for i in range(info["n_trans"])]
    train_aug_params = [dict() for i in range(info["n_trans"])] 
    for i in range(info["n_trans"]):
        if aug_pkg == "torchvision":
            train_aug_ops[i] = info["augmentations"] + ["ToTensor","Normalize"]
        else:
            train_aug_ops[i] = info["augmentations"] + ["Normalize","ToTensor"]
    if aug_pkg == "albumentations":
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
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
        if not skip_validation:
            train_dataset,val_dataset = torch.utils.data.random_split(train_dataset,[0.9,0.1])
        print("only one tranform is applied for MNIST01 toy model")
        train_aug_ops = [["ToTensor","RepeatChannel"] + info["augmentations"] + ["Normalize"]]
    if info["dataset"] == "MNIST":
        mean = [0.131,0.131,0.131]
        std = [0.308,0.308,0.308]
        data_dir = "./datasets/mnist"
        train_dataset = datasets.MNIST(data_dir,train = True,download = True)
        test_dataset = datasets.MNIST(data_dir,train = False,download = True)
        if not skip_validation:
            train_dataset,val_dataset = torch.utils.data.random_split(train_dataset,[0.9,0.1])
        print("only one tranform is applied for MNIST toy model")
        train_aug_ops = [["ToTensor","RepeatChannel"] + info["augmentations"] + ["Normalize"]]
    elif info["dataset"] == "CIFAR10":
        data_dir = "./datasets/cifar10"
        mean = [0.491,0.482,0.446]
        std = [0.247,0.243,0.261]
        if aug_pkg == "torchvision":
            train_dataset = datasets.CIFAR10(root=data_dir, train=True,download=True)
            test_dataset = datasets.CIFAR10(root=data_dir, train=False,download=True)
        else:
            train_dataset = Cifar10SearchDataset(root=data_dir, train=True,download=True)
            test_dataset = Cifar10SearchDataset(root=data_dir, train=False,download=True)
        train_dataset,val_dataset = torch.utils.data.random_split(train_dataset,[0.95,0.05])
    elif info["dataset"] == "CIFAR100":
        data_dir = "./datasets/cifar100"
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        train_dataset = datasets.CIFAR100(root=data_dir, train=True,download=True)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False,download=True)
        train_dataset,val_dataset = torch.utils.data.random_split(train_dataset,[0.95,0.05])
    elif info["dataset"] == "FLOWERS102":
        data_dir = "./datasets/flower102"
        # use std and mean for imagenet for transfer learning datasets
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        train_dataset = datasets.Flowers102(root=data_dir,split="train",download=True)
        test_dataset = datasets.Flowers102(root=data_dir,split="test",download=True)
        val_dataset = datasets.Flowers102(root=data_dir,split="val",download=True)
    elif info["dataset"] == "FOOD101":
        data_dir = "./datasets/food101"
        # use std and mean for imagenet for transfer learning datasets
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        train_dataset = datasets.Food101(root=data_dir,split="train",download=True)
        test_dataset = datasets.Food101(root=data_dir,split="test",download=True)
        train_dataset,val_dataset = torch.utils.data.random_split(train_dataset,[0.95,0.05])
    #elif info["dataset"] == "PascalVOC":
    #    data_dir = "./datasets/pascalvoc"
    #    # use std and mean for imagenet for transfer learning datasets
    #    mean=[0.485, 0.456, 0.406]
    #    std=[0.229, 0.224, 0.225]
    #    train_dataset = datasets.VOCDetection(root=data_dir,image_set="train",year="2007",download=True)
    #    test_dataset = datasets.VOCDetection(root=data_dir,image_set="test",year="2007",download=True)
    #    val_dataset = datasets.VOCDetection(root=data_dir,image_set="val",year="2007",download=True)
    elif info["dataset"] == "DTD":
        data_dir = "./datasets/dtd"
        # use std and mean for imagenet for transfer learning datasets
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        train_dataset = datasets.DTD(root=data_dir,split="train",download=True)
        test_dataset = datasets.DTD(root=data_dir,split="test",download=True)
        val_dataset = datasets.DTD(root=data_dir,split="val",download=True)
    elif info["dataset"] == "IMAGENET1K":
        train_dir = info["imagenet_train_dir"]
        val_dir = info["imagenet_val_dir"]
        mean= [0.485, 0.456, 0.406]
        std= [0.229, 0.224, 0.225]
        if train_dir.endswith("lmdb") and val_dir.endswith("lmdb"):
            img_type = "PIL" if aug_pkg=="torchvision" else "Numpy"
            train_dataset = ImageFolderLMDB(train_dir,img_type=img_type)
            test_dataset = ImageFolderLMDB(val_dir,img_type=img_type)
        elif aug_pkg == "albumentations":
            train_dataset = datasets.ImageFolder(root=train_dir,
                                                loader = lambda img_path:cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB))
            test_dataset = datasets.ImageFolder(root=val_dir,
                                                loader = lambda img_path:cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB))
        elif aug_pkg == "torchvision":
            train_dataset = datasets.ImageFolder(root=train_dir)
            test_dataset = datasets.ImageFolder(root=val_dir)
        train_dataset,val_dataset = torch.utils.data.random_split(train_dataset,[0.99,0.01])
    elif re.search(r"IMAGENET1K-(\d+)percent", info["dataset"]):
        percentage = int(re.search(r"IMAGENET1K-(\d+)percent", info["dataset"]).group(1))  
        train_dir = info["imagenet_train_dir"]
        val_dir = info["imagenet_val_dir"]
        mean= [0.485, 0.456, 0.406]
        std= [0.229, 0.224, 0.225]
        if train_dir.endswith("lmdb") and val_dir.endswith("lmdb"):
            img_type = "PIL" if aug_pkg=="torchvision" else "Numpy"
            train_dataset = ImageFolderLMDB(train_dir,img_type=img_type)
            test_dataset = ImageFolderLMDB(val_dir,img_type=img_type)
        elif aug_pkg == "albumentations":
            train_dataset = datasets.ImageFolder(root=train_dir,
                                                loader = lambda img_path:cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB))
            test_dataset = datasets.ImageFolder(root=val_dir,
                                                loader = lambda img_path:cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB))
        elif aug_pkg == "torchvision":
            train_dataset = datasets.ImageFolder(root=train_dir)
            test_dataset = datasets.ImageFolder(root=val_dir)
        train_dataset,val_dataset = torch.utils.data.random_split(train_dataset,[0.99,0.01])
        num_images_per_class = 1280*percentage / 100.0
        num_samples = len(train_dataset)
        # draw subset_ratio shuffled indices 
        indices = torch.randperm(num_samples)[:int(num_images_per_class*1000 + 0.5)]
        train_dataset = torch.utils.data.Subset(train_dataset, indices=indices)
    elif info["dataset"] == "IMAGENET1K-simclr-1percent" or info["dataset"] == "IMAGENET1K-simclr-10percent" :
        percentage = int(re.search(r"IMAGENET1K-simclr-(\d+)percent", info["dataset"]).group(1))  
        train_dir = info["imagenet_train_dir"]
        val_dir = info["imagenet_val_dir"]
        mean= [0.485, 0.456, 0.406]
        std= [0.229, 0.224, 0.225]
        if train_dir.endswith("lmdb") and val_dir.endswith("lmdb"):
            img_type = "PIL" if aug_pkg=="torchvision" else "Numpy"
            train_dataset = ImageFolderLMDB(train_dir,img_type=img_type)
            test_dataset = ImageFolderLMDB(val_dir,img_type=img_type)
            assert "IMAGENET1K-simclr-Xprecent is not supported for lmdb currently" 
        else:
            if aug_pkg == "albumentations":
                train_dataset = datasets.ImageFolder(root=train_dir,
                                                loader = lambda img_path:cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB))
                test_dataset = datasets.ImageFolder(root=val_dir,
                                                loader = lambda img_path:cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB))
            elif aug_pkg == "torchvision":
                train_dataset = datasets.ImageFolder(root=train_dir)
                test_dataset = datasets.ImageFolder(root=val_dir)
            _dataset,val_dataset = torch.utils.data.random_split(train_dataset,[0.99,0.01])
            print("imagenet-simclr subdataset")
            print("length of valdataset")
            print(len(val_dataset))
            # code taken from https://github.com/facebookresearch/swav/blob/main/eval_semisup.py
            # take either 1% or 10% of images
            subset_file = urllib.request.urlopen("https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/" + str(percentage) + "percent.txt")
            list_imgs = [li.decode("utf-8").split('\n')[0] for li in subset_file]
            train_dataset.samples = [(
                os.path.join(train_dir, li.split('_')[0], li),
                train_dataset.class_to_idx[li.split('_')[0]]
            ) for li in list_imgs]
    elif info["dataset"] == "IMAGENET100":
        if "imagenet_train_dir" in info:
            train_dir = info["imagenet_train_dir"]
            val_dir = info["imagenet_val_dir"]
        else:
            train_dir = "./datasets/imagenet100/train.lmdb"
            val_dir =  "./datasets/imagenet100/val.lmdb"
        mean= [0.485, 0.456, 0.406]
        std= [0.229, 0.224, 0.225]
        if train_dir.endswith("lmdb") and val_dir.endswith("lmdb"):
            img_type = "PIL" if aug_pkg=="torchvision" else "Numpy"
            train_dataset = ImageFolderLMDB(train_dir,img_type=img_type)
            test_dataset = ImageFolderLMDB(val_dir,img_type=img_type)
        elif aug_pkg == "albumentations":
            train_dataset = datasets.ImageFolder(root=train_dir,
                                                loader = lambda img_path:cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB))
            test_dataset = datasets.ImageFolder(root=val_dir,
                                                loader = lambda img_path:cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB))
        elif aug_pkg == "torchvision":
            train_dataset = datasets.ImageFolder(root=train_dir)
            test_dataset = datasets.ImageFolder(root=val_dir)
        train_dataset,val_dataset = torch.utils.data.random_split(train_dataset,[0.95,0.05])
        
    # create transform for 1) testing 2) training 3)validation
    if info["dataset"] == "MNIST01" or info["dataset"]=="MNIST":
        test_transforms = [v2.Compose([v2.ToImage(),v2.ToDtype(torch.float32,scale=True),
                                      v2.Lambda(lambda x:x.repeat(3,1,1)),
                                      v2.Normalize(mean=mean,std=std)])]
    elif standardized_to_imagenet:
        test_transforms = [v2.Compose([v2.ToImage(),v2.ToDtype(torch.float32,scale=True),
                                v2.Normalize(mean=mean,std=std),
                                v2.Resize(size=256,interpolation=v2.InterpolationMode.BICUBIC),
                                v2.CenterCrop(size=224)])]
    else:
        test_transforms = [v2.Compose([v2.ToImage(),v2.ToDtype(torch.float32,scale=True),
                                     v2.Normalize(mean=mean,std=std)])]
    # get the transform for training
    train_transforms = []
    for i in range(info["n_trans"]):
        for k in info:
            if isinstance(info[k],list) and k!= "augmentations":
                train_aug_params[i][k] = info[k][i]
            train_aug_params[i]["mean4norm"] = mean 
            train_aug_params[i]["std4norm"] = std
        train_transforms.append(get_transform(train_aug_ops[i],aug_params=train_aug_params[i],aug_pkg=aug_pkg))
    train_dataset = WrappedDataset(train_dataset,train_transforms,n_views = info["n_views"],aug_pkg=aug_pkg)
    test_dataset = WrappedDataset(test_dataset,test_transforms)
    if augment_val_set:
        val_dataset = WrappedDataset(val_dataset,train_transforms,n_views = info["n_views"],aug_pkg=aug_pkg)
    else:
        val_dataset = WrappedDataset(val_dataset,test_transforms,n_views=1)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = batch_size,shuffle=True,drop_last=True,
                                               num_workers=num_workers,pin_memory=True,persistent_workers=True,prefetch_factor=prefetch_factor)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size = batch_size,shuffle=False,drop_last=True,
                                              num_workers = num_workers,pin_memory=True,persistent_workers=True,prefetch_factor=prefetch_factor)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size = batch_size,shuffle=False,drop_last=True,
                                                 num_workers = num_workers,pin_memory=True,persistent_workers=True,prefetch_factor=prefetch_factor)
    if len(train_dataset) < batch_size:
        print("Train dataset is smaller than batch size, it may cause error. Try decreasing the batch size")
    if len(val_dataset) < batch_size:
        print("Validation dataset is smaller than batch size, it may cause error. Try decreasing the batch size")
    if len(test_dataset) < batch_size:
        print("Validation dataset is smaller than batch size, it may cause error. Try decreasing the batch size")
    return train_loader,test_loader,val_loader

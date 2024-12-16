import os
import os.path as osp
from PIL import Image
from io import BytesIO
import lmdb
import pickle

import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

'''
Please read this wonderful repo
https://github.com/Lyken17/Efficient-PyTorch .
Codes in this file are adapted from this repo,
I learned quite a lot as a humble student from this repo and
would like to tribute to the author Ligeng Zhu
'''


class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path:str, transform=None,target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = pickle.loads(byteflow)

        # load img
        imgbuf = unpacked[0]
        # only for python 3
        buf =  BytesIO() 
        buf.write(imgbuf)
        # Move the cursor to the start of the buffer
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # return img, target
        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def folder2lmdb(dpath, out_folder, name="train", write_frequency=5000, num_workers=16):
    directory = osp.expanduser(osp.join(dpath, name))
    print("Loading dataset from %s" % directory)
    dataset = ImageFolder(directory, loader=raw_reader)
    data_loader = DataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x)

    lmdb_path = osp.join(out_folder, "%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)
    
    print(len(dataset), len(data_loader))
    txn = db.begin(write=True)
    for idx, data in enumerate(data_loader):
        # print(type(data), data)
        image, label = data[0]
        txn.put(u'{}'.format(idx).encode('ascii'), pickle.dumps((image, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(keys))
        txn.put(b'__len__', pickle.dumps(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str)
    parser.add_argument('-s', '--split', type=str, default="val")
    parser.add_argument('--out', type=str, default=".")
    parser.add_argument('-p', '--procs', type=int, default=20)

    args = parser.parse_args()

    folder2lmdb(args.folder, args.out, num_workers=args.procs, name=args.split)
from torch.utils.data import Dataset,Subset
import torchvision
import os
from google_drive_downloader import GoogleDriveDownloader as gdd
from sklearn.model_selection import train_test_split
import numpy as np

class chars74K(Dataset):
    def __init__(self, root_dir, train=True, download=True, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transforms = transform
        self.target_transform = target_transform
        if download:
            self.download(path=self.root_dir)

        dataset = torchvision.datasets.ImageFolder(
            root=self.root_dir, transform=transform)
        
        train_idx, valid_idx = train_test_split(np.arange(
            len(dataset.targets)), test_size=0.2, shuffle=True, stratify=dataset.targets)
        if train:
            self.dataset = Subset(dataset, train_idx)

        else:
            self.dataset = Subset(dataset, valid_idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        if self.target_transform:
            target = self.target_transform(target)
        return (img, target)

    def download(self, path):
        '''
        Download dataset if not present
        '''
        url = "https://drive.google.com/file/d/1XnA-CGAFwG-IYgbDzT5r2g5c1v2XBbcy/view?usp=sharing"
        id = url.split("/")[5]
        zipName = "chars74K.zip"
        if not os.path.isdir(path):
            ogPath = os.path.split(path)[0]
            gdd.download_file_from_google_drive(file_id=id,
                                                dest_path=os.path.join(
                                                    ogPath, zipName),
                                                unzip=True)
            os.remove(os.path.join(ogPath, zipName))
            print("Downloaded")
        else:
            print("Dataset already downloaded")

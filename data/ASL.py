from torch.utils.data import Dataset, dataset
import torchvision
import os

class ASL(Dataset):
    def __init__(self, root_dir, train=True, download=False, transforms=None, target_transform=None):
        if train:
            root_dir = os.path.join(root_dir,"train")
        else:
            root_dir = os.path.join(root_dir,"test")
        self.transforms = transforms
        self.target_transform = target_transform
        dataset = torchvision.datasets.ImageFolder(root=root_dir,transforms=transforms)
        self.classes = dataset.classes
        self.samples = dataset.samples
        self.class_to_idx = dataset.class_to_idx
        self.targets = dataset.targets
        self.imgs = dataset.imgs  

        if download:
            self.download()


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, target = self.samples[idx]
        target = self.target_transform(target)
        return img,target
         
    def download(self):
        '''
        Add download logic
        '''

    

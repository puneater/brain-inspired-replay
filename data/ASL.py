from torch.utils.data import Dataset, dataset
import torchvision


class ASL(Dataset):
    def __init__(self,root_dir = None,transforms = None):
        dataset = torchvision.datasets.ImageFolder(root=root_dir,transforms=transforms)
        self.classes = dataset.classes
        self.samples = dataset.samples
        self.class_to_idx = dataset.class_to_idx
        self.targets = dataset.targets
        self.imgs = dataset.imgs        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
         

    

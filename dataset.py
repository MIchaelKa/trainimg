import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image

import matplotlib.pyplot as plt

class ImageDataset(Dataset):
    def __init__(self, df, path, transform=None):
        self.df = df
        self.path = path
        self.transform = transform
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_id = row.image_id
        label = row.label
        
        image = Image.open(self.path+image_id)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def show_dataset_grid(dataset):
    nrow, ncol = 3, 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(18, 18))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        image, label = dataset[i]
#         ax.imshow(image)
        ax.imshow(np.array(T.ToPILImage()(image)))
    
        ax.set_title(f'Label: {label}\nShape: {np.array(image).shape}', fontsize=16)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def show_dataset(dataset, count=5):    
    size = 5
    plt.figure(figsize=(count*size,size))
    indices = np.random.choice(np.arange(len(dataset)), count, replace=False)
     
    for i, index in enumerate(indices):    
        image, label = dataset[index]
        plt.subplot(1,count,i+1)
        plt.title(f'Label: {label}\nShape: {np.array(image).shape}', fontsize=16)
#         plt.imshow(image)
        plt.imshow(np.array(T.ToPILImage()(image)))
        plt.grid(False)
        plt.axis('off')
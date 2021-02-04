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

    def get_image_to_show(self, image):
        return np.array(T.ToPILImage()(image))


class ImageDatasetAlbu(Dataset):
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
        
        pillow_image = Image.open(self.path+image_id)
        image = np.array(pillow_image)
        
        if self.transform:
            image = self.transform(image=image)['image']

        image = image.astype(np.float32)
        image /= 255
        image = image.transpose(2, 0, 1)
            
        return image, label

    def get_image_to_show(self, image):
        return image.transpose(1, 2, 0)


def show_dataset_grid(dataset):
    nrow, ncol = 3, 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(18, 18))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        image, label = dataset[i]
        plt.imshow(dataset.get_image_to_show(image))
        ax.set_title(f'Label: {label}\nShape: {np.array(image).shape}', fontsize=16)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def show_dataset(dataset, count=5, random=True):    
    size = 5
    plt.figure(figsize=(count*size,size))

    if random:   
        indices = np.random.choice(np.arange(len(dataset)), count, replace=False)
    else:
        indices = np.arange(count)
     
    for i, index in enumerate(indices):    
        image, label = dataset[index]
        plt.subplot(1,count,i+1)
        plt.title(f'Label: {label}\nShape: {np.array(image).shape}', fontsize=16)
        plt.imshow(dataset.get_image_to_show(image))
        plt.grid(False)
        plt.axis('off')
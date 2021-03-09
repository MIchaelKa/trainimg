import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from torchvision import transforms as T

# Use mean and std for pretrained models
# https://pytorch.org/docs/stable/torchvision/models.html

def get_train_transform(img_size):
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        # T.Normalize(mean=[0.485, 0.456, 0.406],
        #             std=[0.229, 0.224, 0.225])
    ])
    return transform

def get_valid_transform(img_size):
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        # T.Normalize(mean=[0.485, 0.456, 0.406],
        #             std=[0.229, 0.224, 0.225])
    ])
    return transform

class ImageDataset(Dataset):
    def __init__(self, df, path, transform=None):
        print('Init dataset T')
        self.df = df
        self.path = path
        self.transform = transform

        # self.target_columns = df.columns[1:-1]

        self.target_columns = [
            # 'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
            # 'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
            'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
            # 'Swan Ganz Catheter Present'
        ]
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_id = row.StudyInstanceUID
        label = row[self.target_columns].values.astype(float)
        
        image = Image.open(self.path+image_id+'.jpg')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

    def get_image_to_show(self, image):
        return np.array(T.ToPILImage()(image))

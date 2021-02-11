import pandas as pd

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

from sklearn.model_selection import train_test_split

import argparse

from train import *
# from dataset import *
from dataset_a import *
from utils import *
from model import *

def create_datasets(
    path_to_data,
    reduce_train,
    train_number,
    valid_number
    ):
    train_df_all = pd.read_csv(path_to_data + 'train.csv')
    print(f'Dataset size: {train_df_all.shape}')

    # train_df = train_df_all
    train_df = train_df_all.sample(frac=1).reset_index(drop=True)

    train_df, valid_df = train_test_split(train_df, test_size=0.2)
    print(f'Dataset size, train: {len(train_df)}, valid: {len(valid_df)}')

    if reduce_train:
        train_df = train_df.sample(frac=1).reset_index(drop=True).head(train_number)
        valid_df = valid_df.sample(frac=1).reset_index(drop=True).head(valid_number)
        print(f'Reduce dataset size, train: {len(train_df)}, valid: {len(valid_df)}')

    img_path = path_to_data + '/train_images/'

    train_dataset = ImageDataset(train_df, img_path, get_train_transform())
    valid_dataset = ImageDataset(valid_df, img_path, get_valid_transform())

    return train_dataset, valid_dataset

def create_train_dataset(path_to_data):
    train_df_all = pd.read_csv(path_to_data + 'train.csv')
    print(f'Dataset size: {train_df_all.shape}')

    img_path = path_to_data + '/train_images/'

    train_dataset = ImageDataset(train_df_all, img_path, get_train_transform())

    return train_dataset

def create_dataloaders(
    train_dataset,
    valid_dataset,
    batch_size
    ):
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    print(f'DataLoader size, train: {len(train_loader)}, valid: {len(valid_loader)}, batch_size: {batch_size}')
    
    return train_loader, valid_loader

def create_dataloaders_sampler(
    train_dataset,
    batch_size,
    train_number,
    valid_number
    ):
    
    all_number = train_number + valid_number

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(range(train_number)))
    valid_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(range(train_number, all_number)))

    print(f'DataLoader size, train: {len(train_loader)}, valid: {len(valid_loader)}')
    
    return train_loader, valid_loader

def get_device():
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print('No GPU available, using the CPU instead.')

    return device

def run_loader(
    model,
    train_loader,
    valid_loader,
    learning_rate=3e-4,
    weight_decay=1e-3,
    num_epoch=10
    ):

    run_decription = (
        f"learning_rate = {learning_rate}\n"
        f"weight_decay = {weight_decay}\n"
        f"num_epoch = {num_epoch}\n"
    )
    
    print(run_decription)
  
    device = get_device()

    model.to(device)

    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # resnet18
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 5, 7, 8, 9], gamma=0.2)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 5, 8], gamma=0.4)

    # resnet32
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1, 3, 5, 7, 9], gamma=0.4)
    
    # effnet
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [4, 6, 8], gamma=0.4)
    
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch) # V17

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100], gamma=1)

    print('')
    print('Start training...')
    train_info = train_model(model, device, train_loader, valid_loader, loss, optimizer, scheduler, num_epoch)

    return train_info

def run(
    path_to_data='/',
    batch_size=32,
    reduce_train=False,
    train_number=0,
    valid_number=0,
    learning_rate=3e-4,
    weight_decay=1e-3,
    num_epoch=10
    ):
    
    train_dataset, valid_dataset = create_datasets(path_to_data, reduce_train, train_number, valid_number)
    train_loader, valid_loader = create_dataloaders(train_dataset, valid_dataset, batch_size)

    # train_dataset = create_train_dataset(path_to_data)
    # train_loader, valid_loader = create_dataloaders(train_dataset, batch_size, train_number, valid_number)

    # model = ResNetModel()
    # model = EfficientNetModel()
    model = DenseNetModel()

    train_info = run_loader(model, train_loader, valid_loader, learning_rate, weight_decay, num_epoch)

    return train_info, model
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_data", type=str, default='/cassava/')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2020)
    args = parser.parse_args()

    # seed_everything(args.seed)

    train_info, model = run(
        args.path_to_data,
        args.batch_size,
        num_epoch=args.num_epoch
        )

    

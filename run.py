import pandas as pd

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold

import argparse

from train import *
from utils import *

# from dataset import *
from dataset_a import *

from model import *
# from model_effnet import *

def create_datasets(
    path_to_data,
    img_size,
    reduce_train,
    train_number,
    valid_number
    ):
    train_df_all = pd.read_csv(path_to_data + 'train.csv')
    print(f'Dataset size: {train_df_all.shape}, img_size: {img_size}')

    # train_df = train_df_all
    train_df = train_df_all.sample(frac=1).reset_index(drop=True)

    train_df, valid_df = train_test_split(train_df, test_size=0.2)
    print(f'Dataset size, train: {len(train_df)}, valid: {len(valid_df)}')

    if reduce_train:
        train_df = train_df.sample(frac=1).reset_index(drop=True).head(train_number)
        valid_df = valid_df.sample(frac=1).reset_index(drop=True).head(valid_number)
        print(f'Reduce dataset size, train: {len(train_df)}, valid: {len(valid_df)}')

    img_path = path_to_data + '/train_images/'

    train_dataset = ImageDataset(train_df, img_path, get_train_transform(img_size))
    valid_dataset = ImageDataset(valid_df, img_path, get_valid_transform(img_size))

    return train_dataset, valid_dataset, valid_df

def create_train_dataset(path_to_data, img_size):
    train_df_all = pd.read_csv(path_to_data + 'train.csv')
    print(f'Dataset size: {train_df_all.shape}')

    img_path = path_to_data + '/train_images/'

    train_dataset = ImageDataset(train_df_all, img_path, get_train_transform(img_size))

    return train_dataset

def create_dataloaders(
    train_dataset,
    valid_dataset,
    batch_size_train,
    batch_size_valid
    ):
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size_valid, shuffle=False)

    print(f'DataLoader size, train: {len(train_loader)}, valid: {len(valid_loader)}, batch_size_train: {batch_size_train}')
    
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
    num_epoch=10,
    fold=0
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

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2, 4, 6, 8, 9], gamma=0.4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch) # V17  
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100], gamma=1)

    print('')
    print('Start training...')
    train_info = train_model(model, device, train_loader, valid_loader, loss, optimizer, scheduler, num_epoch, fold)

    return train_info

def run(
    path_to_data='/',
    batch_size_train=32,
    batch_size_valid=32,
    reduce_train=False,
    train_number=0,
    valid_number=0,
    img_size=256,
    learning_rate=3e-4,
    weight_decay=1e-3,
    num_epoch=10
    ):
    
    train_dataset, valid_dataset, valid_df = create_datasets(path_to_data, img_size, reduce_train, train_number, valid_number)
    train_loader, valid_loader = create_dataloaders(train_dataset, valid_dataset, batch_size_train, batch_size_valid)

    # train_dataset = create_train_dataset(path_to_data)
    # train_loader, valid_loader = create_dataloaders(train_dataset, batch_size, train_number, valid_number)

    # model = SimpleModel()
    model = ResNetModel(pretrained=True)
    # model = EfficientNetModel()
    # model = DenseNetModel()

    train_info = run_loader(model, train_loader, valid_loader, learning_rate, weight_decay, num_epoch, 0)

    valid_df['probs'] = train_info['probs']
    valid_df['preds'] = train_info['preds']
    valid_df.to_csv('valid_df.csv', index=False)

    return train_info, model

def run_cv(
    path_to_data,
    batch_size_train=32,
    batch_size_valid=32,
    learning_rate=3e-4,
    weight_decay=1e-3,
    num_epoch=10,
    debug=False
    ):

    train_df = pd.read_csv(path_to_data + 'train.csv')
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    # train_df = train_df.sort_values('label').reset_index(drop=True)
    if debug:
        train_df = train_df.sample(frac=1).reset_index(drop=True).head(20)

    print(f'Dataset size: {train_df.shape}')

    # skf = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
    skf = StratifiedKFold(n_splits=5)
    # skf = KFold(n_splits=5)

    X = train_df  
    y = train_df['label'].values # train.label.values

    img_path = path_to_data + '/train_images/'

    train_infos = []
    
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_valid = X.loc[train_index], X.loc[test_index]

        print('')
        print(f'Fold: {fold}')
        print(X_train.shape, X_valid.shape)
        print(X_train['label'].value_counts())
        print(X_valid['label'].value_counts())
        print('')

        # print(train_index.shape, test_index.shape)
        # print(train_index[:5], train_index[-5:], test_index[:5])

        train_dataset = ImageDataset(X_train, img_path, get_train_transform())
        valid_dataset = ImageDataset(X_valid, img_path, get_valid_transform())

        train_loader, valid_loader = create_dataloaders(train_dataset, valid_dataset, batch_size_train, batch_size_valid)

        model = ResNetModel(pretrained=True)
        train_info = run_loader(model, train_loader, valid_loader, learning_rate, weight_decay, num_epoch, fold)
        train_infos.append(train_info)

        X_valid['probs'] = train_info['probs']
        X_valid['preds'] = train_info['preds']
        X_valid.to_csv(f'valid_df_{fold}.csv', index=False)

    return train_infos


def main(path_to_data, debug=False):
    SEED = 2020
    seed_everything(SEED)
    print_version()

    if debug:
        params = {
            'path_to_data'     : path_to_data,
            'batch_size_train' : 2,
            'batch_size_valid' : 2,
            'reduce_train'     : True,
            'train_number'     : 10,
            'valid_number'     : 10,
            'img_size'         : 384,
            'learning_rate'    : 3e-4, # 1e-4
            'weight_decay'     : 0, # 1e-3, 5e-4
            'num_epoch'        : 5
        }
    else:
        params = {
            'path_to_data'     : path_to_data,
            'batch_size_train' : 32,
            'batch_size_valid' : 32,
            'reduce_train'     : False,
            'train_number'     : 12000,
            'valid_number'     : 1000,
            'img_size'         : 384,
            'learning_rate'    : 2e-4,
            'weight_decay'     : 0, # 1e-3, 5e-4
            'num_epoch'        : 10
        }

    return run(**params)

def inference(
    path_to_data,
    model_path,
    model_name,
    batch_size,
    img_size
    ):

    t0 = time.time()

    test_df = pd.read_csv(path_to_data + 'sample_submission.csv')
    print(f'Dataset size: {test_df.shape}, img_size: {img_size}')

    img_path = path_to_data + '/test_images/'
    test_dataset = TestImageDataset(test_df, img_path, get_valid_transform(img_size))

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = get_device()

    model = ResNetModel(pretrained=False)
    model.load_state_dict(torch.load(model_path + model_name))
    model.eval()
    model.to(device)

    predictions = []

    with torch.no_grad():
        for _, (x_batch) in enumerate(test_loader):
            x_batch = x_batch.to(device)        
            output = model(x_batch)
            indices = torch.argmax(output, 1)
            predictions.append(indices)

    predictions = torch.cat(predictions).cpu().numpy()

    test_df['label'] = predictions
    test_df[['image_id', 'label']].to_csv('./submission.csv', index=False)

    print('Inference finished for: {}'.format(format_time(time.time() - t0)))


def inference_k_fold():
    return 0

def main_inference(
    path_to_data,
    model_path,
    model_name,
    ):

    SEED = 2020
    seed_everything(SEED)
    print_version()

    params = {
        'path_to_data' : path_to_data,
        'model_path'   : model_path,
        'model_name'   : model_name,
        'batch_size'   : 32,
        'img_size'     : 384
    }
    inference(**params)


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

    

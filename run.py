import pandas as pd

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold

import argparse

from train import *
from utils import *

# from dataset import *
from dataset_a import *

from model import *
# from model_timm import *
# from model_effnet import *

from config import GlobalConfig

def create_datasets(
    path_to_data,
    path_to_img,
    path_to_train,
    img_size,
    reduce_train,
    train_number,
    valid_number
    ):
    if path_to_train:
        train_df_all = pd.read_csv(path_to_train)
    else:
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

    if path_to_img is None:
        path_to_img = path_to_data + 'train/'

    train_dataset = ImageDataset(train_df, path_to_img, get_train_transform(img_size))
    valid_dataset = ImageDataset(valid_df, path_to_img, get_valid_transform(img_size))

    return train_dataset, valid_dataset, valid_df

def create_train_dataset(
    path_to_data,
    img_size,
    reduce_train=False,
    train_number=200
    ):
    train_df = pd.read_csv(path_to_data + 'train.csv')
    print(f'Dataset size: {train_df.shape}')

    if reduce_train:
        train_df = train_df.sample(frac=1).reset_index(drop=True).head(train_number)
        print(f'Reduce dataset size: {len(train_df)}')

    img_path = path_to_data + 'train/'

    train_dataset = ImageDataset(train_df, img_path, get_train_transform(img_size))

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

def get_model(model_name, pretrained=True):

    # model = CustomResNet(model_name, pretrained)

    # model = SimpleModel()
    model = ResNetModel(model_name, pretrained)
    # model = EfficientNetModel(model_name, pretrained)
    # model = DenseNetModel()
    return model

# TODO: move to model_*.py ???
def get_model_name():
    # model_name = 'efficientnet-b2'
    # model_name = 'resnet200d' # 'resnet50d', 'resnet101d', 'resnet200d'
    model_name = 'resnet18' #'resnet18', 'resnext50d_32x4d'
    return model_name

def get_scheduler(optimizer, num_epoch):
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2, 4, 6, 8, 9], gamma=0.4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=1e-6, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100], gamma=1)
    return scheduler

def test_scheduler(learning_rate=3e-4, num_epoch=10):
    lr_history = []
    params = (torch.tensor([1,2,3]) for t in range(2))
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    scheduler = get_scheduler(optimizer, num_epoch)
    for epoch in range(num_epoch):
        optimizer.step()
        lr_history.append(scheduler.get_last_lr())  
        scheduler.step()

    return lr_history

def run_loader(
    model,
    device,
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
  
    model.to(device)

    # loss = nn.CrossEntropyLoss()
    # pos_weight = torch.tensor([1,1,1,1,1,1,1,1.1,1.1,0.9,1]).to(device)
    # loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = get_scheduler(optimizer, num_epoch)

    print('')
    print('Start training...')
    train_info = train_model(model, device, train_loader, valid_loader, loss, optimizer, scheduler, num_epoch, fold)

    return train_info

def run(
    path_to_data='/',
    path_to_img=None,
    path_to_train=None,
    model_name='',
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
    
    train_dataset, valid_dataset, valid_df = create_datasets(path_to_data, path_to_img, path_to_train, img_size, reduce_train, train_number, valid_number)
    train_loader, valid_loader = create_dataloaders(train_dataset, valid_dataset, batch_size_train, batch_size_valid)

    # train_dataset = create_train_dataset(path_to_data)
    # train_loader, valid_loader = create_dataloaders(train_dataset, batch_size, train_number, valid_number)

    device = get_device()

    model = get_model(model_name, pretrained=True)

    train_info = run_loader(model, device, train_loader, valid_loader, learning_rate, weight_decay, num_epoch, 0)

    return train_info, model

def run_cv(
    path_to_data,
    path_to_img=None,
    path_to_train=None,
    batch_size_train=32,
    batch_size_valid=32,
    img_size=256,
    learning_rate=3e-4,
    weight_decay=1e-3,
    num_epoch=10,
    debug=False
    ):

    t0 = time.time()

    if path_to_train:
        train_df = pd.read_csv(path_to_train)
    else:
        train_df = pd.read_csv(path_to_data + 'train.csv')

    train_df = train_df.sample(frac=1).reset_index(drop=True)
    # train_df = train_df.sort_values('label').reset_index(drop=True)
    if debug:
        train_df = train_df.sample(frac=1).reset_index(drop=True).head(20)

    print(f'Dataset size: {train_df.shape}')

    # cv = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
    # cv = StratifiedKFold(n_splits=5)
    # cv = KFold(n_splits=5)
    cv = GroupKFold(n_splits=5)

    X = train_df  
    y = train_df[GlobalConfig.target_columns].values # train.label.values
    groups = train_df['PatientID'].values

    if path_to_img is None:
        path_to_img = path_to_data + '/train_images/'

    train_infos = []

    device = get_device()
    
    for fold, (train_index, test_index) in enumerate(cv.split(X, y, groups)):
        X_train, X_valid = X.loc[train_index], X.loc[test_index]

        print('')
        print(f'Fold: {fold}')
        print(X_train.shape, X_valid.shape)
        # print(X_train['PatientID'].value_counts())
        # print(X_valid['PatientID'].value_counts())
        print('')

        # print(train_index.shape, test_index.shape)
        # print(train_index[:5], train_index[-5:], test_index[:5])

        train_dataset = ImageDataset(X_train, path_to_img, get_train_transform(img_size))
        valid_dataset = ImageDataset(X_valid, path_to_img, get_valid_transform(img_size))

        train_loader, valid_loader = create_dataloaders(train_dataset, valid_dataset, batch_size_train, batch_size_valid)

        model = get_model(get_model_name(), pretrained=True)

        train_info = run_loader(model, device, train_loader, valid_loader, learning_rate, weight_decay, num_epoch, fold)
        train_infos.append(train_info)


    cv_scores = np.array([np.mean(i['best_scores']) for i in train_infos])
    print('')
    print(f'CV results: {cv_scores}')
    print(f'CV mean: {cv_scores.mean()}, std: {cv_scores.std()}')
    print('CV finished for: {}'.format(format_time(time.time() - t0)))

    return train_infos

def main(
    path_to_data,
    path_to_img=None,
    path_to_train=None,
    debug=False
    ):
    SEED = 2020
    seed_everything(SEED)
    print_version()

    model_name = get_model_name()

    if debug:
        params = {
            'path_to_data'     : path_to_data,
            'path_to_img'      : path_to_img,
            'path_to_train'    : path_to_train,
            'model_name'       : model_name,
            'batch_size_train' : 2,
            'batch_size_valid' : 2,
            'reduce_train'     : True,
            'train_number'     : 10,
            'valid_number'     : 10,
            'img_size'         : 256,
            'learning_rate'    : 3e-4, # 1e-4
            'weight_decay'     : 0, # 1e-3, 5e-4
            'num_epoch'        : 5
        }
    else:
        params = {
            'path_to_data'     : path_to_data,
            'path_to_img'      : path_to_img,
            'path_to_train'    : path_to_train,
            'model_name'       : model_name,
            'batch_size_train' : 32,
            'batch_size_valid' : 32,
            'reduce_train'     : False,
            'train_number'     : 12000,
            'valid_number'     : 1000,
            'img_size'         : 256,
            'learning_rate'    : 2e-4,
            'weight_decay'     : 0, # 1e-3, 5e-4
            'num_epoch'        : 3
        }

    return run(**params)

def main_cv(
    path_to_data,
    path_to_img=None,
    path_to_train=None,
    debug=False
    ):
    SEED = 2020
    seed_everything(SEED)
    print_version()

    if debug:
        params = {
            'path_to_data'     : path_to_data,
            'path_to_img'      : path_to_img,
            'path_to_train'    : path_to_train,
            'batch_size_train' : 2,
            'batch_size_valid' : 2,
            'img_size'         : 32,
            'learning_rate'    : 2e-4,
            'weight_decay'     : 0, # 1e-3, 5e-4
            'num_epoch'        : 5,
            'debug'            : debug
        }
    else:
        params = {
            'path_to_data'     : path_to_data,
            'path_to_img'      : path_to_img,
            'path_to_train'    : path_to_train,
            'batch_size_train' : 32,
            'batch_size_valid' : 32,
            'img_size'         : 256,
            'learning_rate'    : 2e-4,
            'weight_decay'     : 0, # 1e-3, 5e-4
            'num_epoch'        : 6,
            'debug'            : debug
        }

    return run_cv(**params)

#
# inference
#

def inference(
    test_dataset,
    model_path,
    model,
    model_state,
    batch_size,
    debug,
    ):

    t0 = time.time()
    print(f'Start inference')

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = get_device()

    model.load_state_dict(torch.load(model_path + model_state))
    model.eval()
    model.to(device)

    predictions = []

    with torch.no_grad():
        for _, (x_batch) in enumerate(test_loader):
            x_batch = x_batch.to(device)        
            output = model(x_batch)
            probs = torch.softmax(output, 1).cpu().numpy()
            predictions.append(probs)

    predictions = np.concatenate(predictions)

    print('Inference finished for: {}'.format(format_time(time.time() - t0)))
    return predictions


def inference_kfold(
    test_dataset,
    model_path,
    model,
    model_states,
    batch_size,
    debug,
    ):

    t0 = time.time()
    print(f'Start inference {len(model_states)}-fold')

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = get_device()

    model.to(device)

    predictions = []

    with torch.no_grad():
        for _, (x_batch) in enumerate(test_loader):
            x_batch = x_batch.to(device)

            ave_probs = []
            
            # TODO: move out of for, compare time!
            for model_state in model_states:   
                model.load_state_dict(torch.load(model_path + model_state))
                model.eval()
            
                output = model(x_batch)
                probs = torch.softmax(output, 1).cpu().numpy()
                ave_probs.append(probs)
        
            ave_probs = np.mean(ave_probs, axis=0)
            predictions.append(ave_probs)

    predictions = np.concatenate(predictions)

    print('Inference finished for: {}'.format(format_time(time.time() - t0)))
    return predictions


def main_inference(
    path_to_data,
    model_path,
    model_path_2,
    debug,
    ):

    SEED = 2020
    seed_everything(SEED)
    print_version()

    if debug:
        img_path = path_to_data + '/train_images/'
        img_size = 32

        test_df = pd.read_csv(path_to_data + 'train.csv')
        test_df = test_df.sample(frac=1).reset_index(drop=True).head(20)
    else:
        img_path = path_to_data + '/test_images/'
        img_size = 384

        test_df = pd.read_csv(path_to_data + 'sample_submission.csv')
    
    print(f'Dataset size: {test_df.shape}, img_size: {img_size}')
    test_dataset = TestImageDataset(test_df, img_path, get_valid_transform(img_size))

    #
    # ResNetModel
    #

    model = ResNetModel(pretrained=False)
    model_states = ['model_0.pth', 'model_1.pth', 'model_2.pth', 'model_3.pth', 'model_4.pth']

    if debug:
        params = {
            'test_dataset' : test_dataset,
            'model_path'   : model_path,
            'model'        : model,
            'model_states' : model_states,
            'batch_size'   : 4,
            'debug'        : debug,
        }
    else:
        params = {
            'test_dataset' : test_dataset,
            'model_path'   : model_path,
            'model'        : model,
            'model_states' : model_states,
            'batch_size'   : 32,
            'debug'        : debug,
        }

    preds_1 = inference_kfold(**params)

    #
    # EfficientNetModel
    #

    model = EfficientNetModel('efficientnet-b2', pretrained=False)
    model_state = 'model_0.pth'

    if debug:
        params = {
            'test_dataset' : test_dataset,
            'model_path'   : model_path_2,
            'model'        : model,
            'model_state'  : model_state,
            'batch_size'   : 4,
            'debug'        : debug,
        }
    else:
        params = {
            'test_dataset' : test_dataset,
            'model_path'   : model_path_2,
            'model'        : model,
            'model_state'  : model_state,
            'batch_size'   : 32,
            'debug'        : debug,
        }
   
    preds_2 = inference(**params)

    predictions = 0.6 * preds_1 + 0.4 * preds_2
    predictions = predictions.argmax(1)
        
    test_df['label'] = predictions
    test_df[['image_id', 'label']].to_csv('./submission.csv', index=False)

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

    

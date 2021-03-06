import pandas as pd

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold

import argparse

from config import GlobalConfig

if GlobalConfig.val_every:
    print('import val_every')
    from train_val_every import *
else:
    print('import train')
    from train import *

from utils import *
from loss import *

# from dataset import *
from dataset_a import *

# from model import *
# from model_timm import *
# from model_effnet import *

#
# create
#

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
    valid_dataset,
    batch_size,
    train_number,
    valid_number,
    num_workers,
    pin_memory
    ):
    
    all_number = train_number + valid_number

    train_sampler = SubsetRandomSampler(range(train_number))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    valid_sampler = SubsetRandomSampler(range(train_number, all_number))
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # TODO: verbose support
    # print(f'data_loader_size train: {len(train_loader)}, valid: {len(valid_loader)}')
    
    return train_loader, valid_loader

#
# get
#

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
    model_name = 'resnext50_32x4d' #'resnet18', 'resnext50_32x4d'
    return model_name

def get_scheduler(optimizer, scheduler_params):
    name = scheduler_params['name']
    num_iter = scheduler_params['num_iter']

    if name == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=iter_number,
            eta_min=1e-6,
            last_epoch=-1
        )
    elif name == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=766*2,
            T_mult=2,
            eta_min=1e-6,
            last_epoch=-1
        )
    elif name == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scheduler_params['max_lr'],
            total_steps=num_iter,
            anneal_strategy='linear',
            pct_start=scheduler_params['pct_start']
        )
    elif name == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [25, 40], gamma=0.4)
    elif name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    elif name == 'LRRangeTest':
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.1)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda i: 2*i)
        # scheduler = LinearLR(optimizer, 5.0, iter_number)
        scheduler = ExponentialLR(optimizer, 5.0, iter_number)
    elif name == 'None':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100], gamma=1)

    return scheduler

def test_scheduler(scheduler_params):
    learning_rate = scheduler_params['learning_rate']
    num_iter = scheduler_params['num_iter']

    lr_history = []

    params = (torch.tensor([1,2,3]) for t in range(2))
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    scheduler = get_scheduler(optimizer, scheduler_params)

    for epoch in range(num_iter):
        optimizer.step()
        lr_history.append(scheduler.get_last_lr())
        scheduler.step()

    return lr_history

def get_optimizer(name, parameters, lr, weight_decay): 
    if name == 'Adam':
        eps = 1e-4 if GlobalConfig.half_precision else 1e-08
        optimizer = torch.optim.Adam(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps
        )
    elif name == 'SGD':
        optimizer = torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True
        )
    else:
        print("Error: unsupported optimizer")
    
    return optimizer

#
# run
#

def run_dataset(
    model,
    device,
    train_dataset,
    valid_dataset,
    batch_size,
    optimizer_name,
    learning_rate=3e-4,
    weight_decay=1e-3,
    scheduler_params=None,
    num_epoch=10,
    fold=0,
    verbose=True,
    ):

    train_number = 49000
    valid_number = 1000
    num_workers = 2
    pin_memory = True

    if verbose:
        run_decription = (
            f"batch_size = {batch_size}"
        )
        print(run_decription)

    train_loader, valid_loader = create_dataloaders_sampler(
        train_dataset, valid_dataset,
        batch_size, train_number, valid_number,
        num_workers, pin_memory)

    params = {
        'optimizer_name'   : optimizer_name,
        'learning_rate'    : learning_rate,
        'weight_decay'     : weight_decay,
        'scheduler_params' : scheduler_params,
        'num_epoch'        : num_epoch,
        'verbose'          : verbose
    }

    return run_loader(model, device, train_loader, valid_loader, **params)
    
def run_loader(
    model,
    device,
    train_loader,
    valid_loader,
    optimizer_name,
    learning_rate=3e-4,
    weight_decay=1e-3,
    scheduler_params=None,
    num_epoch=10,
    fold=0,
    verbose=True,
    ):

    scheduler_batch_update = scheduler_params['batch_update']
    if scheduler_batch_update:
        num_iter = len(train_loader) * num_epoch
    else:
        num_iter = num_epoch
    scheduler_params['num_iter'] = num_iter

    if verbose:
        run_decription = (
            f"optimizer_name = {optimizer_name}\n"
            f"learning_rate = {learning_rate}\n"
            f"weight_decay = {weight_decay}\n"
            f"scheduler_params = {scheduler_params}\n"
            f"num_epoch = {num_epoch}\n"
        )
        print(run_decription)

    model.to(device)

    loss = LabelSmoothingV2()
    # loss = nn.CrossEntropyLoss()
    # pos_weight = torch.tensor([1,1,1,1,1,1,1,1.1,1.1,0.9,1]).to(device)
    # loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # loss = nn.BCEWithLogitsLoss()

    optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate, weight_decay)

    scheduler = get_scheduler(optimizer, scheduler_params)

    if verbose:
        print('Start training...')
    
    train_info = train_model(model, device, train_loader, valid_loader, loss, optimizer, scheduler, scheduler_batch_update, num_epoch, fold, verbose)

    return train_info

def run_iter(
    model,
    device,
    train_loader,
    valid_loader,
    optimizer_name,
    learning_rate=3e-4,
    weight_decay=1e-3,
    num_iter=300,
    verbose=True,
    ):

    if verbose:
        run_decription = (
            f"optimizer_name = {optimizer_name}\n"
            f"learning_rate = {learning_rate}\n"
            f"weight_decay = {weight_decay}\n"
            f"num_iter = {num_iter}\n"
        )
        print(run_decription)

    model.to(device)

    loss = nn.CrossEntropyLoss()

    optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate, weight_decay)

    if verbose:
        print('')
        print('Start training...')

    t0 = time.time()
    
    t_loss_history, t_score_history, v_loss_history, v_score_history = train_iter(model, device, train_loader, valid_loader, loss, optimizer, num_iter, verbose)

    # do not save best, better save last?
    best_index = np.argmax(v_score_history)
    valid_best_score = v_score_history[best_index]
    valid_best_loss = v_loss_history[best_index]

    if verbose:
        print('')
        print('[valid] best: {:>3d}, loss = {:.5f}, score = {:.5f}'.format(best_index, valid_best_loss, valid_best_score))
        print('training finished for: {}'.format(format_time(time.time() - t0)))

    train_info = {
        'train_loss_history' : t_loss_history,
        'train_score_history' : t_score_history,
        'valid_loss_history' : v_loss_history,
        'valid_score_history' : v_score_history,
        'best_score' : valid_best_score,
    }

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

    params = {
        'path_to_data'     : path_to_data,
        'path_to_img'      : path_to_img,
        'path_to_train'    : path_to_train,
        'model_name'       : model_name,
        'batch_size_train' : 16,
        'batch_size_valid' : 16,
        'reduce_train'     : False,
        'train_number'     : 12000,
        'valid_number'     : 1000,
        'img_size'         : 512,
        'learning_rate'    : 2e-4,
        'weight_decay'     : 0, # 1e-5, 1e-3, 5e-4
        'num_epoch'        : 6
    }

    if debug:
        params['batch_size_train'] = 2 # better test real
        params['batch_size_valid'] = 2
        params['reduce_train'] = True
        params['train_number'] = 10
        params['valid_number'] = 10
        params['num_epoch'] = 6

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

    params = {
        'path_to_data'     : path_to_data,
        'path_to_img'      : path_to_img,
        'path_to_train'    : path_to_train,
        'batch_size_train' : 32,
        'batch_size_valid' : 32,
        'img_size'         : 256,
        'learning_rate'    : 2e-4,
        'weight_decay'     : 1e-5, # 1e-3, 5e-4
        'num_epoch'        : 6,
        'debug'            : debug
    }

    if debug:
        params['batch_size_train'] = 2
        params['batch_size_valid'] = 2
        params['img_size'] = 32
        params['num_epoch'] = 5

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

    

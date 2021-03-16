import numpy as np
import torch

from utils import *
from config import GlobalConfig
from metrics import AccuracyMeter

def train_batch():
    return

def train_val_model():
    return

def validate(model, device, val_loader, criterion):

    model.eval()
    
    loss_history = []
    meter = AccuracyMeter()

    # TODO: try to use only with model
    with torch.no_grad():
        for _, (x_batch, y_batch) in enumerate(val_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            output = model(x_batch)
            loss = criterion(output, y_batch)
            
            # Save loss value
            loss_item = loss.item()
            loss_history.append(loss_item)
            
            # Compute score
            meter.update(y_batch, output)

    valid_score = meter.compute_score()
    valid_scores = meter.history()
    valid_cms = meter.compute_cm()

    # print('[valid] -------------------------- score = {:.5f}'.format(valid_score_total))
    
    return loss_history, valid_score, valid_scores, valid_cms

def train_epoch(model, device, train_loader, criterion, optimizer, scheduler):   
    
    model.train()

    train_loss = []
    lr_history = []
    loss_accum = 0
    meter = AccuracyMeter()

    print_every = int(len(train_loader) / 5)
    print_every = 1 if print_every == 0 else print_every

    for index, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        output = model(x_batch)
        loss = criterion(output, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Save loss value
        loss_item = loss.item()
        loss_accum += loss_item
        train_loss.append(loss_item)
        
        # Save predictions
        meter.update(y_batch, output)

        # Scheduler update
        if GlobalConfig.scheduler_batch_update:
            lr_history.append(scheduler.get_last_lr())  
            scheduler.step()

        # running_loss = loss_accum / (index + 1)
        # running_score = np.mean(meter.compute_score())
        # train_scores.append(running_score)

        # if index % print_every == 0:
        #     print('[train] _iter: {:>2d}, loss = {:.5f}, score = {:.5f}'.format(index, running_loss, running_score))

    # ave_loss = loss_accum / (index + 1)
    train_score = meter.compute_score()
    train_scores = meter.history()

    # print('[train] _iter: {:>2d}, loss = {:.5f}, score = {:.5f}'.format(index, ave_loss, ave_score))
    
    # if index % val_every == 0:
    #     validate(model, loader_val)

    return train_loss, train_score, train_scores, lr_history

def train_model(model, device, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, fold, verbose):
   
    train_loss_history = []
    train_score_history = []
    
    train_loss_epochs = []
    train_score_epochs = []
    
    valid_loss_history = []
    valid_score_history = []
    
    valid_loss_epochs = []
    valid_score_epochs = []

    valid_best_score = 0
    best_epoch = 0
    best_cms = []

    lr_history = []
    
    t0 = time.time()
    
    for epoch in range(num_epochs):

        # Train
        t1 = time.time()
        train_loss, train_score, train_scores, lr_history_epoch = train_epoch(model, device, train_loader, criterion, optimizer, scheduler)

        train_loss_history.extend(train_loss)
        train_score_history.extend(train_scores)
        
        # TODO: AverageMeter()
        train_loss_mean = np.mean(train_loss)
        
        train_loss_epochs.append(train_loss_mean)
        train_score_epochs.append(train_score)
        
        if verbose:
            print('[train] epoch: {:>2d}, loss = {:.5f}, score = {:.5f}, time: {}' \
                .format(epoch+1, train_loss_mean, train_score, format_time(time.time() - t1)))

        # Validate
        t2 = time.time()     
        valid_loss, valid_score, valid_scores, valid_cms = validate(model, device, val_loader, criterion)
        
        valid_loss_history.extend(valid_loss)  
        valid_score_history.extend(valid_scores)
        
        # TODO: AverageMeter()
        valid_loss_mean = np.mean(valid_loss)
        
        valid_loss_epochs.append(valid_loss_mean)
        valid_score_epochs.append(valid_score)

        if valid_score > valid_best_score:
            valid_best_score = valid_score
            best_epoch = epoch
            best_cms = valid_cms

            #save model
            # torch.save(model.state_dict(), f'model_{fold}.pth')

        if GlobalConfig.scheduler_batch_update:
            lr_history.extend(lr_history_epoch)
        else:
            lr_history.append(scheduler.get_last_lr())
            scheduler.step()

        if verbose:   
            print('[valid] epoch: {:>2d}, loss = {:.5f}, score = {:.5f}, time: {}' \
                .format(epoch+1, valid_loss_mean, valid_score, format_time(time.time() - t1)))
            print('')

    if verbose:
        print('[valid] best epoch {:>2d}, score = {:.5f}'.format(best_epoch+1, valid_best_score))
        print('training finished for: {}'.format(format_time(time.time() - t0)))

    train_info = {
        'train_loss_history' : train_loss_history,
        'train_score_history' : train_score_history,
        'train_loss_epochs' : train_loss_epochs,
        'train_score_epochs' : train_score_epochs,
        'valid_loss_history' : valid_loss_history,
        'valid_score_history' : valid_score_history,
        'valid_loss_epochs' : valid_loss_epochs,
        'valid_score_epochs' : valid_score_epochs,
        'lr_history' : lr_history,
        'best_score' : valid_best_score,
        'best_cms' : best_cms,
    }
 
    return train_info
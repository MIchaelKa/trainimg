import numpy as np
import torch

from utils import *
from config import GlobalConfig
from metrics import AccuracyMeter, AverageMeter

def train_batch():
    return

def train_val_model():
    return

def validate(model, device, val_loader, criterion):

    model.eval()
    
    loss_meter = AverageMeter()
    score_meter = AccuracyMeter()

    # TODO: try to use only with model
    with torch.no_grad():
        for _, (x_batch, y_batch) in enumerate(val_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            output = model(x_batch)
            loss = criterion(output, y_batch)
            
            # Update meters
            loss_meter.update(loss.item())
            score_meter.update(y_batch, output)
   
    return loss_meter, score_meter

def train_epoch(model, device, train_loader, criterion, optimizer, scheduler):
    
    model.train()
    
    loss_meter = AverageMeter()
    score_meter = AccuracyMeter()
    lr_history = []

    for index, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        output = model(x_batch)
        loss = criterion(output, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update meters
        loss_meter.update(loss.item())
        score_meter.update(y_batch, output)

        # Scheduler update
        if GlobalConfig.scheduler_batch_update:
            lr_history.append(scheduler.get_last_lr())  
            scheduler.step()
    
    return loss_meter, score_meter, lr_history

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
        t_loss_meter, t_score_meter, lr_history_epoch = train_epoch(model, device, train_loader, criterion, optimizer, scheduler)

        train_loss_history.extend(t_loss_meter.history)
        train_score_history.extend(t_score_meter.history)
        
        train_loss = t_loss_meter.compute_average()
        train_score = t_score_meter.compute_score()
        
        train_loss_epochs.append(train_loss)
        train_score_epochs.append(train_score)

        if verbose:
            print('[train] epoch: {:>2d}, loss = {:.5f}, score = {:.5f}, time: {}' \
                .format(epoch+1, train_loss, train_score, format_time(time.time() - t1)))

        # Validate
        t2 = time.time()     
        v_loss_meter, v_score_meter = validate(model, device, val_loader, criterion)

        valid_loss_history.extend(v_loss_meter.history)
        valid_score_history.extend(v_score_meter.history)
        
        valid_loss = v_loss_meter.compute_average()
        valid_score = v_score_meter.compute_score()
        
        valid_loss_epochs.append(valid_loss)
        valid_score_epochs.append(valid_score)

        if valid_score > valid_best_score:
            valid_best_score = valid_score
            best_epoch = epoch
            best_cms = v_score_meter.compute_cm()

            #save model
            # torch.save(model.state_dict(), f'model_{fold}.pth')
        
        if GlobalConfig.scheduler_batch_update:
            lr_history.extend(lr_history_epoch)
        else:
            lr_history.append(scheduler.get_last_lr())
            scheduler.step()

        if verbose:   
            print('[valid] epoch: {:>2d}, loss = {:.5f}, score = {:.5f}, time: {}' \
                .format(epoch+1, valid_loss, valid_score, format_time(time.time() - t1)))
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
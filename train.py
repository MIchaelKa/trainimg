import numpy as np
import torch

from sklearn.metrics import confusion_matrix

from utils import *
from metrics import RocAucMeter

def train_batch():
    return

def train_val_model():
    return

def validate(model, device, val_loader, criterion):

    model.eval()
    
    loss_history = []
    meter = RocAucMeter(11)

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

    valid_score_total = np.mean(meter.compute_score())
    # print('[valid] -------------------------- score = {:.5f}'.format(valid_score_total))
    
    return loss_history, valid_score_total

def train_epoch(model, device, train_loader, criterion, optimizer):   
    
    model.train()

    train_loss = []
    train_scores = []
  
    loss_accum = 0

    meter = RocAucMeter(11)

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
      
        running_loss = loss_accum / (index + 1)
        running_score = np.mean(meter.compute_score())
        train_scores.append(running_score)

        if index % print_every == 0:
            print('[train] _iter: {:>2d}, loss = {:.5f}, score = {:.5f}'.format(index, running_loss, running_score))

    ave_loss = loss_accum / (index + 1)
    ave_score = np.mean(meter.compute_score())

    print('[train] _iter: {:>2d}, loss = {:.5f}, score = {:.5f}'.format(index, ave_loss, ave_score))
    
    # if index % val_every == 0:
    #     validate(model, loader_val)

    return train_loss, train_scores, ave_score

def train_model(model, device, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, fold):
   
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

    lr_history = []
    
    t0 = time.time()
    
    for epoch in range(num_epochs):

        # Train
        t1 = time.time()
        train_loss, train_scores, ave_score = train_epoch(model, device, train_loader, criterion, optimizer)

        train_loss_history.extend(train_loss)
        train_score_history.extend(train_scores)
        
        train_loss_mean = np.mean(train_loss)
        
        train_loss_epochs.append(train_loss_mean)
        train_score_epochs.append(ave_score)
        
        print('[train] epoch: {:>2d}, loss = {:.5f}, score = {:.5f}, time: {}' \
              .format(epoch+1, train_loss_mean, ave_score, format_time(time.time() - t1)))

        # Validate
        t2 = time.time()     
        valid_loss, valid_score = validate(model, device, val_loader, criterion)
        
        valid_loss_history.extend(valid_loss)
        
        valid_loss_mean = np.mean(valid_loss)
        
        valid_loss_epochs.append(valid_loss_mean)
        valid_score_epochs.append(valid_score)

        if valid_score > valid_best_score:
            valid_best_score = valid_score
            best_epoch = epoch

            #save model
            # torch.save(model.state_dict(), f'model_{fold}.pth')

        lr_history.append(scheduler.get_last_lr())  
        scheduler.step()
             
        print('[valid] epoch: {:>2d}, loss = {:.5f}, score = {:.5f}, time: {}' \
              .format(epoch+1, valid_loss_mean, valid_score, format_time(time.time() - t1)))
        
        # Epoch
        # print('------- epoch: {:>2d}, time: {}'.format(epoch+1, format_time(time.time() - t1)))
        # print("---------------------------------------------------------")
        print('')

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
    }
 
    return train_info
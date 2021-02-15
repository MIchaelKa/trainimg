import numpy as np
import torch

import time
import datetime

from sklearn.metrics import confusion_matrix

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def train_batch():
    return

def train_val_model():
    return

def validate(model, device, val_loader, criterion):

    model.eval()
    
    loss_history = []
    acc_history = [] # do we really need acc by batches?

    total_correct_samples = 0
    total_samples = 0

    predictions = []
    ground_truth = []
    probs = []

    with torch.no_grad():
        for _, (x_batch, y_batch) in enumerate(val_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            output = model(x_batch)
            loss = criterion(output, y_batch)
            
            # Save loss value
            loss_item = loss.item()     
            loss_history.append(loss_item)
            
            # Compute accuracy
            indices = torch.argmax(output, 1)
            correct_samples = torch.sum(indices == y_batch)
            accuracy = float(correct_samples) / y_batch.shape[0]
            acc_history.append(accuracy)
                      
            total_correct_samples += correct_samples
            total_samples += y_batch.shape[0]

            # Save data for calculating of confusion matrix
            predictions.append(indices)
            ground_truth.append(y_batch)

            # Save probs
            max_probs, _ = torch.max(torch.softmax(output, 1), 1)
            probs.append(max_probs)

    predictions = torch.cat(predictions).cpu().numpy()
    ground_truth = torch.cat(ground_truth).cpu().numpy()

    cm = confusion_matrix(ground_truth, predictions)

    probs = torch.cat(probs).cpu().numpy()

    valid_acc_total = float(total_correct_samples) / total_samples   
    # print('[valid] -------------------------- accuracy = {:.5f}'.format(valid_acc_total))
    
    return loss_history, acc_history, cm, probs, predictions

def train_epoch(model, device, train_loader, criterion, optimizer):   
    
    model.train()

    train_loss = []
    train_acc = []
  
    loss_accum = 0
    total_correct_samples = 0
    total_samples = 0

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
        
        # Compute train accuracy
        indices = torch.argmax(output, 1)
        correct_samples = torch.sum(indices == y_batch)
        accuracy = float(correct_samples) / y_batch.shape[0] # VS / one time on total_count
        train_acc.append(accuracy)
        
        total_correct_samples += correct_samples
        total_samples += y_batch.shape[0]
        
        running_loss = loss_accum / (index + 1)
        running_acc = np.mean(train_acc)

        # if index % print_every == 0:
        #     print('[train] _iter: {:>2d}, loss = {:.5f}, accuracy = {:.5f}'.format(index, running_loss, running_acc))

    ave_loss = loss_accum / (index + 1)
    ave_acc = np.mean(train_acc)
    ave_acc_2 = float(total_correct_samples) / total_samples

    # print('[train] _iter: {:>2d}, loss = {:.5f}, accuracy = {:.5f}'.format(index, ave_loss, ave_acc_2))
    
    # if index % val_every == 0:
    #     validate(model, loader_val)

    return train_loss, train_acc

def train_model(model, device, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, fold):
   
    train_loss_history = []
    train_acc_history = []
    
    train_loss_epochs = []
    train_acc_epochs = []
    
    valid_loss_history = []
    valid_acc_history = []
    
    valid_loss_epochs = []
    valid_acc_epochs = []

    valid_best_acc = 0
    best_epoch = 0
    best_cm = []
    best_probs = []
    best_preds = []

    lr_history = []
    
    t0 = time.time()
    
    for epoch in range(num_epochs):
  
        # Train
        t1 = time.time()     
        train_loss, train_acc = train_epoch(model, device, train_loader, criterion, optimizer)
        
        train_loss_history.extend(train_loss)
        train_acc_history.extend(train_acc)
        
        train_loss_mean = np.mean(train_loss)
        train_acc_mean = np.mean(train_acc)
        
        train_loss_epochs.append(train_loss_mean)
        train_acc_epochs.append(train_acc_mean)
        
        print('[train] epoch: {:>2d}, loss = {:.5f}, accuracy = {:.5f}, time: {}' \
              .format(epoch+1, train_loss_mean, train_acc_mean, format_time(time.time() - t1)))
  
        # Validate
        t2 = time.time()     
        valid_loss, valid_acc, cm, probs, preds = validate(model, device, val_loader, criterion)
        
        valid_loss_history.extend(valid_loss)
        valid_acc_history.extend(valid_acc)
        
        valid_loss_mean = np.mean(valid_loss)
        valid_acc_mean = np.mean(valid_acc)
        
        valid_loss_epochs.append(valid_loss_mean)
        valid_acc_epochs.append(valid_acc_mean)

        if valid_acc_mean > valid_best_acc:
            valid_best_acc = valid_acc_mean
            best_epoch = epoch
            best_cm = cm
            best_probs = probs
            best_preds = preds

            #save model
            torch.save(model.state_dict(), f'model_{fold}.pth')

        lr_history.append(scheduler.get_last_lr())  
        scheduler.step()
             
        print('[valid] epoch: {:>2d}, loss = {:.5f}, accuracy = {:.5f}, time: {}' \
              .format(epoch+1, valid_loss_mean, valid_acc_mean, format_time(time.time() - t1)))
        
        # Epoch
        # print('------- epoch: {:>2d}, time: {}'.format(epoch+1, format_time(time.time() - t1)))
        # print("---------------------------------------------------------")
        print('')

    print('[valid] best epoch {:>2d}, accuracy = {:.5f}'.format(best_epoch+1, valid_best_acc)) 
    print('training finished for: {}'.format(format_time(time.time() - t0)))

    train_info = {
        'train_loss_history' : train_loss_history,
        'train_acc_history' : train_acc_history,
        'train_loss_epochs' : train_loss_epochs,
        'train_acc_epochs' : train_acc_epochs,
        'valid_loss_history' : valid_loss_history,
        'valid_acc_history' : valid_acc_history,
        'valid_loss_epochs' : valid_loss_epochs,
        'valid_acc_epochs' : valid_acc_epochs,
        'lr_history' : lr_history,
        'cm' : best_cm,
        'probs' : best_probs,
        'preds' : best_preds,
    }
 
    return train_info
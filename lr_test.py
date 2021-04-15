#
# LR range test
#

from torch.optim.lr_scheduler import _LRScheduler

class LinearLR(_LRScheduler):
    """Linearly increases the learning rate between two boundaries over a number of
    iterations.
    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float): the final learning rate.
        num_iter (int): the number of iterations over which the test occurs.
        last_epoch (int, optional): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr

        if num_iter <= 1:
            raise ValueError("`num_iter` must be larger than 1")
        self.num_iter = num_iter

        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        r = self.last_epoch / (self.num_iter - 1)
        return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]

class ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.
    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float): the final learning rate.
        num_iter (int): the number of iterations over which the test occurs.
        last_epoch (int, optional): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr

        if num_iter <= 1:
            raise ValueError("`num_iter` must be larger than 1")
        self.num_iter = num_iter

        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        r = self.last_epoch / (self.num_iter - 1)
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]

def lr_range_test(
    model,
    device,
    train_loader,
    valid_loader,
    optimizer_name,
    num_iter,
    verbose):

    print_every = 100

    loss_meter = AverageMeter()
    score_meter = AccuracyMeter()
    lr_history = []

    model.to(device)
    criterion = nn.CrossEntropyLoss()

    learning_rate = 1 / num_iter
    weight_decay = 0
    optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate, weight_decay)
    scheduler = get_scheduler('LRRangeTest', optimizer, num_iter)

    model.train()

    t0 = time.time()

    generator = iter(train_loader)
    for index in range(num_iter):

        try:
            # Samples the batch
            x_batch, y_batch = next(generator)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            generator = iter(train_loader)
            x_batch, y_batch = next(generator)

        x_batch = x_batch.to(device, dtype=GlobalConfig.dtype)
        y_batch = y_batch.to(device, dtype=torch.long)

        output = model(x_batch)
        loss = criterion(output, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item())
        score_meter.update(y_batch, output)

        lr_history.append(scheduler.get_last_lr())  
        scheduler.step()

        if index % print_every == 0:
            if verbose:
                print('lr range test iter: {:>3d}, time: {}'.format(index, format_time(time.time() - t0)))
            
    if verbose:
        print('')
        print('lr range test finished for: {}'.format(format_time(time.time() - t0)))

    train_info = {
        'train_loss_history' : loss_meter.history,
        'train_score_history' : score_meter.history,
        'lr_history' : lr_history,
    }

    return train_info, loss_meter
import torch
import torch.nn as nn

class LabelSmoothing(nn.Module):
    def __init__(self, smoothing = 0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze()
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class LabelSmoothingV2(nn.Module):
    def __init__(self, smoothing = 0.1, num_cls = 10):
        super(LabelSmoothingV2, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_cls = num_cls

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(x)
            true_dist.fill_(self.smoothing / (self.num_cls - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * logprobs, dim=-1))
import numpy as np
import torch

from sklearn.metrics import roc_auc_score

class RocAucMeter(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.y_true = np.vstack((
            np.array([0] * self.num_classes),
            np.array([1] * self.num_classes)
        ))
        self.y_pred = np.full((2, self.num_classes), 0.5)

    def update(self, y_true, y_pred):
        # y_pred = (y_pred > 0).float()
        y_pred = torch.sigmoid(y_pred)

        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        self.y_true = np.vstack((self.y_true, y_true))
        self.y_pred = np.vstack((self.y_pred, y_pred))

    def compute_score(self):
        return [roc_auc_score(self.y_true[:,i], self.y_pred[:,i]) for i in range(self.num_classes)]
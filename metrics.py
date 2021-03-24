import numpy as np
import torch

from sklearn.metrics import roc_auc_score, confusion_matrix

class RocAucMeter():
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

    def compute_cm(self):
        return [confusion_matrix(self.y_true[:,i], (self.y_pred[:,i] > 0.5).astype(float)) for i in range(self.num_classes)]


class AccuracyMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.history = []
        self.predictions = []
        self.ground_truth = []
        self.total_correct_samples = 0
        self.total_samples = 0

    def update(self, y_batch, output):
        # TODO: check if it use GPU memory
        indices = torch.argmax(output, 1)
        correct_samples = torch.sum(indices == y_batch)
        accuracy = float(correct_samples) / y_batch.shape[0]

        self.history.append(accuracy)               
        self.total_correct_samples += correct_samples
        self.total_samples += y_batch.shape[0]

        # Save data for calculating of confusion matrix
        self.predictions.append(indices)
        self.ground_truth.append(y_batch)

    def compute_score(self):
        return float(self.total_correct_samples) / self.total_samples
        
    def compute_cm(self):
        predictions = torch.cat(self.predictions).cpu().numpy()
        ground_truth = torch.cat(self.ground_truth).cpu().numpy()

        return confusion_matrix(ground_truth, predictions)

class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.history = []
        self.total_sum = 0
        self.total_count = 0

    def update(self, x):
        self.history.append(x)
        self.total_sum += x
        self.total_count += 1

    def compute_average(self):
        return np.mean(self.history)





    
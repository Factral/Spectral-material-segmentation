import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix


class Metrics:
    """
    A class to compute and manage evaluation metrics for the material segmentation task.
    """
    def __init__(self, num_classes=44, ignore_index=255):
        """
        Initialize the Metrics class.

        Args:
            num_classes (int): Number of classes in the dataset.
            ignore_index (int): Label index to ignore when updating metrics.
        """
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))     


    def compute(self):
        """
        Compute pixel accuracy, mean accuracy, and mean Intersection over Union (IoU).

        Returns:
            tuple: pixel accuracy, mean accuracy, mean IoU
        """
        pixel_acc = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + 1e-9)
        per_class_acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + 1e-9)
        macc = per_class_acc.sum() / ((np.sum(self.confusion_matrix, axis=1) > 0).sum() + 1e-9)
        per_class_iou = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1)
            + np.sum(self.confusion_matrix, axis=0)
            - np.diag(self.confusion_matrix)
            + 1e-9
        )
        miou = per_class_iou.sum() / ((np.sum(self.confusion_matrix, axis=1) > 0).sum() + 1e-9)

        return pixel_acc, macc, miou
    

    def update(self, pred, label):
        """
        Update the confusion matrix based on predictions and labels.

        Args:
            pred (torch.Tensor): Predictions from the model.
            label (torch.Tensor): Ground truth labels.
        """

        gt = label.cpu().numpy()
        mask = gt != self.ignore_index
        pred = pred.numpy()

        # Update confusion matrix with valid entries
        self.confusion_matrix += sklearn_confusion_matrix(
            gt[mask], pred[mask], labels=[_ for _ in range(self.num_classes)]
        )

    
    def reset(self):
        """
        Reset the confusion matrix to zero.
        """
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))     



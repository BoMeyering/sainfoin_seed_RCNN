import torch
from torchmetrics.detection import IntersectionOverUnion
preds = [
   {
       "boxes": torch.tensor([[120, 134, 234, 256],
                              [457, 234, 473, 245],
                              [10, 345, 23, 362],
                              [109, 203, 115, 225],
                              [542, 342, 567, 377]]),
       "scores": torch.tensor([0.999, 0.96, 0.13, .87, .98]),
       "labels": torch.tensor([1, 1, 1, 2, 2]),
   }
]
target = [
   {
       "boxes": torch.tensor([[457, 234, 473, 245],
                              [118, 132, 233, 258],
                              [15, 355, 29, 357],
                              [542, 342, 569, 375]]),
       "labels": torch.tensor([1, 1, 2, 1]),
   }
]
metric = IntersectionOverUnion(iou_threshold=.5, respect_labels=True, class_metrics=True)
print(metric(preds, target))

from model.model import create_model
from config import train_dir, val_dir, test_dir, annotation_path, chkpt_dir, tensorboard_dir, log_dir, inference_dir
from config import device, cores, classes, n_classes, resize_to, n_epochs, batch_size
from config import base_name, lr, momentum, gamma

from torchmetrics.classification import BinaryAccuracy

train_accuracy = BinaryAccuracy()
valid_accuracy = BinaryAccuracy()

for epoch in range(n_epochs):
    for x, y in train_data:
        y_hat = model(x)

        # training step accuracy
        batch_acc = train_accuracy(y_hat, y)
        print(f"Accuracy of batch{i} is {batch_acc}")

    for x, y in valid_data:
        y_hat = model(x)
        valid_accuracy.update(y_hat, y)

    # total accuracy over all training batches
    total_train_accuracy = train_accuracy.compute()

    # total accuracy over all validation batches
    total_valid_accuracy = valid_accuracy.compute()

    print(f"Training acc for epoch {epoch}: {total_train_accuracy}")
    print(f"Validation acc for epoch {epoch}: {total_valid_accuracy}")

    # Reset metric states after each epoch
    train_accuracy.reset()
    valid_accuracy.reset()
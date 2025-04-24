"""
src/callbacks.py
Torch model callback classes and functions
BoMeyering 2025
"""

import torch
import os
import logging
from datetime import datetime

class ModelCheckpoint:
    def __init__(self, filepath, monitor='val_loss', top_k=5, metadata=None):
        self.filepath = filepath
        self.monitor = monitor
        self.monitor_op = torch.lt  # assume lower is better (e.g. val_loss)
        self.logger = logging.getLogger()
        self.metadata = metadata if metadata is not None else {}
        self.top_k = top_k
        self.top_checkpoints = []  # min-heap of (val_loss, filepath)

    def __call__(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            self.logger.warning(f"Warning: Metric '{self.monitor}' is not available. Skipping checkpoint.")
            return None

        # If we don't have enough checkpoints yet or current is better than the worst of top_k
        should_save = len(self.top_checkpoints) < self.top_k or self.monitor_op(current, self.top_checkpoints[-1][0])
        if should_save:
            now = datetime.now().isoformat(timespec='seconds', sep='_').replace(":", ".")
            chkpt_filename = f"{self.filepath}_epoch_{epoch}_vloss-{current:.6f}.pth"
            chkpt = {
                'model_state_dict': logs['model_state_dict'],
                'epoch': epoch,
                'monitor': self.monitor,
                self.monitor: current,
                **self.metadata
            }
            torch.save(chkpt, chkpt_filename)
            self.logger.info(f"Epoch {epoch} - '{self.monitor}' improved or is in top-{self.top_k}. Saved to {chkpt_filename}")

            self.top_checkpoints.append((current, chkpt_filename))
            self.top_checkpoints.sort(key=lambda x: x[0])  # sort by val_loss (ascending)

            # If we now have too many checkpoints, remove the worst
            if len(self.top_checkpoints) > self.top_k:
                worst_loss, worst_path = self.top_checkpoints.pop()
                if os.path.exists(worst_path):
                    os.remove(worst_path)
                    self.logger.info(f"Removed checkpoint: {worst_path} (no longer in top-{self.top_k})")
        else:
            self.logger.info(f"Epoch {epoch} - '{self.monitor}' did not improve top-{self.top_k}. Skipping checkpoint.")
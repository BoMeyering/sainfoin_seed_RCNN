
import logging.handlers
import os
import sys
import logging
import torch
import omegaconf
from pathlib import Path

def setup_loggers(conf: omegaconf.OmegaConf):
    """
    Configures a simple logger to log outputs to the console and the output file.

    conf:
        conf (omegaconf.OmegaConf): arguments object from the configuration file.
    """
    filename = conf.run_name + '.log'
    filepath = Path(conf.directories.log_dir) / filename
    if not os.path.exists(conf.directories.log_dir):
        os.mkdir(conf.directories.log_dir)

    root_logger = logging.getLogger()

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.handlers.RotatingFileHandler(filepath, 'a', 1000000, 3)
    stream_handler = logging.StreamHandler(sys.stdout,)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    root_logger.setLevel(logging.DEBUG)

class TBLogger:
    def __init__(self, writer: torch.utils.tensorboard.writer.SummaryWriter):
        self.writer = writer

    def log_scalar_dict(self, main_tag, scalar_dict, step):
        # Epoch Average Metric Logging
        for k, v in scalar_dict.items():
            self.writer.add_scalar(
                tag=f"{main_tag}/{k}", scalar_value=v, global_step=step, new_style=True
            )

    def log_tensor_dict(self, main_tag, tensor_dict, step, class_map):
        # Log each tensor value for each key in the tensor_dict
        for k, v in tensor_dict.items():
            for i, tensor_value in enumerate(v):
                class_name = class_map[str(i)]
                self.writer.add_scalar(
                    tag=f"{main_tag}/{k}/{class_name}",
                    scalar_value=tensor_value,
                    global_step=step,
                    new_style=True,
                )
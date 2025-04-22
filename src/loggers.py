
import logging.handlers
import os
import sys
import logging
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

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.handlers.RotatingFileHandler(filepath, 'a', 1000000, 3)
    stream_handler = logging.StreamHandler(sys.stdout,)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    root_logger.setLevel(logging.DEBUG)
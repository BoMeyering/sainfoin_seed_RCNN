import logging
import datetime

def create_logger(output_file: str=None):
    """
    
    """
    today = datetime.date.today()

    logger = logging.getLogger("main_app")
    logger.propagate = False

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s: - %(message)s')

    # Set stream level configurations
    s_handler = logging.StreamHandler()
    s_handler.setLevel(logging.DEBUG)
    s_handler.setFormatter(formatter)
    
    logger.addHandler(s_handler)

    if output_file is not None:
        f_handler = logging.FileHandler(f'./logs/app_{today}.log')
        f_handler.setLevel(logging.DEBUG)
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)

    return logger
import logging
import datetime

def create_logger():
    """
    
    """
    today = datetime.date.today()

    logger = logging.getLogger("main_app")
    logger.propagate = False

    logger.setLevel(logging.DEBUG)

    s_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(f'./logs/app_{today}.log')

    s_handler.setLevel(logging.DEBUG)
    f_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s: - %(message)s')
    s_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    logger.addHandler(s_handler)
    logger.addHandler(f_handler)

    return logger
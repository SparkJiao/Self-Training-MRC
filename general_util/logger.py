import logging
import sys
import os


_root_name = 'BERT_for_RC'


def get_child_logger(child_name):
    return logging.getLogger(_root_name + '.' + child_name)


def setting_logger(log_file: str):
    model_name = "-".join(log_file.replace('/', ' ').split()[1:])

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(_root_name)
    logger.setLevel(logging.INFO)

    rf_handler = logging.StreamHandler(sys.stderr)
    rf_handler.setLevel(logging.INFO)
    rf_handler.setFormatter(logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                              datefmt='%m/%d/%Y %H:%M:%S'))

    output_dir = './log_dir'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    f_handler = logging.FileHandler(os.path.join(output_dir, model_name + '-output.log'))
    f_handler.setLevel(logging.INFO)
    f_handler.setFormatter(logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                                             datefmt='%m/%d/%Y %H:%M:%S'))

    # logger.addHandler(rf_handler)
    logger.addHandler(f_handler)
    return logger

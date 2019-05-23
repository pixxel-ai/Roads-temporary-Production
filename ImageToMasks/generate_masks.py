"""
Takes as input:
    imsize : Model Image Size (not output image size)
model_path : Path of the model file
 in_folder : Path containing images to predict on
out_folder : Path to output masks to
 resize_to : [OPTIONAL] Output image size of masks and graphs

Outputs:
Predicted masks for each image in `out_folder`
"""
import argparse
from pathlib import Path
import datetime
import logging
from .predict import *
# -------- Setting up logging -------- #
logger = logging.getLogger(__name__)
f_handler = logging.FileHandler('logs.log', filemode='w')
f_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
f_handler.setFormatter(f_formatter)
logger.addHandler(f_handler)

curr_time = datetime.datetime.now().__str__()
logger.info("######################--" + curr_time + "--#####################")


def log_value(name: str):
    logger.info(name + ' : ' + str(globals()[name]))
# ------------------------------------ #


def get_arguments():
    parser = argparse.ArgumentParser(description='Images to Masks')
    parser.add_argument('model_path', type=str, help="Absolute path of the model file (eg : export.pkl)")
    parser.add_argument('in_folder', type=str, help="Absolute path of folder containing images to predict on")
    parser.add_argument('out_folder', type=str, help="Absolute path of folder to store masks in")
    parser.add_argument('model_imsize', type=int, help="Image size that the model has been trained on")
    parser.add_argument('--resize_to', type = int, default=-1, help="Output image size of masks and graphs")

    arguments = parser.parse_args()

    global model_path, in_folder, out_folder, model_imsize, resize_to

    model_path = Path(arguments.model_path);log_value('model_path')
    in_folder = Path(arguments.in_folder); log_value('in_folder')
    out_folder = Path(arguments.out_folder); log_value('out_folder')
    model_imsize = int(arguments.model_imsize); log_value('model_imsize')
    resize_to = int(arguments.resize_to); log_value('resize_to')

    logging


if __name__ == '__main__':
    get_arguments()
    process(model_path, in_folder, out_folder, model_imsize, resize_to, logger)

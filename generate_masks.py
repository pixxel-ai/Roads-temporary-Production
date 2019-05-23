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
from fastai.vision import *
import argparse
from pathlib import Path
import datetime
import logging
from ImageToMasks.predict import *
import sys
sys.path.append('/home/akash/Roads/Models/Roads-temporary-Production/ImageToMasks/')
# -------- Setting up logging -------- #
logger = logging.getLogger(__name__)
f_handler = logging.FileHandler('logs.log', 'w')
f_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
f_handler.setFormatter(f_formatter)
logger.addHandler(f_handler)
logger.setLevel(logging.DEBUG)

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
    
    out_folder.mkdir(exist_ok=True)


if __name__ == '__main__':
    get_arguments()

    # Loading trained fastai model
    model = load_learner(model_path.parent, model_path.name)

    process(model, in_folder, out_folder, model_imsize, resize_to, logger)

"""
Generates masks and graphs for each image in a given folder using a trained Fastai model.
   Parameters:
  model_path : Path of the Fastai model file
   in_folder : Path containing images to predict on
  out_folder : Path to output masks to
      model_imsize : Model Image Size (not output image size)
graphs_output: Path to file (without extension) to which graphs will be output
   resize_to : [OPTIONAL] Output image size of masks and graphs

      Outputs: Predicted masks for each image in `out_folder`
               Graphs generated for each mask in `graphs_output.txt`
"""

# from fastai.vision import *
import argparse
from pathlib import Path
from datetime import datetime
import logging
# from ImageToMasks.predict import *
from sys import path
from os import system
path.append('/home/akash/Roads/Models/Roads-temporary-Production/ImageToMasks/')


# -------- Set up logging -------- #
logger = logging.getLogger(__name__)
f_handler = logging.FileHandler('logs.log', 'a') # We will always append to the log file

# Format for logging information
f_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
f_handler.setFormatter(f_formatter)
logger.addHandler(f_handler) # Attach all the above configuration to the logger object
logger.setLevel(logging.DEBUG) # Makes sure all kinds of information is logged.

curr_time = datetime.now().__str__()
logger.info("\n\n\n######################--" + curr_time + "--#####################")


def log_value(name: str, value, logger=logger):
    """
    Logs value of global variable called `name`
    :param name: Name of GLOBAL variable in string format
    :param value: value of `name` variable
    :return: The string logged (without formatting) if logging is successful.
             -1 otherwise.
    """
    # todo: Write tests
    to_log = name + ' : ' + str(value)
    logger.info(to_log)
    return to_log
# ------------------------------------ #


def get_arguments():
    """
    Handles command line input
    Defines all variables as GLOBAL variables
    Creates `out_folder` directory if it doesn't exist.
    :return: No return value
    """
    #todo: write tests
    parser = argparse.ArgumentParser(description='Images to Masks')
    parser.add_argument('model_path', type=str, help="Absolute path of the model file (eg : export.pkl)")
    parser.add_argument('in_folder', type=str, help="Absolute path of folder containing images to predict on")
    parser.add_argument('out_folder', type=str, help="Absolute path of folder to store masks in")
    parser.add_argument('model_imsize', type=int, help="Image size that the model has been trained on")
    parser.add_argument('graphs_output', type=str, help="Absolute path of file (without extension) to which graphs will be "
                                                      + "output in LineString format")
    parser.add_argument('--resize_to', type = int, default=-1, help="Output image size of masks and graphs")

    arguments = parser.parse_args()

    global model_path, in_folder, out_folder, model_imsize, resize_to, graphs_output

    model_path = Path(arguments.model_path);log_value('model_path', model_path)
    in_folder = Path(arguments.in_folder); log_value('in_folder', in_folder)
    out_folder = Path(arguments.out_folder); log_value('out_folder', out_folder)
    out_folder.mkdir(exist_ok=True)
    model_imsize = int(arguments.model_imsize); log_value('model_imsize', model_imsize)
    graphs_output = Path(arguments.graphs_output); log_value('graphs_output', graphs_output)
    resize_to = int(arguments.resize_to); log_value('resize_to', resize_to)


if __name__ == '__main__':
    get_arguments()
    # Loading trained fastai model
    model = load_learner(model_path.parent, model_path.name)
    process(model, in_folder, out_folder, model_imsize, resize_to, logger)

    # Creating graphs
    masks_to_graphs = Path("MaskToGraph/mask_to_graph.py").absolute()
    os.system("python3 " + str(masks_to_graphs) + " " + str(out_folder) + " " + str(graphs_output))
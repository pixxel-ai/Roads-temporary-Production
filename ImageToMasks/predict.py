from fastai.vision import *
from tqdm import tqdm as tqdm
import logging

def get_image(path, imsize=224):
    """
    Returns image at `path` wrapped in the Fastai Image Class.
    :param path: Path to image
    :param imsize: Resize image to `imsize`
    :return: `image` resized to `imsize` wrapped in `fastai.vision.Image`
             'skip' if the image at `path` is not readable
    """
    # todo: Write tests
    try:
        return open_image(path).resize(imsize)
    except:
        return 'skip'


def get_prediction(model, image, resize_to):
    """
    Returns the predicted mask for `image` using `model` resized to `resize_to` (if applicable)
    :param model: Trained Fastai Model
    :param image: image to make prediction on
    :param resize_to: the predicted mask will be resized to `resize_to` before returning
    :return: Binary Mask (1 channel image) predicted by `model` on `image`
            [type: fastai.vision.ImageSegment]
    """
    #todo: Write tests
    if resize_to != -1:  # Pass -1 to return images with size `model_imsize`
        return (model.predict(image)[0]).resize(resize_to)
    else:
        return model.predict(image)[0]


def process(model, in_folder, out_folder, imsize, resize_to, logger: logging.Logger):
    """
    Generates mask for each image in `in_folder` and stores them in `out_folder` using `model`

    :param model: Trained Fastai Model
    :param in_folder: folder containing images to predict on
    :param out_folder: folder to output masks to
    :param imsize: model image size (as in get_image())
    :param resize_to: output mask size (as in get_prediction)
    :return: No return value
    """
    #todo: Write tests
    logger.info("\n \n \n NOW MAKING PREDICTIONS \n \n")
    for f in tqdm(in_folder.iterdir()):
        image = get_image(f, imsize=imsize)
        if image == 'skip':
            if logger:
                logger.debug("SKIPPED : " + str(f))
            continue
        else:
            pred = get_prediction(model, image, resize_to)
            pred.save(out_folder/f.name)
            if logger:
                logger.info("PROCESSED : " + str(f))
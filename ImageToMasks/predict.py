from fastai.vision import *
from tqdm import tqdm as tqdm
import logging


def get_image(path, imsize=224):
    """
    :param path: Path to image
    :param imsize: Resize image to `imsize`
    :return: `image` resized to `imsize` wrapped in `fastai.vision.Image`

    The function returns 'skip' if the image at `path` is not readable
    """
    try:
        return open_image(path).resize(imsize)
    except:
        return 'skip'


def get_prediction(model, image, resize_to):
    """
    :param model: Path to model file (eg: `export.pkl`)
    :param image: image to make prediction on
    :param resize_to: the predicted mask will be resized to `resize_to` before returning
    :return: Binary Mask (1 channel image) predicted by `model` on `image`
    [type: fastai.vision.ImageSegment]
    """
    if resize_to != -1:  # Pass -1 to return images with size `model_imsize`
        return (model.predict(image)[0]).resize(resize_to)
    else:
        return model.predict(image)[0]


def process(model, in_folder, out_folder, imsize=224, resize_to=-1, logger: logging.Logger):
    """
    Generates mask for each image in `in_folder` and stores them in `out_folder` using `model`

    :param model: path to model (as in get_prediction())
    :param in_folder: folder containing images to predict on
    :param out_folder: folder to output masks to
    :param imsize: model image size (as in get_image())
    :param resize_to: output mask size (as in get_prediction)
    :return: No return value
    """
    logger.info("NOW MAKING PREDICTIONS")
    for f in tqdm(in_folder.iterdir()):
        image = get_image(f, imsize=imsize)
        if image == 'skip':
            if logger:
                logger.info("SKIPPED : " + str(f))
            continue
        else:
            pred = get_prediction(model, image, resize_to)
            pred.save(out_folder/f.name)
            if logger:
                logger.info("PROCESSED : " + str(f))
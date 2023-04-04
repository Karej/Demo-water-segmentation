import os, cv2
import numpy as np
import pandas as pd
import sys
import torch
from pathlib import Path
from torch.utils import data
# import albumentations as album
from PIL import Image
import torchvision.transforms as tf
from utility import utils

def norm_imagenet(img_pil, dims):
    """
    Normalizes and resizes input image
    :param img_pil: PIL Image
    :param dims: Model's expected input dimensions
    :return: Normalized Image as a Tensor
    """

    # Mean and stddev of ImageNet dataset
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    # Resize, convert to tensor, normalize
    transform_norm = tf.Compose([
        tf.Resize([dims[0], dims[1]]),
        tf.ToTensor(),
        tf.Normalize(mean, std)
    ])

    img_norm = transform_norm(img_pil)
    return img_norm

def predict_one(path, model, device):
    """
    Predicts a single image from path
    :param path: Path to image
    :param model: Loaded Torch Model
    :param mask_outdir: Filepath to mask out directory
    :param overlay_outdir: Filepath to overlay out directory
    :return: None
    """
    img_pil = utils.load_image_in_PIL(os.path.join(os.getenv('STORAGE'),'image',path))

    # Prediction is an PIL Image of 0s and 1s
    prediction = predict_pil(model, img_pil, model_dims=(416, 416), device=device)

    basename = str(Path(os.path.basename(path)).stem)
    mask_savepth = os.path.join(os.getenv('STORAGE'),'mask',basename + '.png')
    # mask_save = prediction.convert('RGB')
    prediction.save(mask_savepth)

    over_savepth = os.path.join(os.getenv('STORAGE'),'overlay',basename + '.png')
    img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    overlay_np = utils.add_overlay(img_np, np.array(prediction))
    cv2.imwrite(over_savepth, overlay_np)

def predict_pil(model, img_pil, model_dims, device):
    """
    Predicts a single PIL Image
    :param model: Loaded PyTorch model
    :param img_pil: PIL image
    :param model_dims: Model input dimensions
    :return: Segmentation prediction as PIL Image
    """

    img_np = np.array(img_pil)
    img_tensor_norm = norm_imagenet(img_pil, model_dims)

    # Pipeline to resize the prediction to the original image dimensions
    pred_resize = tf.Compose([tf.Resize([img_np.shape[0], img_np.shape[1]])])

    # Add extra dimension at front as model expects input 1*3*dimX*dimY (batch size of 1)
    input_data = img_tensor_norm.unsqueeze(0)

    try:
        # print("Converted input image to cuda.")
        prediction = model.predict(input_data.to(device))
    except:
        print("Did not convert input image to cuda.")
        prediction = model.predict(input_data)
    prediction = pred_resize(prediction)
    # prediction =
    # # prediction = np.where(prediction>0.001,1.0,0.0)


    prediction = utils.postprocessing_pred( prediction.squeeze().cpu().round().numpy().astype(np.uint8))
    prediction = Image.fromarray(prediction).convert('P')
    prediction.putpalette(utils.color_palette)
    return prediction

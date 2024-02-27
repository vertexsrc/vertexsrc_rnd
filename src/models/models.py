from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch
import requests

def YOLOS():
    '''
    https://huggingface.co/hustvl/yolos-tiny

    YOLOS model fine-tuned on COCO 2017 object detection (118k annotated images).
    Vision transformer trained using the DETR loss
    '''
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
    image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

    return model, image_processor
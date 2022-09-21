import pickle, json 

import numpy as np
import operator as op 
import itertools as it, functools as ft 

import cv2 
from PIL import Image 

import torch as th 
import torch.nn as nn 

from os import path 
from glob import glob
from time import time 

from rich.progress import track 
from typing import Dict, Any

from torchvision.utils import make_grid
from torchvision import models as models_downloader, transforms as T  

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from sentence_transformers import SentenceTransformer
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, CLIPTextModel, CLIPTokenizer 
from libraries.log import logger
from settings import MAP_SOURCE_LANGUAGE2CODE 


map_serializer2mode = {
    json: ('r', 'w'), 
    pickle: ('rb', 'wb')
}

def measure(func):
    @ft.wraps(func)
    def _measure(*args, **kwargs):
        start = int(round(time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            duration = end_ if end_ > 0 else 0
            logger.debug(f"{func.__name__:<20} total execution time: {duration:04d} ms")
    return _measure

def is_valid(var_value, var_name, var_type=None):
    if var_value is None:
        raise ValueError(f'{var_name} is not defined | please look the helper to see available env variables')
    if var_type is not None:
        if not op.attrgetter(var_type)(path)(var_value):
            raise ValueError(f'{var_name} should be a valid file or dir')

def serialize(path2location, data, serializer=pickle):
    mode = map_serializer2mode.get(serializer, None)
    if mode is None:
        raise ValueError(f'serializer option must be pickle or json')
    with open(path2location, mode=mode[1]) as fp:
        serializer.dump(data, fp)

def deserialize(path2location, serializer=pickle):
    mode = map_serializer2mode.get(serializer, None)
    if mode is None:
        raise ValueError(f'serializer option must be pickle or json')
    with open(path2location, mode=mode[0]) as fp:
        data = serializer.load(fp)
        return data 

def pull_images(path2images, exts='*.jpg'):
    return sorted( glob(path.join(path2images, '**' ,exts), recursive=True) )

def th2cv(th_image):
    red, green, blue = th_image.numpy()
    return cv2.merge((blue, green, red))

def cv2th(cv_image):
    blue, green, red = cv2.split(cv_image)
    return th.as_tensor(np.stack([red, green, blue]))

def cv2pil(cv_image):
    return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

def pil2cv(pil_image):
    return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

def read_image(path2image, convert2cv=False, size=None):
    pil_image = Image.open(path2image).convert('RGB')
    if convert2cv:
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        if size is not None:
            return cv2.resize(cv_image, size, interpolation=cv2.INTER_CUBIC)
        return cv_image
    return pil_image 

def save_image(cv_image, path2location):
    cv2.imwrite(path2location, cv_image)

def merge_images(images, nb_images_per_row=2):
    acc = []
    for cv_image in images:
        th_image = cv2th(cv_image)
        acc.append(th_image)
    # ...! 
    
    th_image = make_grid(acc, nb_images_per_row)
    return th2cv(th_image)

def load_transformer(path2transformer):
    if path.isfile(path2transformer):
        logger.debug(f'transformer was found | it will be load from {path2transformer}')
        vectorizer = deserialize(path2transformer, pickle)
    else:
        try:
            _, file_name = path.split(path2transformer)
            vectorizer = SentenceTransformer(file_name)
            logger.debug('transformer was downloaded')
            serialize(vectorizer, path2transformer, pickle)
        except Exception as e:
            logger.error(e)
            raise Exception(f'can not download {file_name}')    
    return vectorizer

def vectorize(data, vectorizer, device='cpu'):
    fingerprint = vectorizer.encode(data, device=device)
    return fingerprint

def prepare_image(th_image):
    normalied_th_image = th_image / th.max(th_image)
    return T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(normalied_th_image)

def load_vectorizer(path2vectorizer, device='cpu', index=-1):
    if path.isfile(path2vectorizer):
        logger.debug('the model was found ... it will be loaded')
        vectorizer = th.load(path2vectorizer, map_location=device) 
    else:
        logger.debug('the model was not found, download will start')
        _, vectorizer_filename = path.split(path2vectorizer)
        vectorizer_name, _ = vectorizer_filename.split('.') 
        try:
            vectorizer = op.attrgetter(vectorizer_name)(models_downloader)(pretrained=True)
            th.save(vectorizer, path2vectorizer)
        except Exception as e:
            raise ValueError(f'{vectorizer_name} is not a valid model. check torchvision models list')
    
    vectorizer = nn.Sequential(*list(vectorizer.children())[:index])
    for prm in vectorizer.parameters():
        prm.requires_grad = False
    vectorizer.to(device)
    return vectorizer.eval()

def load_diffusion_pipeline(path2diffusion, token):
    pipe = StableDiffusionPipeline.from_pretrained(
        path2diffusion, 
        revision="fp16", 
        torch_dtype=th.float16,
        use_auth_token=token
    )
    return pipe 

def scoring(fingerprint, fingerprint_matrix):
    scores = fingerprint @ fingerprint_matrix.T 
    X = np.linalg.norm(fingerprint)
    Y = np.linalg.norm(fingerprint_matrix, axis=1)
    W = X * Y 
    weighted_scores = scores / W 
    return weighted_scores

def top_k(weighted_scores, k=16):
    scores = th.as_tensor(weighted_scores).float()
    weights, indices = th.topk(scores, k, largest=True)
    return weights, indices.tolist()
     
def customize_openapi(app:FastAPI, app_description:Dict[str, str], api_schema:Dict[str, Any]):
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(**app_description, routes=app.routes)
    for key, value in api_schema.items():
        openapi_schema["paths"][key] = value 
    app.openapi_schema = openapi_schema
    return app.openapi_schema    

def load_translator(source_language):
    language_code = MAP_SOURCE_LANGUAGE2CODE[source_language]
    tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{language_code}-en")
    model = AutoModelForSeq2SeqLM.from_pretrained(f"Helsinki-NLP/opus-mt-{language_code}-en")
    model.eval()
    agent = {
        'tokenizer': tokenizer,
        'predictor': model
    }
    return agent 

def forward(agent, target):
    input_ids = th.tensor(
        [agent['tokenizer'].encode(target, add_special_tokens=True)]
    )
    predict = agent['predictor'].generate(input_ids)    
    response = agent['tokenizer'].decode(predict[0], skip_special_tokens=True)
    return response  
    
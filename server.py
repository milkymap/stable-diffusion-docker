import io 
import zmq 
import zmq.asyncio as aiozrmq 

import cv2
import numpy as np 
import itertools as it, functools as ft 

import json 
import pickle 

from libraries.log import logger 

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi.openapi.docs import get_swagger_ui_html

from starlette.responses import StreamingResponse
from settings import ZMQ_SERVER_PORT, API_SCHEMA


map_type2fun = {'text': str, 'number': float, 'integer': int}

app = FastAPI()

ctx = None 
dealer_socket = None 

@app.on_event('startup')
async def handle_startup():
    global ctx 
    global dealer_socket

    ctx = aiozrmq.Context()

    dealer_socket = ctx.socket(zmq.DEALER)
    dealer_socket.setsockopt(zmq.LINGER, 0)
    dealer_socket.connect(f'tcp://localhost:{ZMQ_SERVER_PORT}')

    logger.success('api service is up and ready the exchange messages')

@app.on_event('shutdown')
async def handle_shutdown():
    dealer_socket.close()
    ctx.term() 
    logger.success('api service has removed all ressources')

@app.get('/', include_in_schema=False)
async def handle_entrypoint():
    return get_swagger_ui_html(
        openapi_url='/openapi.json',
        title=app.title + " - Swagger UI"
    )

@app.post('/create_image', include_in_schema=False)
async def handle_create_image(incoming_req: Request):
    try:
        query_params = dict(incoming_req.query_params.items())
        parameters_map = {}
        for item_ in API_SCHEMA['create_image']['post']['parameters']:
            if item_['name'] in query_params:
                fun = map_type2fun[item_['schema']['type']]
                parameters_map[item_['name']] = fun(query_params[item_['name']])
            else:
                raise Exception(f'{item_["name"]} is a required option')

        await dealer_socket.send_multipart([b''], flags=zmq.SNDMORE)
        await dealer_socket.send_pyobj(parameters_map)

        _, encoded_response = await dealer_socket.recv_multipart()
        response = pickle.loads(encoded_response)
        if response['status'] == 1:
            _, encoded_images = cv2.imencode('.jpg', response['image'])
            images_iostream = io.BytesIO(encoded_images.tobytes())
            return StreamingResponse(
                status_code=200, 
                content=images_iostream, 
                media_type='image/jpg'
            )
        else:
            return JSONResponse(status_code=200, content=response)
    except Exception as e:
        catched_message = f'Exception : {e}'
        return JSONResponse(
            status_code=400, 
            content=catched_message
        )



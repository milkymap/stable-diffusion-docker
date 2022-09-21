import io 
import cv2 
import click 
import requests
import gradio as gr 

import numpy as np 
from settings import FASTAPI_SERVER_PORT, MAP_SOURCE_LANGUAGE2CODE

def gradio_handler(initial_seed, number_inference_steps, guidance_scale, width, height, number_images, source_language, prompt):
    response = requests.post(
        url=f'http://localhost:{FASTAPI_SERVER_PORT}/create_image',
        params={
            'seed': int(initial_seed), 
            'width': int(width), 
            'height': int(height),
            'guidance_scale': guidance_scale, 
            'num_inference_steps': int(number_inference_steps),
            'num_images': int(number_images), 
            'source_language': source_language,  
            'language_query': prompt
        } 
    )

    binarystream = response.content
    fingerprints = np.frombuffer(binarystream, dtype=np.uint8)
    return cv2.cvtColor(
        cv2.imdecode(fingerprints, cv2.IMREAD_COLOR), 
        cv2.COLOR_BGR2RGB
    )

def start_displayer(display_port):
    demo = gr.Interface(
        fn=gradio_handler, 
        inputs=[
            gr.Number(value=0, elem_id='initial_seed'), 
            gr.Slider(minimum=1, maximum=100, value=50), 
            gr.Slider(minimum=1, maximum=20, value=7.5),
            gr.Slider(minimum=512, maximum=1024, step=64), 
            gr.Slider(minimum=512, maximum=1024, step=64), 
            gr.Slider(1, 9, 1),
            gr.Dropdown(list(MAP_SOURCE_LANGUAGE2CODE.keys()), value='english'),
            gr.Textbox(max_lines=5, elem_id='prompt'),
        ], 
        outputs=gr.Image(shape=(512, 640)), 
        title='multilingual-stable-diffusion', 
        description='multilingual stable diffusion', 
        css="#prompt {font-weight: bold}"
    )
    demo.launch(
        server_name='0.0.0.0',
        server_port=display_port
    )

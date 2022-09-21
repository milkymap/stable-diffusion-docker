import zmq 
import click 
import uvicorn
import multiprocessing as mp 

from libraries.log import logger 
from libraries.strategies import * 

from settings import API_SCHEMA, APP_DESCRIPTIONS, ZMQ_SERVER_PORT, MAP_SOURCE_LANGUAGE2CODE
from server import app 

from os import getenv, path 
from torch.cuda.amp import autocast

@click.command()
@click.option('--gpu_index', type=int, default=0, help='for multi gpu infra')
@click.option('--token', type=str, help='huggingface token', default='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')
@click.option('--path2diffusion', type=str, default='CompVis/stable-diffusion-v1-4')
@click.option('--diffusion_model_name', type=str, default='diffusion_pipe.pkl', help='the cached(pickle) model')
@click.option('--server_port', type=int, default=8000)
@click.option('--hostname', default='0.0.0.0')
@click.option('--language_source', help='the source language', type=click.Choice(list(MAP_SOURCE_LANGUAGE2CODE.keys())), default='french')
def serving(gpu_index, token, path2diffusion, diffusion_model_name, server_port, hostname, language_source):
    try:
        path2cache = getenv('CACHE')
        is_valid(path2cache, 'PATH2CACHE', 'isdir')

        path2cached_model = path.join(path2cache, diffusion_model_name)
        if path.isfile(path2cached_model):
            logger.debug(f'diffusion model will be loaded from {path2cached_model}')
            pipe = deserialize(path2cached_model, pickle)
        else:
            logger.debug(f'there is no internal diffusion model at {path2cached_model}')
            logger.debug(f'the model will be download form {path2diffusion}')
            pipe = load_diffusion_pipeline(path2diffusion, token)
            logger.debug(f'diffusion model was dumped at {path2cached_model}')
            serialize(path2cached_model, pipe, pickle)
        
        logger.debug('load the translator')
        translator = load_translator(language_source)

        logger.success('the translator was loaded')

        logger.debug('move model to cuda (can take some time)')
        pipe = pipe.to(f'cuda:{gpu_index}')
        pipe.enable_attention_slicing()
        logger.success('model is ready')

        zmq_ctx = zmq.Context()
        router_socket = zmq_ctx.socket(zmq.ROUTER)
        router_socket.setsockopt(zmq.LINGER, 0)
        router_socket.bind(f'tcp://*:{ZMQ_SERVER_PORT}')

        router_poller = zmq.Poller()
        router_poller.register(router_socket, zmq.POLLIN)
        ZMQ_INIT = 1

        logger.success('zmq service is up')

        app.openapi = lambda: customize_openapi(app, APP_DESCRIPTIONS, API_SCHEMA)
        api_process = mp.Process(
            target=uvicorn.run, 
            kwargs={'app': app, 'port': server_port, 'host': hostname}
        )

        api_process.start()

        keep_routing = True
        logger.debug(f'server is up and listens at port {ZMQ_SERVER_PORT}')
             
        while keep_routing:
            incoming_events = dict(router_poller.poll(5000))
            if router_socket in incoming_events:
                if incoming_events[router_socket] == zmq.POLLIN:
                    client_address, delimiter, encoded_msg = router_socket.recv_multipart()
                    try:
                        decoded_msg = pickle.loads(encoded_msg)
                        seed = decoded_msg['seed']
                        num_images = decoded_msg['num_images']

                        width = decoded_msg['width']
                        height = decoded_msg['height']
                        guidance_scale = decoded_msg['guidance_scale']
                        num_inference_steps = decoded_msg['num_inference_steps']

                        language_query = decoded_msg['language_query']
                        source_language = decoded_msg['source_language']
                        if source_language == 'french':
                            translated_query = forward(translator, language_query)
                            logger.debug(f'translation : {translated_query}')
                        elif source_language == 'english':  # do not apply translation if source_language is english 
                            translated_query = language_query
                        else:
                            raise Exception(f'{source_language} is not supported yet. Use french or english')

                        with autocast(True):
                            if seed > 0:
                                generator = th.Generator(f"cuda:{gpu_index}").manual_seed(seed)
                            else:
                                generator = None 
                            
                            generated_images = pipe(
                                width=width, 
                                height=height, 
                                prompt=[translated_query] * num_images,
                                generator=generator,  
                                guidance_scale=guidance_scale, 
                                num_inference_steps=num_inference_steps 
                            ).images
                            
                            if num_images == 1:
                                bgr_image = pil2cv(generated_images[0])
                            else:
                                bgr_images = list(map(pil2cv, generated_images))
                                bgr_image = merge_images(bgr_images)

                        encoded_rsp = pickle.dumps({'status': 1,'image': bgr_image})
                    except Exception as e:
                        logger.error(e)
                        encoded_rsp = pickle.dumps({'status': 0, 'data': {}})
                    router_socket.send_multipart([client_address, delimiter, encoded_rsp])
        # end while routing loop

    except KeyboardInterrupt as e:
        pass 
    except Exception as e:
        logger.error(e)
    finally:
        if ZMQ_INIT:
            router_poller.unregister(router_socket)
            router_socket.close()
            zmq_ctx.term()
            logger.success('zmq services has removed all ressources')

if __name__ == '__main__':
    logger.debug('... image-generation ...')
    serving()
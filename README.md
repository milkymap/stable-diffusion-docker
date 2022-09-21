# image-semantic-search
This tools allows to use stable diffusion model

# docker env variables 
* CACHE 
    * model will be cached here 
    * the model is huge so it is important to cached it 
* TRANSFORMERS_CACHE

# initialization 
* create an internal cache dir 
* mount this dir to the docker system with -v path2internal_cache:/home/solver/cache

```bash
    mkdir cache 
    mkdir cache/transformers
    # -v path2cache:/home/solver/cache 
```

# build and run server-mode for cpu 
```bash
docker build -t stable-diffusion:0.0 -f Dockerfile.gpu .
docker run --rm --name stable-diffusion --interactive --tty --gpus all -v path2cache:/home/solver/cache -p hostport:server_port -p hostport:display_port  stable-diffusion:0.0 --server_port 8000 --display_port 7068 --language_source 'french' --token your-token
``` 

# examples of prompts 
```
    seed = 4276421838
    guidance_sclace = 7
    width, height = 512, 512
    prompt = gal gadot, a colorful and vibrant majestic white queen open wide eyes drops a tear with flowers on her hair, glowing light orbs, intricate concept art, elegant, digital painting, smooth, sharp focus, ethereal opalescent mist, outrun, vaporware, cyberpunk darksynth, ethereal, ominous, misty, 8 k, by ruan jia and miho hirano, 8 k, rendered in octane 
```


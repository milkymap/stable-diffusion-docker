ZMQ_SERVER_PORT=1200
FASTAPI_SERVER_PORT=8000
MAP_SOURCE_LANGUAGE2CODE = {'english': 'en', 'german': 'de', 'french': 'fr', 'spanish': 'es', 'italian': 'it', 'arabic': 'ar', 'chinese': 'zh', 'russian': 'ru'}

APP_DESCRIPTIONS = {
    "title":"image generation", 
    "version": 0.1, 
    "description":"""
        This app is an image-generator based on stable-diffusion
        It exposes two ressources : 
            /create_image 
            /change_image (next version) 
        it allows the user to create image from a query 
        it is also possible to modify an existing image by using another query 
    """
}

API_SCHEMA = {
    "/create_image":{
        "post": {
            "parameters": [
                {
                    "name": "seed",
                    "in": "query",
                    "description": "initialisation du générateur aléatoire",
                    "schema": {
                        "type": "integer"
                    }, 
                    "default": 0
                },
                 {
                    "name": "width",
                    "in": "query",
                    "description": "largeur de l'image",
                    "schema": {
                        "type": "integer"
                    }, 
                    "minimum": 256,
                    "maximum": 1024, 
                    "default": 512
                },

                 {
                    "name": "height",
                    "in": "query",
                    "description": "hauteur de l'image",
                    "schema": {
                        "type": "integer"
                    }, 
                    "minimum": 256,
                    "maximum": 1024, 
                    "default": 512
                },
                {
                    "name": "guidance_scale",
                    "in": "query",
                    "description": "permet de contrôler la correspondance entre le texte et l'image.",
                    "schema": {
                        "type": "number",
                        "format": "float"
                    }, 
                    "minimum": 3,
                    "maximum": 20, 
                    "default": 7.5
                },
                {
                    "name": "num_inference_steps",
                    "in": "query",
                    "description": "nombre d'iterations",
                    "schema": {
                        "type": "integer",
                    }, 
                    "minimum": 10,
                    "maximum": 100,
                    "default": 50
                },
                {
                    "name": "num_images",
                    "in": "query",
                    "description": "nombre d'images",
                    "schema": {
                        "type": "integer"
                    }, 
                    "minimum": 1, 
                    "maximum": 4, 
                    "default": 1
                },
                {
                    "name": "source_language",
                    "in": "query",
                    "description": "langue source",
                    "schema": {
                        "type": "text",
                        "enum" : list(MAP_SOURCE_LANGUAGE2CODE.keys())
                    }, 
                    "default": "french"
                },
                {
                    "name": "language_query",
                    "in": "query",
                    "description": "description du query en langage naturel",
                    "schema": {
                        "type": "text",
                        "default": "un robot qui mange des chocolats"
                    }
                }
            ],
            "responses": {
                "200": {
                    "description": "Réponse normale",
                    "content": {
                        "image/jpg": {}, 
                        "application/json": {
                            "schema": {
                                "type": "string"
                            }
                        }
                    }
                }
            }
        }
    }

}


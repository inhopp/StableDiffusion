# StableDiffusion
StableDiffusion from scratch (pytorch lightning)



<br>

![Untitled](https://github.com/inhopp/inhopp/assets/96368476/20f0c76f-6e08-47a9-b17e-138ea34f63fd){: width="50%" height="60%"}

<br>

## Repository Directory 

``` python 
├── StableDiffusion
        ├── data
        ├── Attention.py
        ├── CLIP.py
        ├── DDPM.py
        ├── Decoder.py
        ├── Diffusion.py
        ├── Encoder.py
        ├── Inference.py
        ├── ModelConverter.py
        ├── ModelLoader.py
        ├── Pipeline.py
        ├── requirments.txt
        └── README.md
```

- `data` : Model Weights, CLIP Toknizer
- `DDPM.py` : DDPM Sampler
- `Encoder/Decoder` : VAE Encoder/Decoder
- `ModelConverter.py` : load from standard weights
- `ModelLoader.py` : Construct StableDiffusion
- `diffusion.py` : Diffusion Forward/Backward Process
- `Pipeline.py` : Generating Pipeline

<br>

## Tutoral

### Clone repo and install model and depenency

> Download vocab.json and merges.txt from https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/tokenizer and save them in the data folder <br>Download v1-5-pruned-emaonly.ckpt from https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main and save it in the data folder

``` python
# Clone this repo and install dependency
git clone https://github.com/inhopp/DDPM.git
pip install -r "DDPM/requirments.txt"
```

<br>


### Inference.py

``` python
# Text to Image
prompt = "a dog, fullmetal, robotic, mechanical parts, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
uncond_prompt = ""  # Also known as negative prompt
do_cfg = True
cfg_scale = 8  # min: 1, max: 14

# Image to Image
input_image = None
image_path = "./images/dog.jpg"
strength = 0.9

sampler = "ddpm"
num_inference_steps = 50
seed = 42
```



<br>

### Main Reference
https://github.com/hkproj/pytorch-stable-diffusion
import ModelLoader
import Pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch

DEVICE = "cuda"

tokenizer = CLIPTokenizer("./data/vocab.json", merges_file="./data/merges.txt")
model_file = "./data/v1-5-pruned-emaonly.ckpt"
models = ModelLoader.preload_models_from_standard_weights(model_file, DEVICE)

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

output_image = Pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    device=DEVICE,
    idle_device="cpu",
    tokenizer=tokenizer,
)

output_image = Image.fromarray(output_image)
output_image.save("output.jpg", 'JPEG')
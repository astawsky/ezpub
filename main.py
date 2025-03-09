from diffusers import AutoPipelineForText2Image #, AutoPipelineForImage2Image, StableDiffusionXLPipeline
from diffusers.utils import load_image, make_image_grid
import torch

print("Torch version:",torch.__version__)

print("Is CUDA enabled?",torch.cuda.is_available())

print()

pipeline = AutoPipelineForText2Image.from_pretrained(
	"stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
# pipeline = StableDiffusionXLPipeline.from_single_file(
#     "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors",
#     torch_dtype=torch.float16
# ).to("cuda")

image = pipeline(
	"stained glass of darth vader, backlight, centered composition, masterpiece, photorealistic, 8k"
).images[0]

# prompt = "stained glass of darth vader, backlight, centered composition, masterpiece, photorealistic, 8k"
# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"
# init_image = load_image(url)

# image = pipeline(
# 	prompt = "stained glass of darth vader, backlight, centered composition, masterpiece, photorealistic, 8k"
#     , image=init_image, strength=0.8, guidance_scale=10.5
# ).images[0]

# make_image_grid([init_image, image], rows=1, cols=2)

print(image)

print()